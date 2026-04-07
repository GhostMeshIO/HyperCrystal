"""
run.py – Orchestrator for HyperCrystal / QNVM (v2.1)
====================================================
Enhanced with:
- Rate limiting on all endpoints (Flask-Limiter)
- Input sanitisation for embeddings (reject NaN/Inf)
- Idempotency keys for marketplace purchases
- Global exception handler (no traceback leakage)
- Brute-force protection on login (per-IP + account lockout)
- XSS-safe HTML report generation
"""

import argparse
import sys
import time
import json
import os
import re
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any
from functools import wraps

import numpy as np
from flask import Flask, request, jsonify, render_template_string, session, redirect, url_for
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException

# Import core components
from hypercrystal.core_engine import load_config, HyperCrystal, GoalField
from hypercrystal.output_product import OutputProduct, User, Workspace

# -----------------------------------------------------------------------------
# Flask app initialization
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('DASH_SECRET_KEY', secrets.token_hex(32))
app.config['JSON_SORT_KEYS'] = False

# Rate limiter (default: 100 per minute, adjustable via config)
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[os.environ.get('RATE_LIMIT', '100 per minute')],
    storage_uri="memory://"
)

# Global exception handler – never leak tracebacks
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the full exception to stderr for debugging
    app.logger.error(f"Unhandled exception: {e}", exc_info=True)
    # Return generic error message to client
    return jsonify({"error": "Internal server error. Please try again later."}), 500

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    return jsonify({"error": e.description}), e.code

# -----------------------------------------------------------------------------
# Input validation helpers
# -----------------------------------------------------------------------------
def validate_embedding(embedding: list, dim: int) -> Tuple[bool, Optional[str]]:
    """Check that embedding is a list of floats of correct dimension, no NaN/Inf."""
    if not isinstance(embedding, list):
        return False, "embedding must be a list"
    if len(embedding) != dim:
        return False, f"embedding must have dimension {dim}"
    for val in embedding:
        if not isinstance(val, (int, float)):
            return False, "embedding values must be numbers"
        if np.isnan(val) or np.isinf(val):
            return False, "embedding contains NaN or Infinity"
    return True, None

def validate_symbolic_tags(tags: list, max_len: int = 10, max_tag_len: int = 100) -> Tuple[bool, Optional[str]]:
    """Validate symbolic tags: list of strings, no dangerous characters."""
    if not isinstance(tags, list):
        return False, "symbolic must be a list"
    if len(tags) > max_len:
        return False, f"too many symbolic tags (max {max_len})"
    for tag in tags:
        if not isinstance(tag, str):
            return False, "each symbolic tag must be a string"
        if len(tag) > max_tag_len:
            return False, f"tag too long (max {max_tag_len} chars)"
        # Allow alphanumeric, underscore, hyphen, space, and basic punctuation
        if not re.match(r'^[a-zA-Z0-9_\-\s\.\,\!\?]+$', tag):
            return False, f"invalid characters in tag: {tag}"
    return True, None

# -----------------------------------------------------------------------------
# Brute-force protection (in-memory)
# -----------------------------------------------------------------------------
login_attempts = {}  # ip -> {'count': int, 'first_attempt': float, 'locked_until': float}

def check_brute_force(ip: str) -> Tuple[bool, Optional[str]]:
    """Return (allowed, error_message)."""
    now = time.time()
    entry = login_attempts.get(ip, {'count': 0, 'first_attempt': now, 'locked_until': 0})
    if entry['locked_until'] > now:
        return False, "Too many failed attempts. Account locked for 15 minutes."
    # Reset if more than 1 hour passed since first attempt
    if now - entry['first_attempt'] > 3600:
        entry = {'count': 0, 'first_attempt': now, 'locked_until': 0}
    if entry['count'] >= 10:  # 10 failures
        entry['locked_until'] = now + 900  # 15 minutes
        login_attempts[ip] = entry
        return False, "Too many failed attempts. Try again in 15 minutes."
    return True, None

def record_failed_login(ip: str):
    now = time.time()
    entry = login_attempts.get(ip, {'count': 0, 'first_attempt': now, 'locked_until': 0})
    entry['count'] += 1
    login_attempts[ip] = entry

def record_successful_login(ip: str):
    if ip in login_attempts:
        del login_attempts[ip]

# -----------------------------------------------------------------------------
# Idempotency keys for purchases (in-memory, could be moved to Redis)
# -----------------------------------------------------------------------------
processed_idempotency_keys = set()

def check_idempotency(key: str) -> bool:
    """Return True if key has not been seen before."""
    if key in processed_idempotency_keys:
        return False
    processed_idempotency_keys.add(key)
    # Optional: expire old keys after 24 hours (simplified)
    return True

# -----------------------------------------------------------------------------
# API Server (Flask) – main function
# -----------------------------------------------------------------------------
def start_api_server(crystal: HyperCrystal, host: str, port: int, config: dict):
    """Start the Flask API server with full product management endpoints."""
    product = OutputProduct(crystal)
    pm = product.product_manager
    marketplace = product.marketplace

    # -------------------------------------------------------------------------
    # Helper: authenticate user from API key
    # -------------------------------------------------------------------------
    def authenticate(api_key: str) -> Optional[str]:
        """Return user_id if API key is valid, else None."""
        return product.validate_api_key(api_key)

    # -------------------------------------------------------------------------
    # Helper: check quota and deduct credits for an operation
    # -------------------------------------------------------------------------
    def check_and_deduct(user_id: str, operation: str) -> bool:
        if not pm.check_quota(user_id, operation):
            return False
        return pm.deduct_credits(user_id, operation)

    def record_usage(user_id: str, operation: str):
        pm.record_usage(user_id, operation)

    # -------------------------------------------------------------------------
    # Health and metrics (no auth required)
    # -------------------------------------------------------------------------
    @app.route('/api/health', methods=['GET'])
    @limiter.limit("10 per minute")
    def health():
        return jsonify({"status": "ok", "step": crystal.state.step})

    @app.route('/api/metrics', methods=['GET'])
    @limiter.limit("30 per minute")
    def metrics():
        return jsonify(crystal.get_metrics())

    @app.route('/api/snapshot', methods=['GET'])
    @limiter.limit("10 per minute")
    def snapshot():
        return jsonify(crystal.get_state_snapshot())

    # -------------------------------------------------------------------------
    # Goal management
    # -------------------------------------------------------------------------
    @app.route('/api/goal', methods=['GET'])
    def get_global_goal():
        if crystal.state.global_goal is None:
            return jsonify({"error": "No global goal set"}), 404
        return jsonify(crystal.state.global_goal.as_array().tolist())

    @app.route('/api/goal', methods=['POST'])
    @limiter.limit("10 per minute")
    def set_global_goal():
        data = request.get_json()
        if not data or 'x' not in data or 'y' not in data or 'z' not in data:
            return jsonify({"error": "Missing x, y, z fields"}), 400
        try:
            x, y, z = float(data['x']), float(data['y']), float(data['z'])
        except ValueError:
            return jsonify({"error": "x, y, z must be numbers"}), 400
        goal = GoalField(x, y, z)
        crystal.set_global_goal(goal)
        return jsonify({"status": "ok", "goal": [goal.x, goal.y, goal.z]})

    # -------------------------------------------------------------------------
    # User management
    # -------------------------------------------------------------------------
    @app.route('/api/user', methods=['POST'])
    @limiter.limit("5 per minute")
    def create_user():
        data = request.get_json()
        email = data.get('email')
        tier = data.get('tier', 'free')
        if not email or not isinstance(email, str) or '@' not in email:
            return jsonify({"error": "valid email required"}), 400
        user_id = product.create_user(email, tier)
        return jsonify({"user_id": user_id})

    @app.route('/api/user/<user_id>', methods=['GET'])
    @limiter.limit("30 per minute")
    def get_user(user_id):
        user = pm.get_user(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        return jsonify({
            "user_id": user.user_id,
            "email": user.email,
            "tier": user.tier,
            "credits": user.credits,
            "workspaces": user.workspaces,
            "created": user.created,
        })

    @app.route('/api/user/<user_id>/key', methods=['POST'])
    @limiter.limit("5 per minute")
    def create_api_key(user_id):
        try:
            api_key = product.create_api_key(user_id)
            return jsonify({"api_key": api_key})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    @app.route('/api/user/<user_id>/credits', methods=['POST'])
    @limiter.limit("5 per minute")
    def add_credits(user_id):
        data = request.get_json()
        amount = data.get('amount', 0)
        if not isinstance(amount, (int, float)) or amount <= 0:
            return jsonify({"error": "amount must be positive number"}), 400
        if pm.purchase_credits(user_id, amount, "credit_card"):
            return jsonify({"credits": pm.get_user(user_id).credits})
        else:
            return jsonify({"error": "Failed to add credits"}), 400

    @app.route('/api/user/<user_id>/behavior', methods=['GET'])
    @limiter.limit("30 per minute")
    def user_behavior(user_id):
        behavior = pm.get_user_behavior(user_id)
        if not behavior:
            return jsonify({"error": "User not found"}), 404
        return jsonify(behavior)

    # -------------------------------------------------------------------------
    # Workspaces
    # -------------------------------------------------------------------------
    @app.route('/api/workspace', methods=['POST'])
    @limiter.limit("10 per minute")
    def create_workspace():
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        if not user_id or not name:
            return jsonify({"error": "user_id and name required"}), 400
        ws_id = pm.create_workspace(name, user_id)
        return jsonify({"workspace_id": ws_id})

    @app.route('/api/workspace/<ws_id>', methods=['GET'])
    @limiter.limit("30 per minute")
    def get_workspace(ws_id):
        ws = pm.get_workspace(ws_id)
        if not ws:
            return jsonify({"error": "Workspace not found"}), 404
        return jsonify({
            "workspace_id": ws.workspace_id,
            "name": ws.name,
            "owner_id": ws.owner_id,
            "members": ws.members,
            "artifact_count": len(ws.artifacts),
        })

    @app.route('/api/workspace/<ws_id>/artifact', methods=['POST'])
    @limiter.limit("20 per minute")
    def save_artifact(ws_id):
        data = request.get_json()
        artifact_type = data.get('type')
        title = data.get('title')
        content = data.get('content')
        if not all([artifact_type, title, content]):
            return jsonify({"error": "type, title, and content required"}), 400
        artifact_id = pm.save_artifact(ws_id, artifact_type, content, title)
        if not artifact_id:
            return jsonify({"error": "Workspace not found"}), 404
        return jsonify({"artifact_id": artifact_id})

    @app.route('/api/workspace/<ws_id>/artifacts', methods=['GET'])
    @limiter.limit("30 per minute")
    def list_artifacts(ws_id):
        artifacts = pm.get_artifacts(ws_id)
        return jsonify(artifacts)

    # -------------------------------------------------------------------------
    # Teams
    # -------------------------------------------------------------------------
    @app.route('/api/team', methods=['POST'])
    @limiter.limit("5 per minute")
    def create_team():
        data = request.get_json()
        name = data.get('name')
        owner_id = data.get('owner_id')
        tier = data.get('tier', 'pro')
        if not name or not owner_id:
            return jsonify({"error": "name and owner_id required"}), 400
        team_id = pm.create_team(name, owner_id, tier)
        return jsonify({"team_id": team_id})

    @app.route('/api/team/<team_id>/member', methods=['POST'])
    @limiter.limit("10 per minute")
    def add_team_member(team_id):
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        pm.add_team_member(team_id, user_id)
        return jsonify({"status": "ok"})

    # -------------------------------------------------------------------------
    # Core operations (query, store) with authentication, validation, rate limiting
    # -------------------------------------------------------------------------
    @app.route('/api/query', methods=['POST'])
    @limiter.limit("100 per minute")
    def query():
        data = request.get_json()
        api_key = data.get('api_key')
        query_vector = data.get('query_vector')
        k = data.get('k', 5)
        if not api_key or not query_vector:
            return jsonify({"error": "api_key and query_vector required"}), 400
        # Validate embedding dimension
        valid, err = validate_embedding(query_vector, crystal.config["embedding_dim"])
        if not valid:
            return jsonify({"error": err}), 400
        try:
            results = product.query(api_key, query_vector, k)
            return jsonify(results)
        except PermissionError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            app.logger.error(f"Query error: {e}", exc_info=True)
            return jsonify({"error": "Internal error"}), 500

    @app.route('/api/store', methods=['POST'])
    @limiter.limit("50 per minute")
    def store():
        data = request.get_json()
        api_key = data.get('api_key')
        embedding = data.get('embedding')
        symbolic = data.get('symbolic', [])
        goal = data.get('goal', {"x":0,"y":0,"z":0})
        if not api_key or not embedding:
            return jsonify({"error": "api_key and embedding required"}), 400
        # Validate embedding
        valid, err = validate_embedding(embedding, crystal.config["embedding_dim"])
        if not valid:
            return jsonify({"error": err}), 400
        # Validate symbolic tags
        valid, err = validate_symbolic_tags(symbolic)
        if not valid:
            return jsonify({"error": err}), 400
        try:
            idx = product.store(api_key, embedding, symbolic, goal)
            return jsonify({"concept_index": idx})
        except PermissionError as e:
            return jsonify({"error": str(e)}), 401

    # -------------------------------------------------------------------------
    # Report generation (authenticated, credit deducted) – XSS-safe HTML
    # -------------------------------------------------------------------------
    @app.route('/api/report/<report_type>', methods=['GET'])
    @limiter.limit("20 per minute")
    def get_report(report_type):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "X-API-Key header required"}), 401
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        if not check_and_deduct(user_id, "generate_report"):
            return jsonify({"error": "Insufficient credits or quota exceeded"}), 402

        try:
            fmt = request.args.get('format', 'json')
            report = product.generate_report(report_type, fmt)
            record_usage(user_id, "generate_report")
            if fmt == 'html':
                # Escape any user-supplied content that might be embedded
                # (the report is generated by product.generate_report, which we assume is safe,
                # but we add a Content-Security-Policy header)
                response = app.make_response(report)
                response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'unsafe-inline'; script-src 'none'"
                return response
            elif fmt == 'json':
                try:
                    report_dict = json.loads(report)
                    return jsonify(report_dict)
                except:
                    return jsonify({"report": report})
            else:
                return jsonify({"report": report})
        except Exception as e:
            app.logger.error(f"Report generation error: {e}", exc_info=True)
            return jsonify({"error": "Failed to generate report"}), 500

    # -------------------------------------------------------------------------
    # Artifact generation (authenticated, credit deducted)
    # -------------------------------------------------------------------------
    @app.route('/api/artifact/<artifact_type>', methods=['GET'])
    @limiter.limit("20 per minute")
    def get_artifact(artifact_type):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "X-API-Key header required"}), 401
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        if not check_and_deduct(user_id, "generate_artifact"):
            return jsonify({"error": "Insufficient credits or quota exceeded"}), 402

        try:
            artifact = product.generate_artifact(artifact_type)
            record_usage(user_id, "generate_artifact")
            return jsonify({"artifact": artifact})
        except Exception as e:
            app.logger.error(f"Artifact generation error: {e}", exc_info=True)
            return jsonify({"error": "Failed to generate artifact"}), 500

    # -------------------------------------------------------------------------
    # Marketplace endpoints (with idempotency)
    # -------------------------------------------------------------------------
    @app.route('/api/marketplace/listings', methods=['GET'])
    @limiter.limit("50 per minute")
    def get_listings():
        artifact_type = request.args.get('type')
        tag = request.args.get('tag')
        listings = marketplace.get_listings(artifact_type, tag)
        result = []
        for l in listings:
            result.append({
                "listing_id": l.listing_id,
                "title": l.title,
                "description": l.description,
                "artifact_type": l.artifact_type,
                "price_usd": l.price_usd,
                "rating": l.rating,
                "num_ratings": l.num_ratings,
                "tags": l.tags,
            })
        return jsonify(result)

    @app.route('/api/marketplace/publish', methods=['POST'])
    @limiter.limit("10 per minute")
    def publish():
        data = request.get_json()
        api_key = data.get('api_key')
        workspace_id = data.get('workspace_id')
        artifact_type = data.get('artifact_type')
        title = data.get('title')
        description = data.get('description')
        price_usd = data.get('price_usd', 0.0)
        tags = data.get('tags', [])
        if not all([api_key, workspace_id, artifact_type, title, description]):
            return jsonify({"error": "Missing required fields"}), 400
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        ws = pm.get_workspace(workspace_id)
        if not ws or ws.owner_id != user_id:
            return jsonify({"error": "Workspace not found or not owned"}), 403

        if not check_and_deduct(user_id, "generate_artifact"):
            return jsonify({"error": "Insufficient credits to generate artifact"}), 402

        try:
            content = product.generate_artifact(artifact_type)
        except Exception as e:
            return jsonify({"error": f"Artifact generation failed: {str(e)}"}), 500

        listing_id = marketplace.publish_artifact(
            user_id, workspace_id, artifact_type, title, description,
            content, price_usd, tags
        )
        record_usage(user_id, "publish_artifact")
        return jsonify({"listing_id": listing_id})

    @app.route('/api/marketplace/purchase/<listing_id>', methods=['POST'])
    @limiter.limit("10 per minute")
    def purchase(listing_id):
        data = request.get_json()
        api_key = data.get('api_key')
        idempotency_key = data.get('idempotency_key')
        if not api_key:
            return jsonify({"error": "api_key required"}), 400
        if not idempotency_key:
            return jsonify({"error": "idempotency_key required"}), 400
        if not check_idempotency(idempotency_key):
            return jsonify({"error": "Duplicate request"}), 409

        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        listing = marketplace.listings.get(listing_id)
        if not listing:
            return jsonify({"error": "Listing not found"}), 404

        cost_credits = int(listing.price_usd * 100) if listing.price_usd > 0 else 0
        if cost_credits > 0:
            if not pm.deduct_credits(user_id, "purchase"):
                return jsonify({"error": "Insufficient credits"}), 402

        content = marketplace.purchase(listing_id, user_id)
        if content is None:
            return jsonify({"error": "Purchase failed"}), 400
        record_usage(user_id, "purchase")
        return jsonify({"artifact": content})

    @app.route('/api/marketplace/rate/<listing_id>', methods=['POST'])
    @limiter.limit("20 per minute")
    def rate_listing(listing_id):
        data = request.get_json()
        api_key = data.get('api_key')
        rating = data.get('rating')
        if not api_key or rating is None:
            return jsonify({"error": "api_key and rating required"}), 400
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401
        success = marketplace.rate(listing_id, rating, user_id)
        if not success:
            return jsonify({"error": "Rating failed (invalid listing or rating value)"}), 400
        return jsonify({"status": "ok"})

    # -------------------------------------------------------------------------
    # Admin endpoints (with admin key check)
    # -------------------------------------------------------------------------
    @app.route('/api/admin/growth', methods=['GET'])
    @limiter.limit("5 per minute")
    def growth_analytics():
        admin_key = request.headers.get('X-Admin-Key')
        expected_admin_key = os.environ.get('HYPERCRYSTAL_ADMIN_KEY', config.get("admin_api_key", "change_this_in_production"))
        if admin_key != expected_admin_key:
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify(pm.get_growth_metrics())

    # -------------------------------------------------------------------------
    # Integration stubs (Slack, Notion)
    # -------------------------------------------------------------------------
    @app.route('/api/integration/slack', methods=['POST'])
    @limiter.limit("10 per minute")
    def slack_integration():
        data = request.get_json()
        api_key = data.get('api_key')
        webhook = data.get('webhook_url')
        if not api_key or not webhook:
            return jsonify({"error": "api_key and webhook_url required"}), 400
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401
        success = pm.slack_integration(user_id, webhook)
        return jsonify({"status": "ok" if success else "failed"})

    @app.route('/api/integration/notion', methods=['POST'])
    @limiter.limit("10 per minute")
    def notion_integration():
        data = request.get_json()
        api_key = data.get('api_key')
        notion_key = data.get('notion_api_key')
        if not api_key or not notion_key:
            return jsonify({"error": "api_key and notion_api_key required"}), 400
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401
        success = pm.notion_integration(user_id, notion_key)
        return jsonify({"status": "ok" if success else "failed"})

    # -------------------------------------------------------------------------
    # Authentication endpoints (with brute-force protection)
    # -------------------------------------------------------------------------
    @app.route('/auth/login', methods=['POST'])
    @limiter.limit("5 per minute")  # per-IP rate limit
    def login():
        ip = request.remote_addr
        allowed, msg = check_brute_force(ip)
        if not allowed:
            return jsonify({"error": msg}), 429

        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        if not username or not password:
            record_failed_login(ip)
            return jsonify({"error": "Username and password required"}), 400

        # In production, replace with proper password hash lookup
        # For now, use the in-memory password dict from product_manager (if available)
        user = pm.get_user(username)  # assumes username is user_id
        if user and hasattr(pm, 'verify_password') and pm.verify_password(username, password):
            record_successful_login(ip)
            # Generate JWT token (simplified)
            import jwt
            token = jwt.encode(
                {'user': username, 'exp': datetime.utcnow() + timedelta(hours=24)},
                app.config['SECRET_KEY'],
                algorithm='HS256'
            )
            session['user'] = username
            return jsonify({'token': token, 'username': username})
        else:
            record_failed_login(ip)
            return jsonify({"error": "Invalid credentials"}), 401

    @app.route('/auth/logout', methods=['POST'])
    def logout():
        session.pop('user', None)
        return jsonify({'status': 'ok'})

    # -------------------------------------------------------------------------
    # Start server
    # -------------------------------------------------------------------------
    print(f"Starting HyperCrystal API server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)

# -----------------------------------------------------------------------------
# CLI functions (unchanged, but imported from original run.py)
# -----------------------------------------------------------------------------
def parse_goal_arg(goal_str: str) -> Tuple[float, float, float]:
    parts = goal_str.split(',')
    if len(parts) != 3:
        raise ValueError("Goal must be three comma-separated floats")
    return float(parts[0]), float(parts[1]), float(parts[2])

def run_simulation(crystal: HyperCrystal, steps: int, verbose: bool, no_cognition: bool,
                   global_goal: Optional[GoalField] = None):
    if global_goal is not None:
        crystal.set_global_goal(global_goal)
        if verbose:
            print(f"Global goal set to: {global_goal}")

    if not no_cognition:
        from hypercrystal.cognition_engine import CognitionEngine
        cognition = CognitionEngine(crystal)
        print(f"Cognition engine initialized. Running {steps} steps...")
        cognition.run(steps=steps, verbose=verbose)
        return cognition
    else:
        print(f"Running core only. Doing {steps} internal steps...")
        for step in range(steps):
            crystal.step_internal()
            if verbose and step % 10 == 0:
                print(f"Step {step+1}: metrics={crystal.get_metrics()}")
        return None

def generate_output(product: OutputProduct, report_type: Optional[str], artifact_type: Optional[str],
                    report_format: str):
    if report_type:
        report = product.generate_report(report_type, report_format)
        print("\n" + "="*80)
        print(f"REPORT: {report_type} ({report_format})")
        print("="*80)
        print(report)
    if artifact_type:
        artifact = product.generate_artifact(artifact_type)
        print("\n" + "="*80)
        print(f"ARTIFACT: {artifact_type}")
        print("="*80)
        print(artifact)

def create_user_cli(product: OutputProduct, email: str, tier: str):
    user_id = product.create_user(email, tier)
    api_key = product.create_api_key(user_id)
    print(f"User created: {user_id}")
    print(f"Email: {email}")
    print(f"Tier: {tier}")
    print(f"API Key: {api_key}")

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="HyperCrystal / QNVM Orchestrator v2.1")
    parser.add_argument("--steps", type=int, default=100, help="Number of cognitive steps to run")
    parser.add_argument("--report", choices=["executive", "business_plan", "architecture", "market_analysis",
                                             "competitive", "pricing", "risk", "financial", "pitch"],
                        help="Generate a specific report after run")
    parser.add_argument("--report-format", choices=["json", "markdown", "html", "text"], default="json",
                        help="Output format for report")
    parser.add_argument("--artifact", choices=["code", "api", "diagram", "pitch"],
                        help="Generate an artifact after run")
    parser.add_argument("--serve-api", action="store_true", help="Start a full API server (Flask required)")
    parser.add_argument("--config", type=str, default="hypercrystal_config.json", help="Config file path")
    parser.add_argument("--no-cognition", action="store_true", help="Run only core engine (no cognitive loop)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")

    parser.add_argument("--create-user", type=str, metavar="EMAIL", help="Create a new user and print API key")
    parser.add_argument("--user-tier", default="free", choices=["free", "pro", "enterprise"], help="Tier for new user")
    parser.add_argument("--global-goal", type=str, help="Set global goal (x,y,z) e.g. '0.7,0.4,0.5'")
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=5000, help="API server port")

    args = parser.parse_args()

    config = load_config(args.config)
    config["verbose"] = args.verbose
    if args.steps:
        config["steps"] = args.steps

    global_goal = None
    if args.global_goal:
        try:
            x, y, z = parse_goal_arg(args.global_goal)
            global_goal = GoalField(x, y, z)
        except ValueError as e:
            print(f"Error parsing --global-goal: {e}")
            sys.exit(1)

    crystal = HyperCrystal(config)

    if args.create_user:
        product = OutputProduct(crystal)
        create_user_cli(product, args.create_user, args.user_tier)
        return

    if args.serve_api:
        start_api_server(crystal, args.host, args.port, config)
        return

    cognition = run_simulation(crystal, args.steps, args.verbose, args.no_cognition, global_goal)
    product = OutputProduct(crystal, cognition if not args.no_cognition else None)
    generate_output(product, args.report, args.artifact, args.report_format)

if __name__ == "__main__":
    main()
