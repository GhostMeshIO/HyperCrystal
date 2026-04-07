"""
run.py – Orchestrator for HyperCrystal / QNVM (v2.0)
====================================================
Enhanced with full API server, user management, credit system, marketplace,
growth analytics, and all report/artifact generation endpoints.
"""

import argparse
import sys
import time
import json
import os
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

# Import core components
from hypercrystal.core_engine import load_config, HyperCrystal, GoalField
from hypercrystal.output_product import OutputProduct, User, Workspace

# For the API server
try:
    from flask import Flask, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# -----------------------------------------------------------------------------
# CLI Functions
# -----------------------------------------------------------------------------

def parse_goal_arg(goal_str: str) -> Tuple[float, float, float]:
    """Parse a string like '0.7,0.4,0.5' into (x, y, z)."""
    parts = goal_str.split(',')
    if len(parts) != 3:
        raise ValueError("Goal must be three comma-separated floats")
    return float(parts[0]), float(parts[1]), float(parts[2])

def run_simulation(crystal: HyperCrystal, steps: int, verbose: bool, no_cognition: bool,
                   global_goal: Optional[GoalField] = None):
    """Run the core simulation (with or without cognition)."""
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
    """Generate requested report and/or artifact."""
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
    """Create a user via CLI and print API key."""
    user_id = product.create_user(email, tier)
    api_key = product.create_api_key(user_id)
    print(f"User created: {user_id}")
    print(f"Email: {email}")
    print(f"Tier: {tier}")
    print(f"API Key: {api_key}")

# -----------------------------------------------------------------------------
# API Server (Flask)
# -----------------------------------------------------------------------------

def start_api_server(crystal: HyperCrystal, host: str, port: int, config: dict):
    """Start the Flask API server with full product management endpoints."""
    if not HAS_FLASK:
        print("Flask not installed. Please install with: pip install flask")
        sys.exit(1)

    product = OutputProduct(crystal)
    pm = product.product_manager
    marketplace = product.marketplace

    app = Flask(__name__)

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
        """Check quota, deduct credits, return True if allowed, else False."""
        if not pm.check_quota(user_id, operation):
            return False
        return pm.deduct_credits(user_id, operation)

    # -------------------------------------------------------------------------
    # Helper: record usage after successful operation
    # -------------------------------------------------------------------------
    def record_usage(user_id: str, operation: str):
        pm.record_usage(user_id, operation)

    # -------------------------------------------------------------------------
    # Health and metrics
    # -------------------------------------------------------------------------
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok", "step": crystal.state.step})

    @app.route('/api/metrics', methods=['GET'])
    def metrics():
        return jsonify(crystal.get_metrics())

    @app.route('/api/snapshot', methods=['GET'])
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
    def set_global_goal():
        data = request.get_json()
        if not data or 'x' not in data or 'y' not in data or 'z' not in data:
            return jsonify({"error": "Missing x, y, z fields"}), 400
        goal = GoalField(data['x'], data['y'], data['z'])
        crystal.set_global_goal(goal)
        return jsonify({"status": "ok", "goal": [goal.x, goal.y, goal.z]})

    # -------------------------------------------------------------------------
    # User management
    # -------------------------------------------------------------------------
    @app.route('/api/user', methods=['POST'])
    def create_user():
        data = request.get_json()
        email = data.get('email')
        tier = data.get('tier', 'free')
        if not email:
            return jsonify({"error": "email required"}), 400
        user_id = product.create_user(email, tier)
        return jsonify({"user_id": user_id})

    @app.route('/api/user/<user_id>', methods=['GET'])
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
    def create_api_key(user_id):
        try:
            api_key = product.create_api_key(user_id)
            return jsonify({"api_key": api_key})
        except ValueError as e:
            return jsonify({"error": str(e)}), 404

    @app.route('/api/user/<user_id>/credits', methods=['POST'])
    def add_credits(user_id):
        data = request.get_json()
        amount = data.get('amount', 0)
        if amount <= 0:
            return jsonify({"error": "amount must be positive"}), 400
        # Simulate payment
        if pm.purchase_credits(user_id, amount, "credit_card"):
            return jsonify({"credits": pm.get_user(user_id).credits})
        else:
            return jsonify({"error": "Failed to add credits"}), 400

    @app.route('/api/user/<user_id>/behavior', methods=['GET'])
    def user_behavior(user_id):
        behavior = pm.get_user_behavior(user_id)
        if not behavior:
            return jsonify({"error": "User not found"}), 404
        return jsonify(behavior)

    # -------------------------------------------------------------------------
    # Workspaces
    # -------------------------------------------------------------------------
    @app.route('/api/workspace', methods=['POST'])
    def create_workspace():
        data = request.get_json()
        user_id = data.get('user_id')
        name = data.get('name')
        if not user_id or not name:
            return jsonify({"error": "user_id and name required"}), 400
        ws_id = pm.create_workspace(name, user_id)
        return jsonify({"workspace_id": ws_id})

    @app.route('/api/workspace/<ws_id>', methods=['GET'])
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
    def list_artifacts(ws_id):
        artifacts = pm.get_artifacts(ws_id)
        return jsonify(artifacts)

    # -------------------------------------------------------------------------
    # Teams
    # -------------------------------------------------------------------------
    @app.route('/api/team', methods=['POST'])
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
    def add_team_member(team_id):
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
        pm.add_team_member(team_id, user_id)
        return jsonify({"status": "ok"})

    # -------------------------------------------------------------------------
    # Core operations (query, store) with authentication, rate limiting, credit deduction
    # -------------------------------------------------------------------------
    @app.route('/api/query', methods=['POST'])
    def query():
        data = request.get_json()
        api_key = data.get('api_key')
        query_vector = data.get('query_vector')
        k = data.get('k', 5)
        if not api_key or not query_vector:
            return jsonify({"error": "api_key and query_vector required"}), 400
        try:
            results = product.query(api_key, query_vector, k)
            return jsonify(results)
        except PermissionError as e:
            return jsonify({"error": str(e)}), 401
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route('/api/store', methods=['POST'])
    def store():
        data = request.get_json()
        api_key = data.get('api_key')
        embedding = data.get('embedding')
        symbolic = data.get('symbolic', [])
        goal = data.get('goal', {"x":0,"y":0,"z":0})
        if not api_key or not embedding:
            return jsonify({"error": "api_key and embedding required"}), 400
        try:
            idx = product.store(api_key, embedding, symbolic, goal)
            return jsonify({"concept_index": idx})
        except PermissionError as e:
            return jsonify({"error": str(e)}), 401

    # -------------------------------------------------------------------------
    # Report generation (authenticated, credit deducted)
    # -------------------------------------------------------------------------
    @app.route('/api/report/<report_type>', methods=['GET'])
    def get_report(report_type):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "X-API-Key header required"}), 401
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        # Check quota and deduct credits
        if not check_and_deduct(user_id, "generate_report"):
            return jsonify({"error": "Insufficient credits or quota exceeded"}), 402

        try:
            fmt = request.args.get('format', 'json')
            report = product.generate_report(report_type, fmt)
            # Record usage after successful generation
            record_usage(user_id, "generate_report")
            # If JSON, parse it for proper JSON response
            if fmt == 'json':
                try:
                    report_dict = json.loads(report)
                    return jsonify(report_dict)
                except:
                    # fallback to raw string
                    return jsonify({"report": report})
            else:
                return jsonify({"report": report})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------------
    # Artifact generation (authenticated, credit deducted)
    # -------------------------------------------------------------------------
    @app.route('/api/artifact/<artifact_type>', methods=['GET'])
    def get_artifact(artifact_type):
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({"error": "X-API-Key header required"}), 401
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        # Check quota and deduct credits
        if not check_and_deduct(user_id, "generate_artifact"):
            return jsonify({"error": "Insufficient credits or quota exceeded"}), 402

        try:
            artifact = product.generate_artifact(artifact_type)
            record_usage(user_id, "generate_artifact")
            return jsonify({"artifact": artifact})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------------------------------------------------------
    # Marketplace endpoints
    # -------------------------------------------------------------------------
    @app.route('/api/marketplace/listings', methods=['GET'])
    def get_listings():
        artifact_type = request.args.get('type')
        tag = request.args.get('tag')
        listings = marketplace.get_listings(artifact_type, tag)
        # Return limited info (exclude content)
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

        # Check workspace ownership
        ws = pm.get_workspace(workspace_id)
        if not ws or ws.owner_id != user_id:
            return jsonify({"error": "Workspace not found or not owned"}), 403

        # Deduct credits for generating artifact
        if not check_and_deduct(user_id, "generate_artifact"):
            return jsonify({"error": "Insufficient credits to generate artifact"}), 402

        # Generate artifact content
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
    def purchase(listing_id):
        data = request.get_json()
        api_key = data.get('api_key')
        if not api_key:
            return jsonify({"error": "api_key required"}), 400
        user_id = authenticate(api_key)
        if not user_id:
            return jsonify({"error": "Invalid API key"}), 401

        listing = marketplace.listings.get(listing_id)
        if not listing:
            return jsonify({"error": "Listing not found"}), 404

        cost_credits = int(listing.price_usd * 100) if listing.price_usd > 0 else 0
        if cost_credits > 0:
            # Deduct credits for purchase (special operation)
            if not pm.deduct_credits(user_id, "purchase"):
                return jsonify({"error": "Insufficient credits"}), 402

        content = marketplace.purchase(listing_id, user_id)
        if content is None:
            return jsonify({"error": "Purchase failed"}), 400
        record_usage(user_id, "purchase")
        return jsonify({"artifact": content})

    @app.route('/api/marketplace/rate/<listing_id>', methods=['POST'])
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
    # Growth analytics (admin only – simple check via environment variable)
    # -------------------------------------------------------------------------
    @app.route('/api/admin/growth', methods=['GET'])
    def growth_analytics():
        # Check admin API key from environment or config
        admin_key = request.headers.get('X-Admin-Key')
        expected_admin_key = os.environ.get('HYPERCRYSTAL_ADMIN_KEY', config.get("admin_api_key", "change_this_in_production"))
        if admin_key != expected_admin_key:
            return jsonify({"error": "Unauthorized"}), 401
        return jsonify(pm.get_growth_metrics())

    # -------------------------------------------------------------------------
    # Integration stubs (Slack, Notion)
    # -------------------------------------------------------------------------
    @app.route('/api/integration/slack', methods=['POST'])
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
    # Start server
    # -------------------------------------------------------------------------
    print(f"Starting HyperCrystal API server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)

# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HyperCrystal / QNVM Orchestrator v2.0")
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

    # User management CLI commands
    parser.add_argument("--create-user", type=str, metavar="EMAIL", help="Create a new user and print API key")
    parser.add_argument("--user-tier", default="free", choices=["free", "pro", "enterprise"], help="Tier for new user")

    # Global goal
    parser.add_argument("--global-goal", type=str, help="Set global goal (x,y,z) e.g. '0.7,0.4,0.5'")

    # API server options
    parser.add_argument("--host", default="0.0.0.0", help="API server host")
    parser.add_argument("--port", type=int, default=5000, help="API server port")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config["verbose"] = args.verbose
    if args.steps:
        config["steps"] = args.steps

    # Parse global goal if given
    global_goal = None
    if args.global_goal:
        try:
            x, y, z = parse_goal_arg(args.global_goal)
            global_goal = GoalField(x, y, z)
        except ValueError as e:
            print(f"Error parsing --global-goal: {e}")
            sys.exit(1)

    # Instantiate core
    crystal = HyperCrystal(config)

    # If --create-user is given, just create user and exit (no simulation)
    if args.create_user:
        product = OutputProduct(crystal)
        create_user_cli(product, args.create_user, args.user_tier)
        return

    # If --serve-api, start server and exit (no simulation)
    if args.serve_api:
        start_api_server(crystal, args.host, args.port, config)
        return

    # Otherwise, run simulation
    cognition = run_simulation(crystal, args.steps, args.verbose, args.no_cognition, global_goal)

    # After run, generate output if requested
    product = OutputProduct(crystal, cognition if not args.no_cognition else None)
    generate_output(product, args.report, args.artifact, args.report_format)

if __name__ == "__main__":
    main()
