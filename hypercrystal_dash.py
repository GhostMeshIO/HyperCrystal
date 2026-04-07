"""
hypercrystal_dash.py – Web Dashboard for HyperCrystal / QNVM (v2.0)
====================================================================
Fixed widget layout, styling, and responsiveness.
"""

import os
import json
import time
import random
import numpy as np
import threading
from datetime import datetime, timedelta, timezone
from functools import wraps
from collections import deque

# Flask and extensions
from flask import Flask, render_template_string, jsonify, request, session, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import jwt
import secrets

# Core HyperCrystal modules
try:
    from core_engine import HyperCrystal, GoalField, load_config
    from cognition_engine import CognitionEngine
    from output_product import OutputProduct
except ImportError:
    # Fallback for running standalone – create dummy crystal
    class DummyCrystal:
        def __init__(self):
            self.state = type('State', (), {'step': 0, 'concepts': [], 'global_goal': None, 'config': {}})()
            self.config = {}
            self.get_metrics = lambda: {
                'sophia': 0.618, 'dark_wisdom': 0.3, 'paradox': 0.3, 'triple_point': 0.2,
                'holo_entropy': 0.5, 'concept_count': 0, 'memory_usage_mb': 0, 'avg_fitness': 0.5,
                'meta_depth': 1, 'non_hermitian': 0.35, 'betti_0': 0, 'betti_1': 0
            }
            self.get_state_snapshot = lambda: {'step': 0, 'concept_count': 0}
            self.retrieve_similar = lambda q, k: []
            self.store_concept = lambda c, g: 0
            self.set_global_goal = lambda g: None
            self.step_internal = lambda: None
            self._fast_novelty = lambda c: 0.5
    crystal = DummyCrystal()
    config = load_config() if 'load_config' in dir() else {}
    cognition = None
    product = None
else:
    config = load_config()
    crystal = HyperCrystal(config)
    cognition = CognitionEngine(crystal)
    product = OutputProduct(crystal, cognition)

# -----------------------------------------------------------------------------
# User database from users.json (do NOT overwrite later)
# -----------------------------------------------------------------------------
def load_users():
    users_file = os.path.join(os.path.dirname(__file__), 'users.json')
    default_users = {
        'admin': {'password': 'admin', 'role': 'admin', 'credits': 1000},
        'guest': {'password': 'guest', 'role': 'guest', 'credits': 100}
    }
    if not os.path.exists(users_file):
        with open(users_file, 'w') as f:
            json.dump(default_users, f, indent=2)
        return default_users
    try:
        with open(users_file, 'r') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return default_users
            return data
    except (json.JSONDecodeError, IOError):
        return default_users

users = load_users()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SECRET_KEY = os.environ.get('DASH_SECRET_KEY', secrets.token_hex(32))
JWT_SECRET = os.environ.get('JWT_SECRET', secrets.token_hex(32))
RATE_LIMIT = os.environ.get('RATE_LIMIT', "100 per minute")
DEBUG = os.environ.get('DEBUG', 'False') == 'True'

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

socketio = SocketIO(app, cors_allowed_origins="*")

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
    storage_uri="memory://"
)

# In‑memory storage
active_sessions = {}
concept_history = deque(maxlen=100)
history_lock = threading.Lock()

# Background metrics thread
def background_metrics():
    while True:
        socketio.sleep(1)
        metrics = crystal.get_metrics()
        socketio.emit('metrics', metrics, room='dashboard')
        if metrics['avg_fitness'] < 0.3 and metrics['step'] > 100:
            socketio.emit('alert', {'level': 'warning', 'message': 'Low average fitness!'}, room='dashboard')
        if metrics['paradox'] > 0.8:
            socketio.emit('alert', {'level': 'info', 'message': 'High paradox intensity – creative explosion!'}, room='dashboard')
        if metrics['triple_point'] > 0.5:
            socketio.emit('alert', {'level': 'warning', 'message': 'System far from triple point!'}, room='dashboard')

# -----------------------------------------------------------------------------
# Authentication helper
# -----------------------------------------------------------------------------
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
            request.user = data['user']
        except:
            return jsonify({'error': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route('/')
def index():
    if 'user' in session:
        return render_template_string(INDEX_HTML)
    else:
        return render_template_string(LOGIN_HTML)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username in users and users[username]['password'] == password:
        exp_time = datetime.now(timezone.utc) + timedelta(hours=24)
        token = jwt.encode(
            {'user': username, 'exp': exp_time},
            JWT_SECRET,
            algorithm='HS256'
        )
        session['user'] = username
        return jsonify({'token': token, 'username': username})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'status': 'ok'})

@app.route('/api/metrics')
@limiter.limit(RATE_LIMIT)
def api_metrics():
    return jsonify(crystal.get_metrics())

@app.route('/api/concepts')
@limiter.limit(RATE_LIMIT)
def api_concepts():
    concepts = crystal.state.concepts[:1000]
    out = []
    for i, c in enumerate(concepts):
        out.append({
            'id': i,
            'sophia': c.sophia_score,
            'dark_wisdom': c.dark_wisdom_density,
            'paradox': c.paradox_intensity,
            'symbolic': c.symbolic[:5],
            'embedding': c.subsymbolic.tolist()[:10]
        })
    return jsonify(out)

@app.route('/api/pareto')
@limiter.limit(RATE_LIMIT)
def api_pareto():
    front = crystal.state.concept_pareto_front
    points = []
    for idx in front:
        c = crystal.state.concepts[idx]
        points.append({
            'sophia': c.sophia_score,
            'dark_wisdom': c.dark_wisdom_density,
            'fitness': crystal.state.concept_fitness.get(idx, 0.5)
        })
    return jsonify(points)

@app.route('/api/search')
@limiter.limit(RATE_LIMIT)
def api_search():
    query = request.args.get('q', '')
    if not query:
        return jsonify([])
    matches = []
    for i, c in enumerate(crystal.state.concepts):
        if any(query.lower() in tag.lower() for tag in c.symbolic):
            matches.append({
                'id': i,
                'symbolic': c.symbolic,
                'sophia': c.sophia_score,
                'dark_wisdom': c.dark_wisdom_density,
                'paradox': c.paradox_intensity
            })
            if len(matches) >= 20:
                break
    return jsonify(matches)

@app.route('/api/embedding/<int:concept_id>')
@limiter.limit(RATE_LIMIT)
def api_embedding(concept_id):
    if 0 <= concept_id < len(crystal.state.concepts):
        emb = crystal.state.concepts[concept_id].subsymbolic.tolist()
        return jsonify(emb)
    else:
        return jsonify({'error': 'Concept not found'}), 404

@app.route('/api/goal', methods=['GET', 'POST'])
def api_goal():
    if request.method == 'GET':
        if crystal.state.global_goal:
            return jsonify(crystal.state.global_goal.as_array().tolist())
        else:
            return jsonify([0,0,0])
    else:
        data = request.get_json()
        x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
        crystal.set_global_goal(GoalField(x, y, z))
        return jsonify({'status': 'ok'})

@app.route('/api/generate', methods=['POST'])
def api_generate():
    if cognition:
        cognition.novelty_injector.step()
        return jsonify({'status': 'generated'})
    else:
        return jsonify({'error': 'Cognition not available'}), 500

@app.route('/api/llm_query', methods=['POST'])
def api_llm_query():
    data = request.get_json()
    query = data.get('query', '')
    if 'sophia' in query.lower():
        answer = f"Sophia score is {crystal.get_metrics()['sophia']:.3f}."
    elif 'paradox' in query.lower():
        answer = f"Paradox intensity is {crystal.get_metrics()['paradox']:.3f}."
    elif 'recommend' in query.lower():
        answer = "Consider increasing the global goal's dark_wisdom component to boost exploration."
    else:
        answer = "I'm not sure. Try asking about sophia, paradox, or recommendations."
    return jsonify({'answer': answer})

@app.route('/api/snapshot')
def api_snapshot():
    with history_lock:
        snapshot = {
            'step': crystal.state.step,
            'concepts': [c.subsymbolic.tolist() for c in crystal.state.concepts[:500]],
            'metrics': crystal.get_metrics()
        }
        concept_history.append(snapshot)
    return jsonify(snapshot)

@app.route('/api/history')
def api_history():
    with history_lock:
        return jsonify(list(concept_history))

@app.route('/api/recommend')
def api_recommend():
    metrics = crystal.get_metrics()
    rec = []
    if metrics['paradox'] < 0.3:
        rec.append("Increase paradox intensity to stimulate creativity.")
    if metrics['dark_wisdom'] < 0.4:
        rec.append("Boost dark wisdom by introducing contradictory concepts.")
    if metrics['triple_point'] > 0.3:
        rec.append("Focus on bringing system closer to triple point (sophia=0.618, dark_wisdom=0.3, paradox=0.3).")
    if metrics['avg_fitness'] < 0.5:
        rec.append("Improve average fitness by steering concepts toward current goal.")
    if not rec:
        rec.append("System is in good health. Consider exploring new goal directions.")
    return jsonify({'recommendations': rec})

# -----------------------------------------------------------------------------
# SocketIO events
# -----------------------------------------------------------------------------
@socketio.on('connect')
def handle_connect():
    join_room('dashboard')
    user = request.args.get('user', 'guest')
    active_sessions[request.sid] = user
    emit('connected', {'user': user})

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_sessions:
        del active_sessions[request.sid]
    leave_room('dashboard')

@socketio.on('set_goal')
def handle_set_goal(data):
    x = data.get('x', 0)
    y = data.get('y', 0)
    z = data.get('z', 0)
    crystal.set_global_goal(GoalField(x, y, z))
    emit('goal_updated', {'x': x, 'y': y, 'z': z}, room='dashboard')

@socketio.on('generate_concept')
def handle_generate():
    if cognition:
        cognition.novelty_injector.step()
        emit('concept_generated', {'message': 'New concept generated via diffusion'})
    else:
        emit('error', {'message': 'Cognition not available'})

@socketio.on('chat_message')
def handle_chat(data):
    user = active_sessions.get(request.sid, 'unknown')
    message = data.get('message', '')
    emit('chat', {'user': user, 'message': message, 'time': time.time()}, room='dashboard')

socketio.start_background_task(background_metrics)

# -----------------------------------------------------------------------------
# HTML Templates (embedded, with fixed layout and styling)
# -----------------------------------------------------------------------------
LOGIN_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>HyperCrystal Dashboard - Login</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
        .login-box { background: #16213e; padding: 40px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.5); width: 300px; }
        h2 { text-align: center; margin-bottom: 30px; }
        input { width: 100%; padding: 10px; margin: 10px 0; border: none; border-radius: 5px; background: #0f3460; color: #eee; }
        button { width: 100%; padding: 10px; background: #e94560; border: none; border-radius: 5px; color: white; font-weight: bold; cursor: pointer; }
        button:hover { background: #ff6b6b; }
        .error { color: #ff6b6b; text-align: center; margin-top: 10px; }
    </style>
</head>
<body>
    <div class="login-box">
        <h2>HyperCrystal Dashboard</h2>
        <form id="loginForm">
            <input type="text" id="username" placeholder="Username" required>
            <input type="password" id="password" placeholder="Password" required>
            <button type="submit">Login</button>
            <div id="error" class="error"></div>
        </form>
    </div>
    <script>
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const response = await fetch('/login', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({username, password})
            });
            if (response.ok) {
                const data = await response.json();
                localStorage.setItem('token', data.token);
                window.location.href = '/';
            } else {
                document.getElementById('error').innerText = 'Invalid credentials';
            }
        });
    </script>
</body>
</html>
"""

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>HyperCrystal Dashboard</title>
    <!-- External libraries -->
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/gridstack@7.2.3/dist/gridstack-all.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/gridstack@7.2.3/dist/gridstack.min.css" rel="stylesheet"/>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --text-color: #eee;
            --accent: #e94560;
            --accent-hover: #ff6b6b;
        }
        body.light {
            --bg-color: #f5f5f5;
            --card-bg: #ffffff;
            --text-color: #333;
            --accent: #e94560;
            --accent-hover: #ff6b6b;
        }
        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            transition: background 0.3s, color 0.3s;
        }
        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        h1 { margin: 0; }
        button, .btn {
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: background 0.2s;
        }
        button:hover, .btn:hover {
            background: var(--accent-hover);
        }
        /* GridStack custom styling */
        .grid-stack {
            background: transparent;
        }
        .grid-stack-item {
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s;
        }
        .grid-stack-item-content {
            padding: 12px;
            overflow-y: auto;
            height: 100%;
            box-sizing: border-box;
        }
        .grid-stack-item-content h3 {
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 1.1rem;
            border-bottom: 1px solid var(--accent);
            display: inline-block;
        }
        /* Metrics cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }
        .metric-card {
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 8px;
            text-align: center;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: bold;
            margin-top: 5px;
        }
        /* Chat box */
        #chat-box {
            height: 200px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 8px;
            margin-bottom: 8px;
        }
        .chat-message {
            margin: 5px 0;
            padding: 5px;
            border-bottom: 1px solid #555;
        }
        .chat-user {
            font-weight: bold;
            color: var(--accent);
        }
        /* Alert */
        .alert {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #ff6b6b;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            z-index: 1000;
            animation: fadeout 5s forwards;
        }
        @keyframes fadeout {
            0% { opacity: 1; }
            80% { opacity: 1; }
            100% { opacity: 0; visibility: hidden; }
        }
        .control-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        /* Responsive */
        @media (max-width: 768px) {
            body { padding: 10px; }
            .grid-stack-item-content { padding: 8px; }
            .metric-value { font-size: 1.2rem; }
        }
        /* Slider styling */
        input[type="range"] {
            width: 100%;
            margin: 8px 0;
        }
        #searchResults {
            max-height: 150px;
            overflow-y: auto;
            margin-top: 8px;
        }
    </style>
</head>
<body class="dark">
<div class="dashboard-header">
    <h1>HyperCrystal v2.0 Dashboard</h1>
    <div class="control-bar">
        <button id="themeToggle">🌓 Dark/Light</button>
        <button id="exportPdf">📄 Export PDF</button>
        <button id="deployBtn">🚀 Deploy to Cloud</button>
        <button id="logoutBtn">🚪 Logout</button>
        <span id="userDisplay" style="margin-left: 10px;"></span>
    </div>
</div>

<div class="grid-stack" id="dashboard-grid"></div>

<script>
    // -------------------------------------------------------------------------
    // Globals
    // -------------------------------------------------------------------------
    let socket = io();
    let grid = null;
    let threeScene, threeCamera, threeRenderer, threePoints, threeControls;
    let paretoChart = null;
    let goalSpaceChart = null;
    let fitnessChart = null;
    let currentUser = '';
    let theme = localStorage.getItem('theme') || 'dark';

    // -------------------------------------------------------------------------
    // Authentication
    // -------------------------------------------------------------------------
    const token = localStorage.getItem('token');
    if (!token) window.location.href = '/login';
    else {
        try {
            const payload = JSON.parse(atob(token.split('.')[1]));
            currentUser = payload.user;
            document.getElementById('userDisplay').innerText = `Logged in as: ${currentUser}`;
        } catch(e) { console.error(e); }
    }

    document.getElementById('logoutBtn').addEventListener('click', async () => {
        await fetch('/logout', {method: 'POST'});
        localStorage.removeItem('token');
        window.location.href = '/login';
    });

    // -------------------------------------------------------------------------
    // Theme toggle
    // -------------------------------------------------------------------------
    function setTheme(themeName) {
        document.body.className = themeName;
        localStorage.setItem('theme', themeName);
        theme = themeName;
    }
    if (theme === 'light') setTheme('light');
    document.getElementById('themeToggle').addEventListener('click', () => {
        setTheme(theme === 'dark' ? 'light' : 'dark');
    });

    // -------------------------------------------------------------------------
    // Widget definitions (with improved sizes and auto-position)
    // -------------------------------------------------------------------------
    const widgetDefinitions = [
        { id: 'metrics', title: 'Live Metrics', content: '<div id="metrics-container"></div>', w: 3, h: 2 },
        { id: '3dcloud', title: 'Concept Cloud (3D)', content: '<canvas id="threeCanvas" style="width:100%; height:100%; background:#0a0a1a;"></canvas>', w: 4, h: 3 },
        { id: 'pareto', title: 'Pareto Front', content: '<div id="paretoChart" style="width:100%; height:180px;"></div>', w: 3, h: 2 },
        { id: 'goalHeatmap', title: 'Goal Space Heatmap', content: '<div id="goalHeatmapChart" style="width:100%; height:180px;"></div>', w: 3, h: 2 },
        { id: 'fitness', title: 'Fitness Landscape', content: '<div id="fitnessChart" style="width:100%; height:180px;"></div>', w: 4, h: 2 },
        { id: 'search', title: 'Concept Search', content: '<input type="text" id="searchInput" placeholder="Search concepts..." style="width:100%;"><div id="searchResults"></div>', w: 3, h: 2 },
        { id: 'goalControl', title: 'Goal Vector', content: '<div id="goalSlider"></div><div id="goalCoords" style="margin:8px 0;"></div><button id="setGoalBtn">Set Goal</button>', w: 3, h: 2 },
        { id: 'chat', title: 'Live Chat', content: '<div id="chat-box"></div><div style="display:flex; gap:5px;"><input type="text" id="chatInput" placeholder="Message..." style="flex:1;"><button id="sendChat">Send</button></div>', w: 3, h: 2 },
        { id: 'generate', title: 'Concept Generation', content: '<button id="genConceptBtn">Generate New Concept (Diffusion)</button>', w: 2, h: 1 },
        { id: 'graph', title: 'Concept Graph', content: '<canvas id="graphCanvas" style="width:100%; height:200px;"></canvas>', w: 4, h: 2 },
        { id: 'recommend', title: 'AI Recommendations', content: '<ul id="recommendations" style="margin:0; padding-left:20px;"></ul><button id="refreshRec">Refresh</button>', w: 2, h: 2 },
        { id: 'history', title: 'Time Travel', content: '<button id="snapshotBtn">Take Snapshot</button> <button id="playbackBtn">Playback</button><div id="historyStatus" style="margin-top:8px;"></div>', w: 2, h: 1 },
        { id: 'llm', title: 'Ask LLM', content: '<input type="text" id="llmQuery" placeholder="Ask about system..." style="width:100%;"><button id="askLLM" style="margin-top:5px;">Ask</button><div id="llmAnswer" style="margin-top:8px;"></div>', w: 3, h: 2 }
    ];

    // -------------------------------------------------------------------------
    // Initialize GridStack with auto-positioning
    // -------------------------------------------------------------------------
    function initGrid() {
        const options = {
            cellHeight: 140,
            verticalMargin: 12,
            column: 12,
            disableResize: false,
            disableDrag: false,
            autoPosition: true   // Fixes overlapping
        };
        grid = GridStack.init(options, '#dashboard-grid');
        widgetDefinitions.forEach(w => {
            grid.addWidget({
                x: 0, y: 0, w: w.w, h: w.h,
                content: `<div class="grid-stack-item-content"><h3>${w.title}</h3>${w.content}</div>`,
                id: w.id
            });
        });
        // After grid built, initialize all widgets
        initMetricsWidget();
        init3DCloud();
        initPareto();
        initGoalHeatmap();
        initFitness();
        initSearch();
        initGoalControl();
        initChat();
        initGenerate();
        initGraph();
        initRecommendations();
        initHistory();
        initLLM();
    }

    // -------------------------------------------------------------------------
    // Metrics widget (grid layout inside)
    // -------------------------------------------------------------------------
    function initMetricsWidget() {
        const container = document.getElementById('metrics-container');
        container.innerHTML = `
            <div class="metrics-grid">
                <div class="metric-card"><div>Sophia</div><div class="metric-value" id="metric-sophia">0</div></div>
                <div class="metric-card"><div>Dark Wisdom</div><div class="metric-value" id="metric-dark">0</div></div>
                <div class="metric-card"><div>Paradox</div><div class="metric-value" id="metric-paradox">0</div></div>
                <div class="metric-card"><div>Triple Point</div><div class="metric-value" id="metric-triple">0</div></div>
                <div class="metric-card"><div>Concepts</div><div class="metric-value" id="metric-concepts">0</div></div>
                <div class="metric-card"><div>Avg Fitness</div><div class="metric-value" id="metric-fitness">0</div></div>
            </div>
        `;
    }

    // -------------------------------------------------------------------------
    // Real-time metrics via WebSocket
    // -------------------------------------------------------------------------
    socket.on('metrics', (data) => {
        document.getElementById('metric-sophia').innerText = data.sophia.toFixed(3);
        document.getElementById('metric-dark').innerText = data.dark_wisdom.toFixed(3);
        document.getElementById('metric-paradox').innerText = data.paradox.toFixed(3);
        document.getElementById('metric-triple').innerText = data.triple_point.toFixed(3);
        document.getElementById('metric-concepts').innerText = data.concept_count;
        document.getElementById('metric-fitness').innerText = data.avg_fitness.toFixed(3);
        refreshPareto();
        refreshFitness();
        refreshGoalHeatmap();
    });

    socket.on('alert', (data) => {
        showAlert(data.message);
    });
    function showAlert(msg) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert';
        alertDiv.innerText = msg;
        document.body.appendChild(alertDiv);
        setTimeout(() => alertDiv.remove(), 5000);
    }

    // -------------------------------------------------------------------------
    // 3D Concept Cloud (fixed canvas resizing)
    // -------------------------------------------------------------------------
    function init3DCloud() {
        const canvas = document.getElementById('threeCanvas');
        if (!canvas) return;
        const resize3D = () => {
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            if (width === 0 || height === 0) return;
            threeCamera.aspect = width / height;
            threeCamera.updateProjectionMatrix();
            threeRenderer.setSize(width, height);
        };
        threeScene = new THREE.Scene();
        threeCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
        threeCamera.position.set(2,2,2);
        threeCamera.lookAt(0,0,0);
        threeRenderer = new THREE.WebGLRenderer({ canvas });
        threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
        const ambientLight = new THREE.AmbientLight(0x404060);
        threeScene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        threeScene.add(directionalLight);

        fetch('/api/concepts')
            .then(res => res.json())
            .then(concepts => {
                const positions = concepts.map(c => new THREE.Vector3(c.embedding[0] || 0, c.embedding[1] || 0, c.embedding[2] || 0));
                const geometry = new THREE.BufferGeometry().setFromPoints(positions);
                const material = new THREE.PointsMaterial({ color: 0xe94560, size: 0.05 });
                threePoints = new THREE.Points(geometry, material);
                threeScene.add(threePoints);
                resize3D();
                window.addEventListener('resize', resize3D);
                animate3D();
            });
        function animate3D() {
            requestAnimationFrame(animate3D);
            if (threeControls) threeControls.update();
            if (threeRenderer && threeScene && threeCamera) {
                threeRenderer.render(threeScene, threeCamera);
            }
        }
    }

    // -------------------------------------------------------------------------
    // Pareto Front
    // -------------------------------------------------------------------------
    function initPareto() {
        const container = document.getElementById('paretoChart');
        paretoChart = d3.select("#paretoChart").append("svg").attr("width", "100%").attr("height", "100%");
        refreshPareto();
    }
    function refreshPareto() {
        fetch('/api/pareto')
            .then(res => res.json())
            .then(data => {
                const width = document.getElementById('paretoChart').clientWidth;
                const height = document.getElementById('paretoChart').clientHeight;
                paretoChart.selectAll("*").remove();
                paretoChart.attr("width", width).attr("height", height);
                const x = d3.scaleLinear().domain([0,1]).range([30, width-20]);
                const y = d3.scaleLinear().domain([0,1]).range([height-20, 20]);
                paretoChart.selectAll("circle")
                    .data(data)
                    .enter()
                    .append("circle")
                    .attr("cx", d => x(d.sophia))
                    .attr("cy", d => y(d.dark_wisdom))
                    .attr("r", 5)
                    .attr("fill", "var(--accent)");
                paretoChart.append("text").attr("x", width-60).attr("y", 20).text("Sophia →").attr("fill", "var(--text-color)").attr("font-size", "10px");
                paretoChart.append("text").attr("x", 10).attr("y", 20).text("Dark Wisdom ↑").attr("fill", "var(--text-color)").attr("font-size", "10px");
            });
    }

    // -------------------------------------------------------------------------
    // Goal Space Heatmap (simulated)
    // -------------------------------------------------------------------------
    function initGoalHeatmap() {
        goalSpaceChart = d3.select("#goalHeatmapChart").append("svg").attr("width", "100%").attr("height", "100%");
        refreshGoalHeatmap();
    }
    function refreshGoalHeatmap() {
        const width = document.getElementById('goalHeatmapChart').clientWidth;
        const height = document.getElementById('goalHeatmapChart').clientHeight;
        goalSpaceChart.selectAll("*").remove();
        goalSpaceChart.attr("width", width).attr("height", height);
        const data = Array.from({length: 20}, () => Array.from({length: 20}, () => Math.random()));
        const cellW = width/20, cellH = height/20;
        const color = d3.scaleSequential(d3.interpolateViridis).domain([0,1]);
        for (let i=0; i<20; i++) {
            for (let j=0; j<20; j++) {
                goalSpaceChart.append("rect")
                    .attr("x", i*cellW)
                    .attr("y", j*cellH)
                    .attr("width", cellW)
                    .attr("height", cellH)
                    .attr("fill", color(data[i][j]));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Fitness Landscape (scatter)
    // -------------------------------------------------------------------------
    function initFitness() {
        fitnessChart = d3.select("#fitnessChart").append("svg").attr("width", "100%").attr("height", "100%");
        refreshFitness();
    }
    function refreshFitness() {
        fetch('/api/pareto')
            .then(res => res.json())
            .then(data => {
                const width = document.getElementById('fitnessChart').clientWidth;
                const height = document.getElementById('fitnessChart').clientHeight;
                fitnessChart.selectAll("*").remove();
                fitnessChart.attr("width", width).attr("height", height);
                const x = d3.scaleLinear().domain([0,1]).range([30, width-20]);
                const y = d3.scaleLinear().domain([0,1]).range([height-20, 20]);
                fitnessChart.selectAll("circle")
                    .data(data)
                    .enter()
                    .append("circle")
                    .attr("cx", d => x(d.fitness))
                    .attr("cy", d => y(d.sophia))
                    .attr("r", 5)
                    .attr("fill", "var(--accent)");
                fitnessChart.append("text").attr("x", width-60).attr("y", 20).text("Fitness →").attr("fill", "var(--text-color)").attr("font-size", "10px");
                fitnessChart.append("text").attr("x", 10).attr("y", 20).text("Sophia ↑").attr("fill", "var(--text-color)").attr("font-size", "10px");
            });
    }

    // -------------------------------------------------------------------------
    // Search (autocomplete)
    // -------------------------------------------------------------------------
    function initSearch() {
        const input = document.getElementById('searchInput');
        const resultsDiv = document.getElementById('searchResults');
        let debounce;
        input.addEventListener('input', () => {
            clearTimeout(debounce);
            debounce = setTimeout(() => {
                const q = input.value;
                if (q.length < 2) { resultsDiv.innerHTML = ''; return; }
                fetch(`/api/search?q=${encodeURIComponent(q)}`)
                    .then(res => res.json())
                    .then(data => {
                        resultsDiv.innerHTML = data.map(c => `<div><strong>${c.symbolic.join(', ')}</strong><br>Sophia: ${c.sophia.toFixed(3)}, Dark: ${c.dark_wisdom.toFixed(3)}</div><hr>`).join('');
                    });
            }, 300);
        });
    }

    // -------------------------------------------------------------------------
    // Goal control with sliders
    // -------------------------------------------------------------------------
    function initGoalControl() {
        const container = document.getElementById('goalSlider');
        container.innerHTML = `
            <label>X (Sophia): <input type="range" id="goalX" min="0" max="1" step="0.01" value="0.618"></label>
            <label>Y (Dark Wisdom): <input type="range" id="goalY" min="0" max="1" step="0.01" value="0.3"></label>
            <label>Z (Paradox): <input type="range" id="goalZ" min="0" max="1" step="0.01" value="0.3"></label>
        `;
        const setBtn = document.getElementById('setGoalBtn');
        setBtn.addEventListener('click', () => {
            const x = parseFloat(document.getElementById('goalX').value);
            const y = parseFloat(document.getElementById('goalY').value);
            const z = parseFloat(document.getElementById('goalZ').value);
            socket.emit('set_goal', {x, y, z});
            document.getElementById('goalCoords').innerHTML = `Goal: (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;
        });
        fetch('/api/goal')
            .then(res => res.json())
            .then(goal => {
                document.getElementById('goalX').value = goal[0];
                document.getElementById('goalY').value = goal[1];
                document.getElementById('goalZ').value = goal[2];
                document.getElementById('goalCoords').innerHTML = `Goal: (${goal[0].toFixed(2)}, ${goal[1].toFixed(2)}, ${goal[2].toFixed(2)})`;
            });
    }

    // -------------------------------------------------------------------------
    // Chat
    // -------------------------------------------------------------------------
    function initChat() {
        const chatBox = document.getElementById('chat-box');
        const input = document.getElementById('chatInput');
        const sendBtn = document.getElementById('sendChat');
        sendBtn.addEventListener('click', () => {
            const msg = input.value.trim();
            if (msg) {
                socket.emit('chat_message', {message: msg});
                input.value = '';
            }
        });
        socket.on('chat', (data) => {
            const div = document.createElement('div');
            div.className = 'chat-message';
            div.innerHTML = `<span class="chat-user">${data.user}</span>: ${data.message}`;
            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        });
    }

    // -------------------------------------------------------------------------
    // Generate concept
    // -------------------------------------------------------------------------
    function initGenerate() {
        const btn = document.getElementById('genConceptBtn');
        btn.addEventListener('click', () => {
            fetch('/api/generate', {method: 'POST'})
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'generated') {
                        showAlert('New concept generated via diffusion!');
                        setTimeout(() => { refreshPareto(); refreshFitness(); refreshGoalHeatmap(); }, 500);
                    } else {
                        showAlert('Error generating concept');
                    }
                });
        });
        socket.on('concept_generated', (data) => { showAlert(data.message); });
    }

    // -------------------------------------------------------------------------
    // Concept graph placeholder
    // -------------------------------------------------------------------------
    function initGraph() {
        const canvas = document.getElementById('graphCanvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const resize = () => {
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
            ctx.fillStyle = 'var(--card-bg)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'var(--text-color)';
            ctx.font = '14px sans-serif';
            ctx.fillText('Concept graph (force-directed placeholder)', 20, 40);
        };
        window.addEventListener('resize', resize);
        resize();
    }

    // -------------------------------------------------------------------------
    // AI Recommendations
    // -------------------------------------------------------------------------
    function initRecommendations() {
        const ul = document.getElementById('recommendations');
        const btn = document.getElementById('refreshRec');
        function loadRec() {
            fetch('/api/recommend')
                .then(res => res.json())
                .then(data => {
                    ul.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
                });
        }
        btn.addEventListener('click', loadRec);
        loadRec();
    }

    // -------------------------------------------------------------------------
    // Time travel history
    // -------------------------------------------------------------------------
    function initHistory() {
        const snapshotBtn = document.getElementById('snapshotBtn');
        const playbackBtn = document.getElementById('playbackBtn');
        const statusDiv = document.getElementById('historyStatus');
        snapshotBtn.addEventListener('click', async () => {
            const res = await fetch('/api/snapshot');
            const data = await res.json();
            statusDiv.innerHTML = `Snapshot taken at step ${data.step}`;
        });
        playbackBtn.addEventListener('click', async () => {
            const res = await fetch('/api/history');
            const history = await res.json();
            statusDiv.innerHTML = `Playback not fully implemented. ${history.length} snapshots available.`;
        });
    }

    // -------------------------------------------------------------------------
    // LLM query
    // -------------------------------------------------------------------------
    function initLLM() {
        const queryInput = document.getElementById('llmQuery');
        const askBtn = document.getElementById('askLLM');
        const answerDiv = document.getElementById('llmAnswer');
        askBtn.addEventListener('click', async () => {
            const query = queryInput.value.trim();
            if (!query) return;
            const res = await fetch('/api/llm_query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query})
            });
            const data = await res.json();
            answerDiv.innerHTML = data.answer;
        });
    }

    // -------------------------------------------------------------------------
    // Export PDF
    // -------------------------------------------------------------------------
    document.getElementById('exportPdf').addEventListener('click', async () => {
        const element = document.getElementById('dashboard-grid');
        const canvas = await html2canvas(element);
        const imgData = canvas.toDataURL('image/png');
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF();
        pdf.addImage(imgData, 'PNG', 0, 0, 210, 297);
        pdf.save('hypercrystal_report.pdf');
    });

    // -------------------------------------------------------------------------
    // Simulated deploy
    // -------------------------------------------------------------------------
    document.getElementById('deployBtn').addEventListener('click', () => {
        showAlert('Deploying to cloud... (simulated)');
    });

    // Start everything after DOM is ready
    document.addEventListener('DOMContentLoaded', initGrid);
</script>
</body>
</html>
"""

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("Starting HyperCrystal Dashboard...")
    print("Access at http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=DEBUG)
