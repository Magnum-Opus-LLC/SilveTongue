"""
Silvertongue Dashboard - Modular Version (app_dev.py)

This is the refactored version of app.py using Flask Blueprints for modular organization.
Use this for testing the new modular implementation before migrating app.py.

Usage:
    python app_dev.py
    # Or
    python run_dashboard.py --dev
"""

from src.routes import register_all_blueprints
from flask import Flask
import sys
from pathlib import Path

# Setup paths - dashboard first, then project root
dashboard_root = Path(__file__).parent
project_root = dashboard_root.parent

# Insert dashboard path FIRST so dashboard's src/ takes precedence over project's src/
sys.path.insert(0, str(dashboard_root))
sys.path.insert(1, str(project_root))


# Import blueprint registration (from dashboard/src/routes)

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    """Add headers to prevent browser caching."""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Register all route blueprints
register_all_blueprints(app)


if __name__ == '__main__':
    print("=" * 60)
    print("Silvertongue Dashboard - MODULAR VERSION (app_dev.py)")
    print("=" * 60)
    print()
    print("This is the refactored modular version using Flask Blueprints.")
    print("Registered blueprints:")
    print("  - core_bp: /api/actors, /api/actor/<name>, /api/config")
    print("  - chat_bp: /api/actor/<name>/chat, /api/actor/<name>/chat/history")
    print("  - knowledge_bp: /api/actor/<name>/knowledge/*, /api/actor/<name>/archive/*")
    print("  - learning_bp: /api/actor/<name>/learn, /api/loop/*, /api/tasks/*")
    print("  - sleep_bp: /api/actor/<name>/sleep, /api/actor/<name>/datasets/merge")
    print("  - pdf_cache_bp: /api/pdf-cache/*, /api/pdf/file/<filename>")
    print("  - trainer_bp: /api/trainer/*, /api/actor/<name>/train")
    print("  - adapters_bp: /api/adapters/*")
    print("  - actor_creation_bp: /api/actor/create, /api/actor/create/manual")
    print()
    print("Starting server on http://localhost:5045")
    print("=" * 60)

    app.run(debug=True, port=5045, threaded=True)
