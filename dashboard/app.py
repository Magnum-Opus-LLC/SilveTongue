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
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            🧠 SILVERTONGUE DASHBOARD (Modular)               ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Using Flask Blueprints for modular organization            ║")
    print("║  PDF Book system with validation & duplicate detection      ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Blueprints:                                                 ║")
    print("║    • core_bp: actors, config                                 ║")
    print("║    • chat_bp: chat, history                                  ║")
    print("║    • knowledge_bp: knowledge, archive                        ║")
    print("║    • learning_bp: learn, loops, tasks                        ║")
    print("║    • sleep_bp: sleep, datasets                               ║")
    print("║    • pdf_cache_bp: PDF library with validation               ║")
    print("║    • trainer_bp: LoRA training                               ║")
    print("║    • adapters_bp: adapter management                         ║")
    print("║    • actor_creation_bp: create actors                        ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Starting server on http://localhost:5050                    ║")
    print("║  Press Ctrl+C to stop                                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    app.run(debug=True, port=5050, threaded=True)
