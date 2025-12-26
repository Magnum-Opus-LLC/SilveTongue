"""
Silvertongue Dashboard - Modular Version

Flask app with Blueprint routes. Handles the src/ namespace conflict between
dashboard/src/ and project/src/ by using importlib for dashboard modules.
"""

from flask import Flask
import sys
from pathlib import Path
import importlib.util

# Setup paths - project root FIRST so project's src/ works for imports
dashboard_root = Path(__file__).parent
project_root = dashboard_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(dashboard_root) not in sys.path:
    sys.path.insert(1, str(dashboard_root))


# Load dashboard routes via importlib to avoid src/ conflict

def _load_module(name, path):
    """Load a module from a specific path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load the routes package
_routes_init = dashboard_root / 'src' / 'routes' / '__init__.py'
_routes_module = _load_module('dashboard_routes', _routes_init)
register_all_blueprints = _routes_module.register_all_blueprints

# Create Flask app
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
    print("║  Starting server on http://localhost:5050                    ║")
    print("║  Press Ctrl+C to stop                                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    app.run(debug=True, port=5050, threaded=True)
