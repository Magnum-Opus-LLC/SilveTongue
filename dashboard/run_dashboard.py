#!/usr/bin/env python
"""
Quick launcher for Silvertongue Dashboard

Usage:
    python run_dashboard.py
    
Or with custom port:
    python run_dashboard.py --port 5050
    
Use legacy app.py:
    python run_dashboard.py --legacy
"""

# Setup sys.path FIRST before any other imports
import sys
from pathlib import Path

dashboard_root = Path(__file__).parent
project_root = dashboard_root.parent

# Project root must be FIRST for project's src/ imports to work
# Dashboard root second for dashboard's own modules
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(dashboard_root))

import argparse
import webbrowser
import threading
import time

# Now import app
from app import app


def open_browser(port):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')


def main():
    parser = argparse.ArgumentParser(description='Run Silvertongue Dashboard')
    parser.add_argument('--port', type=int, default=5050,
                        help='Port to run on (default: 5050)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not open browser automatically')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║            🧠 SILVERTONGUE DASHBOARD                          ║
╠══════════════════════════════════════════════════════════════╣
║  Starting server on http://localhost:{args.port:<5}                  ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    if not args.no_browser:
        threading.Thread(target=open_browser, args=(
            args.port,), daemon=True).start()

    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
