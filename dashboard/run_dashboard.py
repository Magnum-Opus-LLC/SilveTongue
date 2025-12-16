#!/usr/bin/env python
"""
Quick launcher for Silvertongue Dashboard

Usage:
    python run_dashboard.py
    
Or with custom port:
    python run_dashboard.py --port 5050
"""

import argparse
import webbrowser
import threading
import time
from app import app


def open_browser(port):
    """Open browser after a short delay."""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{port}')


def main():
    parser = argparse.ArgumentParser(description='Run Silvertongue Dashboard')
    parser.add_argument('--port', type=int, default=5050, help='Port to run on (default: 5050)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
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
        threading.Thread(target=open_browser, args=(args.port,), daemon=True).start()
    
    app.run(
        host='0.0.0.0',
        port=args.port,
        debug=args.debug,
        threaded=True
    )


if __name__ == '__main__':
    main()
