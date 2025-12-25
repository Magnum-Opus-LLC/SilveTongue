"""
Silvertongue Dashboard - Modular source package.

This package contains modular components for the dashboard:
- utils: Configuration and utility functions
- pdf_stock: PDF cache stock management
- actors: Actor management functions
- routes/: Flask blueprint modules for API routes
"""

from pathlib import Path

# Project paths
DASHBOARD_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = DASHBOARD_ROOT.parent
