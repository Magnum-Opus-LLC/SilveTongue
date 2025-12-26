"""
Flask Blueprint routes for the Silvertongue Dashboard.

Each module provides a Blueprint that handles a specific category of routes.
"""

import sys
from pathlib import Path
import importlib.util

# Get directories
_routes_dir = Path(__file__).parent
_src_dir = _routes_dir.parent


def _load_module(name, path):
    """Load a module from a specific path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Preload dashboard's src modules so relative imports work
# These get registered as the parent package for route modules
_load_module('dashboard', _src_dir / '__init__.py')
_load_module('dashboard.utils', _src_dir / 'utils.py')
_load_module('dashboard.actors', _src_dir / 'actors.py')
_load_module('dashboard.pdf_book', _src_dir / 'pdf_book.py')
_load_module('dashboard.pdf_stock', _src_dir / 'pdf_stock.py')

# Create a fake 'dashboard.routes' package entry
sys.modules['dashboard.routes'] = sys.modules[__name__]


def _load_route_module(name):
    """Load a route module from the routes directory."""
    module_path = _routes_dir / f'{name}.py'
    module_name = f'dashboard.routes.{name}'
    return _load_module(module_name, module_path)


# Load all route modules
_core = _load_route_module('core')
_chat = _load_route_module('chat')
_knowledge = _load_route_module('knowledge')
_learning = _load_route_module('learning')
_sleep = _load_route_module('sleep')
_pdf_cache = _load_route_module('pdf_cache')
_trainer = _load_route_module('trainer')
_adapters = _load_route_module('adapters')
_actor_creation = _load_route_module('actor_creation')

# Export blueprints
core_bp = _core.core_bp
chat_bp = _chat.chat_bp
knowledge_bp = _knowledge.knowledge_bp
learning_bp = _learning.learning_bp
sleep_bp = _sleep.sleep_bp
pdf_cache_bp = _pdf_cache.pdf_cache_bp
trainer_bp = _trainer.trainer_bp
adapters_bp = _adapters.adapters_bp
actor_creation_bp = _actor_creation.actor_creation_bp


def register_all_blueprints(app):
    """Register all blueprints with the Flask app."""
    app.register_blueprint(core_bp)
    app.register_blueprint(chat_bp)
    app.register_blueprint(knowledge_bp)
    app.register_blueprint(learning_bp)
    app.register_blueprint(sleep_bp)
    app.register_blueprint(pdf_cache_bp)
    app.register_blueprint(trainer_bp)
    app.register_blueprint(adapters_bp)
    app.register_blueprint(actor_creation_bp)
