"""
Flask Blueprint routes for the Silvertongue Dashboard.

Each module provides a Blueprint that handles a specific category of routes.
"""

from .core import core_bp
from .chat import chat_bp
from .knowledge import knowledge_bp
from .learning import learning_bp
from .sleep import sleep_bp
from .pdf_cache import pdf_cache_bp
from .trainer import trainer_bp
from .adapters import adapters_bp
from .actor_creation import actor_creation_bp


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
