"""API module for REST endpoints."""

from rag_interface.api.routes import create_app, router

__all__ = ["router", "create_app"]
