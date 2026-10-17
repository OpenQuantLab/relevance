"""Gunicorn entry point for the Prediction Revisited dashboard.

    gunicorn -w 2 --timeout 120 wsgi.demo:server
"""

from relevance.webapp.demo.app import app, server  # noqa: F401
