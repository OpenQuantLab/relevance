"""Gunicorn entry point for the Alpha Lab dashboard.

    gunicorn -w 1 --threads 8 --timeout 300 wsgi.alpha_lab:server
"""

from relevance.webapp.alpha_lab.app import app, server  # noqa: F401
