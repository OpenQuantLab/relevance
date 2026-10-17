"""
Deprecated entry point for the Alpha Lab dashboard.

The app now lives in :mod:`relevance.webapp.alpha_lab.app` and is served in
production through ``wsgi/alpha_lab.py``.  This shim keeps the documented
``gunicorn alpha_lab_app:server`` and ``python alpha_lab_app.py`` commands
working.
"""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("relevance.webapp.alpha_lab.app", run_name="__main__")
else:
    from relevance.webapp.alpha_lab import app as _module

    sys.modules[__name__] = _module
