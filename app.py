"""
Deprecated entry point for the Prediction Revisited dashboard.

The app now lives in :mod:`relevance.webapp.demo.app` and is served in
production through ``wsgi/demo.py``.  This shim keeps the documented
``gunicorn app:server`` and ``python app.py`` commands working.
"""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("relevance.webapp.demo.app", run_name="__main__")
else:
    from relevance.webapp.demo import app as _module

    sys.modules[__name__] = _module
