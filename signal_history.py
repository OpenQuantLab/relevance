"""
Deprecated import location.

The code now lives in :mod:`relevance.storage.signals`; the CLI is also
available as ``relevance-signals``.
"""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("relevance.storage.signals", run_name="__main__")
else:
    from relevance.storage import signals as _module

    sys.modules[__name__] = _module
