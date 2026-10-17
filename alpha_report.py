"""
Deprecated import location.

The code now lives in :mod:`relevance.reporting.html_report`; the CLI is also
available as ``relevance-report``.
"""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("relevance.reporting.html_report", run_name="__main__")
else:
    from relevance.reporting import html_report as _module

    sys.modules[__name__] = _module
