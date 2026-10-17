"""
Deprecated import location.

The code now lives in :mod:`relevance.backtest.harness`; the CLI is also
available as ``relevance-backtest``.
"""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("relevance.backtest.harness", run_name="__main__")
else:
    from relevance.backtest import harness as _module

    sys.modules[__name__] = _module
