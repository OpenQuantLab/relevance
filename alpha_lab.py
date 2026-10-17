"""
Deprecated import location.

The code now lives in :mod:`relevance.backtest.lab`.
"""

import sys

from relevance.backtest import lab as _module

sys.modules[__name__] = _module
