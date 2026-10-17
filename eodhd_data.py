"""
Deprecated import location.

The code now lives in :mod:`relevance.data.eodhd`.
"""

import sys

from relevance.data import eodhd as _module

sys.modules[__name__] = _module
