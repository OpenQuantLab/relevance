"""
Deprecated import location.

The code now lives in :mod:`relevance.studies.paper`.
"""

import sys

from relevance.studies import paper as _module

sys.modules[__name__] = _module
