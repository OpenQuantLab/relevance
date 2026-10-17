"""
Deprecated import location.

The code now lives in :mod:`relevance.core.prediction`.  This shim keeps
``import predictionrevisited`` working for existing scripts and notebooks;
it will be removed once callers have migrated.
"""

import sys

from relevance.core import prediction as _module

sys.modules[__name__] = _module
