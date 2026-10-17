"""
Relevance — transparent, parameter-free prediction applied to cross-sectional
equity alpha research.

Layout
------
``relevance.core``       the relevance/partial-sample-regression maths
``relevance.data``       EODHD market data access and on-disk price caching
``relevance.backtest``   signal specification, panel construction, walk-forward
``relevance.studies``    replications of the published studies
``relevance.storage``    persisted runs and live signal history
``relevance.reporting``  standalone HTML/SVG reports
``relevance.webapp``     the Dash dashboards
``relevance.cli``        command line entry points
"""

__version__ = "0.1.0"
