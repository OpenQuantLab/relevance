"""
The temperature dataset from the book's worked example.

Latitudes are approximate; temperatures are 2010 monthly averages from
Terrestrial Air Temperature 1900-2017 Gridded Monthly Time Series v5.01
(Matsuura and Willmott). January = 0, July = 1.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_OBSERVATIONS = pd.DataFrame(
    [
        ["London", 51.25, 0, 1.1],
        ["Tokyo", 35.75, 0, 6.3],
        ["Rio de Janeiro", -22.75, 0, 27.9],
        ["Buenos Aires", -34.75, 0, 24.8],
        ["London", 51.25, 1, 18.2],
        ["Tokyo", 35.75, 1, 28.0],
        ["Rio de Janeiro", -22.75, 1, 21.1],
        ["Buenos Aires", -34.75, 1, 10.0],
    ],
    columns=["city", "latitude", "january_or_july", "avg_temp_celsius"],
)

DEFAULT_PREDICTIONS = pd.DataFrame(
    [
        ["New York", 40.75, 0, 0.1],
        ["Sydney", -33.75, 0, 23.4],
        ["New York", 40.75, 1, 25.9],
        ["Sydney", -33.75, 1, 12.5],
    ],
    columns=["city", "latitude", "january_or_july", "avg_temp_celsius"],
)
