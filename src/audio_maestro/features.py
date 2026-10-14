"""Fixed frequency bands used in numerical reports."""

BANDS = (
    ("band_sub", 20, 60),
    ("band_bass", 60, 250),
    ("band_low_mid", 250, 500),
    ("band_mid", 500, 2000),
    ("band_high_mid", 2000, 6000),
    ("band_presence", 6000, 10000),
    ("band_air", 10000, None),
)
