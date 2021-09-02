"""Reused settings."""

from pathlib import Path
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

icg_model_path = Path(__file__).parent / "model" / "icg_body_flat.xml"
