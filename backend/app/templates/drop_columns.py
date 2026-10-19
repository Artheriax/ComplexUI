# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    raise ValueError("Provide at least one column to drop")

dropped = df.drop(selected)

return dropped: DATAFRAME
