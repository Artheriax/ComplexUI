# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
how: str = "any"

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    cleaned = df.drop_nulls()
elif how == "all":
    nulls = [pl.col(name).is_null() for name in selected]
    cleaned = df.filter(~pl.all_horizontal(nulls))
else:
    nulls = [pl.col(name).is_null() for name in selected]
    cleaned = df.filter(~pl.any_horizontal(nulls))

return cleaned: DATAFRAME
