# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
drop_first: bool = False

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [
        name
        for name, kind in df.schema.items()
        if kind in (pl.String, pl.Categorical)
    ]

encoded = df.to_dummies(columns=selected, drop_first=drop_first)

return encoded: DATAFRAME
