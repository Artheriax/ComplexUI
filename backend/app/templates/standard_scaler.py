# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
target_column: str = "target"
columns: str = ""

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [
        name
        for name, kind in df.schema.items()
        if kind.is_numeric() and name != target_column
    ]

expressions = []
for name in selected:
    mean = df[name].mean()
    deviation = df[name].std(ddof=0)
    if mean is None or deviation is None or deviation == 0:
        expressions.append(pl.lit(0.0).alias(name))
    else:
        expressions.append(((pl.col(name) - mean) / deviation).alias(name))

scaled = df.with_columns(expressions)

return scaled: DATAFRAME
