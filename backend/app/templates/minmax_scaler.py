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
    minimum = df[name].min()
    maximum = df[name].max()
    if minimum is None or maximum is None or maximum == minimum:
        expressions.append(pl.lit(0.0).alias(name))
    else:
        span = maximum - minimum
        expressions.append(((pl.col(name) - minimum) / span).alias(name))

scaled = df.with_columns(expressions)

return scaled: DATAFRAME
