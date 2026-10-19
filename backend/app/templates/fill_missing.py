# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
strategy: str = "mean"
value: float = 0.0

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [name for name, kind in df.schema.items() if kind.is_numeric()]

expressions = []
for name in selected:
    column = pl.col(name)
    if strategy == "mean":
        expressions.append(column.fill_null(column.mean()))
    elif strategy == "median":
        expressions.append(column.fill_null(column.median()))
    elif strategy == "constant":
        expressions.append(column.fill_null(value))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

filled = df.with_columns(expressions)

return filled: DATAFRAME
