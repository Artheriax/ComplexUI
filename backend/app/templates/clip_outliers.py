# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
lower_quantile: float = 0.01
upper_quantile: float = 0.99

if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
    raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1")

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [name for name, kind in df.schema.items() if kind.is_numeric()]

expressions = []
for name in selected:
    lower = df[name].quantile(lower_quantile)
    upper = df[name].quantile(upper_quantile)
    if lower is None or upper is None:
        continue
    expressions.append(pl.col(name).clip(lower, upper))

clipped = df.with_columns(expressions)

return clipped: DATAFRAME
