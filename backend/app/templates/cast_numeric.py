# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
dtype: str = "float"
errors: str = "coerce"

if dtype not in {"float", "int"}:
    raise ValueError(f"Unsupported dtype: {dtype}")

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [name for name, kind in df.schema.items() if kind == pl.String]

target = pl.Float64 if dtype == "float" else pl.Int64
strict = errors == "raise"
casts = [pl.col(name).cast(target, strict=strict) for name in selected]
converted = df.with_columns(casts)

return converted: DATAFRAME
