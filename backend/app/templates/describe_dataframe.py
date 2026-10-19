# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl
import polars.selectors as cs

df: DATAFRAME

numeric = df.select(cs.numeric())
summary = {
    "rows": int(df.height),
    "columns": int(df.width),
    "column_names": df.columns,
    "dtypes": {name: str(kind) for name, kind in df.schema.items()},
    "missing": {name: int(df[name].null_count()) for name in df.columns},
    "numeric_summary": numeric.describe().to_dicts() if numeric.width else [],
}

return summary: METRICS
