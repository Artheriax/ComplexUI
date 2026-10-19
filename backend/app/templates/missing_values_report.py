# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME

rows = int(df.height)
columns = {
    name: {
        "dtype": str(kind),
        "null_count": int(df[name].null_count()),
        "null_fraction": float(df[name].null_count() / rows) if rows else 0.0,
    }
    for name, kind in df.schema.items()
}
report = {
    "rows": rows,
    "columns_with_nulls": [
        name for name, info in columns.items() if info["null_count"] > 0
    ],
    "columns": columns,
}

return report: METRICS
