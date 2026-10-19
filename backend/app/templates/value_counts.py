# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
column: str = "target"
top_k: int = 20

tallies = df[column].value_counts(sort=True).head(top_k)
total = int(df[column].drop_nulls().len())
rows = [
    {
        "value": str(record[column]),
        "count": int(record["count"]),
        "share": float(record["count"] / total) if total else 0.0,
    }
    for record in tallies.to_dicts()
]

counts = {"column": column, "total": total, "rows": rows}

return counts: METRICS
