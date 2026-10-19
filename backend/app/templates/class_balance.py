# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
target_column: str = "target"

tallies = df[target_column].drop_nulls().value_counts(sort=True)
total = int(tallies["count"].sum())
classes = [
    {
        "value": str(record[target_column]),
        "count": int(record["count"]),
        "share": float(record["count"] / total) if total else 0.0,
    }
    for record in tallies.to_dicts()
]
if classes:
    largest = classes[0]["share"]
    smallest = classes[-1]["share"]
    imbalance = float(largest / smallest) if smallest else None
else:
    imbalance = None
balance = {
    "target_column": target_column,
    "rows": total,
    "n_classes": len(classes),
    "classes": classes,
    "imbalance_ratio": imbalance,
}

return balance: METRICS
