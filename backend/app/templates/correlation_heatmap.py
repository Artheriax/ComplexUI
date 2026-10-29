# /// script
# dependencies = ["polars>=1.0", "numpy>=1.26", "plotly>=6.0"]
# ///
from typing import Literal

import numpy as np
import polars as pl
import polars.selectors as cs
import plotly.graph_objects as go

df: DATAFRAME
method: Literal["pearson", "spearman"] = "pearson"
columns: str = ""
max_columns: int = 20
annotate: bool = True

if not 2 <= int(max_columns) <= 30:
    raise ValueError("max_columns must be between 2 and 30")

requested = [name.strip() for name in columns.split(",") if name.strip()]
if requested:
    missing = [name for name in requested if name not in df.columns]
    if missing:
        raise ValueError(f"Unknown columns: {', '.join(missing)}")
    selected = list(dict.fromkeys(requested))
    non_numeric = [
        name for name in selected
        if not df.schema[name].is_numeric()
    ]
    if non_numeric:
        raise ValueError(
            f"Correlation columns must be numeric: {', '.join(non_numeric)}"
        )
    truncated = len(selected) > max_columns
    numeric = df.select(selected[:max_columns])
else:
    numeric_all = df.select(cs.numeric())
    truncated = numeric_all.width > max_columns
    numeric = numeric_all.select(numeric_all.columns[:max_columns])

if numeric.width < 2:
    raise ValueError("Correlation heatmap needs at least two numeric columns")

names = numeric.columns
matrix = numeric.cast(pl.Float64).to_numpy()
if method == "spearman":
    ranked = np.full(matrix.shape, np.nan)
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        mask = np.isfinite(column)
        ranks = np.full(column.shape, np.nan)
        ranks[mask] = np.argsort(np.argsort(column[mask]))
        ranked[:, column_index] = ranks
    values = np.corrcoef(ranked, rowvar=False)
else:
    values = np.corrcoef(matrix, rowvar=False)
values = np.atleast_2d(values)
clean = [
    [None if not np.isfinite(value) else float(value) for value in row]
    for row in values
]
text = [
    ["" if value is None else f"{value:.2f}" for value in row]
    for row in clean
]

best = None
for row_index in range(len(names)):
    for column_index in range(row_index + 1, len(names)):
        value = clean[row_index][column_index]
        if value is None:
            continue
        if best is None or abs(value) > abs(best["correlation"]):
            best = {
                "column_a": names[row_index],
                "column_b": names[column_index],
                "correlation": value,
            }

figure = go.Figure(
    data=[
        go.Heatmap(
            z=clean,
            x=names,
            y=names,
            colorscale="RdBu",
            reversescale=True,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorbar={"title": "r"},
            text=text if annotate else None,
            texttemplate="%{text}" if annotate else None,
        )
    ]
)
figure.update_layout(
    template="plotly_dark",
    title=f"Correlation heatmap ({method})",
    xaxis_title="",
    yaxis_title="",
)
figure.update_yaxes(autorange="reversed")
display(figure)

summary = {
    "method": method,
    "columns": names,
    "n_columns": len(names),
    "truncated": truncated,
    "strongest": best,
}

return summary: METRICS
