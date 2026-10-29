# /// script
# dependencies = ["polars>=1.0", "plotly>=6.0"]
# ///
from typing import Literal

import polars as pl
import plotly.graph_objects as go

df: DATAFRAME
value_column: str = ""
group_by: str = ""
max_groups: int = 12
points: Literal["outliers", "all", "suspectedoutliers", "none"] = "outliers"

if value_column not in df.columns:
    raise ValueError(f"Unknown value column '{value_column}'")
if not df.schema[value_column].is_numeric():
    raise ValueError(f"Column '{value_column}' must be numeric for a box plot")
if group_by and group_by not in df.columns:
    raise ValueError(f"Unknown group column '{group_by}'")
if group_by == value_column:
    raise ValueError("group_by must be a different column than value_column")
if not 1 <= int(max_groups) <= 50:
    raise ValueError("max_groups must be between 1 and 50")

data = df.drop_nulls([value_column])
if group_by:
    data = data.filter(pl.col(group_by).is_not_null())
if data.height == 0:
    raise ValueError(f"Column '{value_column}' has no non-null values")

box_points = False if points == "none" else points
figure = go.Figure()
groups = []
if group_by:
    groups = (
        data[group_by].unique(maintain_order=True).head(max_groups).to_list()
    )
    for group in groups:
        values = data.filter(pl.col(group_by) == group)[value_column].to_list()
        figure.add_trace(
            go.Box(
                y=values,
                name=str(group),
                boxpoints=box_points,
                jitter=0.3,
            )
        )
else:
    figure.add_trace(
        go.Box(
            y=data[value_column].to_list(),
            name=value_column,
            boxpoints=box_points,
            jitter=0.3,
        )
    )
figure.update_layout(
    template="plotly_dark",
    title=f"Box plot of {value_column}",
    yaxis_title=value_column,
    showlegend=bool(group_by),
)
display(figure)

values = data[value_column].cast(pl.Float64)
summary = {
    "value_column": value_column,
    "group_by": group_by or None,
    "rows": int(data.height),
    "groups_total": int(data[group_by].n_unique()) if group_by else 1,
    "groups_shown": len(groups) if group_by else 1,
    "min": float(values.min()),
    "max": float(values.max()),
    "mean": float(values.mean()),
    "median": float(values.median()),
}
if group_by:
    summary["groups"] = [str(group) for group in groups]

return summary: METRICS
