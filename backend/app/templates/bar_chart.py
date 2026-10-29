# /// script
# dependencies = ["polars>=1.0", "plotly>=6.0"]
# ///
from typing import Literal

import polars as pl
import plotly.graph_objects as go

df: DATAFRAME
category: str = ""
value: str = ""
aggregate: Literal["count", "sum", "mean", "median", "min", "max"] = "count"
top_k: int = 20
horizontal: bool = False

if category not in df.columns:
    raise ValueError(f"Unknown category column '{category}'")
if not 1 <= int(top_k) <= 50:
    raise ValueError("top_k must be between 1 and 50")

data = df.filter(pl.col(category).is_not_null())
if data.height == 0:
    raise ValueError(f"Column '{category}' has no non-null values")

if aggregate == "count":
    grouped = data.group_by(category).agg(pl.len().alias("value"))
    value_label = "rows"
else:
    if value not in df.columns:
        raise ValueError(f"Unknown value column '{value}'")
    if not df.schema[value].is_numeric():
        raise ValueError(f"Column '{value}' must be numeric for aggregate '{aggregate}'")
    column = pl.col(value)
    if aggregate == "sum":
        expression = column.sum()
    elif aggregate == "mean":
        expression = column.mean()
    elif aggregate == "median":
        expression = column.median()
    elif aggregate == "min":
        expression = column.min()
    else:
        expression = column.max()
    grouped = data.group_by(category).agg(expression.alias("value"))
    value_label = value

groups_total = grouped.height
top = grouped.sort("value", descending=True).head(top_k).drop_nulls("value")
if top.height == 0:
    raise ValueError("No numeric values are available for this aggregation")
categories = [str(item) for item in top[category].to_list()]
values = [float(item) for item in top["value"].to_list()]

if horizontal:
    trace = go.Bar(x=values, y=categories, orientation="h", name=value_label)
else:
    trace = go.Bar(x=categories, y=values, name=value_label)
figure = go.Figure(data=[trace])
figure.update_layout(
    template="plotly_dark",
    title=f"{aggregate.title()} of {value_label} by {category}",
    xaxis_title=value_label if horizontal else category,
    yaxis_title=category if horizontal else value_label,
)
if horizontal:
    figure.update_yaxes(autorange="reversed")
display(figure)

summary = {
    "category": category,
    "value": value or None,
    "aggregate": aggregate,
    "rows": int(data.height),
    "groups_total": int(groups_total),
    "groups_shown": int(top.height),
    "max_value": values[0] if values else None,
    "max_category": categories[0] if categories else None,
}

return summary: METRICS
