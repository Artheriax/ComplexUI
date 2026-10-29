# /// script
# dependencies = ["polars>=1.0", "plotly>=6.0"]
# ///
import polars as pl
import plotly.graph_objects as go

df: DATAFRAME
column: str = ""
bins: int = 30
group_by: str = ""
max_groups: int = 12
log_y: bool = False

if column not in df.columns:
    available = ", ".join(df.columns[:20]) or "none"
    raise ValueError(f"Unknown column '{column}'. Available columns: {available}")
if not df.schema[column].is_numeric():
    raise ValueError(f"Column '{column}' must be numeric for a histogram")
if not 1 <= int(bins) <= 500:
    raise ValueError("bins must be between 1 and 500")
if not 1 <= int(max_groups) <= 30:
    raise ValueError("max_groups must be between 1 and 30")
if group_by and group_by not in df.columns:
    raise ValueError(f"Unknown group column '{group_by}'")
if group_by == column:
    raise ValueError("group_by must be a different column than column")

values = df[column].drop_nulls().cast(pl.Float64)
if values.len() == 0:
    raise ValueError(f"Column '{column}' has no non-null values")

summary = {
    "column": column,
    "rows": int(df.height),
    "non_null": int(values.len()),
    "bins": int(bins),
    "min": float(values.min()),
    "max": float(values.max()),
    "mean": float(values.mean()),
    "median": float(values.median()),
}

figure = go.Figure()
if group_by:
    groups = (
        df[group_by].drop_nulls().unique(maintain_order=True).head(max_groups).to_list()
    )
    for group in groups:
        subset = (
            df.filter(pl.col(group_by) == group)[column]
            .drop_nulls()
            .cast(pl.Float64)
        )
        if subset.len() == 0:
            continue
        figure.add_trace(
            go.Histogram(
                x=subset.to_list(),
                name=str(group),
                nbinsx=int(bins),
                opacity=0.65,
            )
        )
    figure.update_layout(barmode="overlay")
    summary["group_by"] = group_by
    summary["groups"] = [str(group) for group in groups]
else:
    figure.add_trace(go.Histogram(x=values.to_list(), nbinsx=int(bins)))

figure.update_layout(
    template="plotly_dark",
    title=f"Histogram of {column}",
    xaxis_title=column,
    yaxis_title="Count",
    bargap=0.05,
)
if log_y:
    figure.update_yaxes(type="log")
display(figure)

return summary: METRICS
