# /// script
# dependencies = ["polars>=1.0", "plotly>=6.0"]
# ///
import math

import polars as pl
import plotly.graph_objects as go

df: DATAFRAME
x: str = ""
y: str = ""
color_by: str = ""
size_by: str = ""
max_points: int = 5000

if x not in df.columns:
    raise ValueError(f"Unknown x column '{x}'")
if y not in df.columns:
    raise ValueError(f"Unknown y column '{y}'")
if not df.schema[x].is_numeric() or not df.schema[y].is_numeric():
    raise ValueError("Scatter axes must be numeric")
if color_by and color_by not in df.columns:
    raise ValueError(f"Unknown color column '{color_by}'")
if size_by and size_by not in df.columns:
    raise ValueError(f"Unknown size column '{size_by}'")
if size_by and not df.schema[size_by].is_numeric():
    raise ValueError(f"Size column '{size_by}' must be numeric")
if not 100 <= int(max_points) <= 50_000:
    raise ValueError("max_points must be between 100 and 50000")

subset = df.drop_nulls([x, y])
if subset.height == 0:
    raise ValueError("No rows have values for both scatter axes")

correlation = subset.select(pl.corr(pl.col(x), pl.col(y))).item()
plotted = subset
sampled = False
if subset.height > max_points:
    step = int((subset.height + max_points - 1) // max_points)
    plotted = subset.gather_every(step)
    sampled = True

marker = {"opacity": 0.7}
if color_by:
    colors = plotted[color_by].to_list()
    if plotted[color_by].dtype.is_numeric():
        marker["color"] = colors
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        marker["colorbar"] = {"title": color_by}
    else:
        lookup: dict[str, int] = {}
        codes = []
        overflow = False
        for value in colors:
            if value is None:
                codes.append(-1)
                continue
            label = str(value)
            if label not in lookup:
                if len(lookup) < 20:
                    lookup[label] = len(lookup)
                else:
                    overflow = True
                    codes.append(20)
                    continue
            codes.append(lookup[label])
        tickvals = [-1] + list(range(len(lookup)))
        ticktext = ["missing"] + list(lookup)
        if overflow:
            tickvals.append(20)
            ticktext.append("other")
        marker["color"] = codes
        marker["colorscale"] = "Viridis"
        marker["showscale"] = True
        marker["colorbar"] = {
            "title": color_by,
            "tickvals": tickvals,
            "ticktext": ticktext,
        }
if size_by:
    raw_sizes = plotted[size_by].cast(pl.Float64)
    low = raw_sizes.min()
    high = raw_sizes.max()
    sizes = []
    for value in raw_sizes.to_list():
        if value is None or low is None or high is None:
            sizes.append(8.0)
        elif high == low:
            sizes.append(14.0)
        else:
            sizes.append(4.0 + 20.0 * (float(value) - float(low)) / (float(high) - float(low)))
    marker["size"] = sizes

figure = go.Figure(
    data=[
        go.Scatter(
            x=plotted[x].to_list(),
            y=plotted[y].to_list(),
            mode="markers",
            marker=marker,
            name=f"{y} vs {x}",
        )
    ]
)
figure.update_layout(
    template="plotly_dark",
    title=f"{y} vs {x}",
    xaxis_title=x,
    yaxis_title=y,
)
display(figure)

summary = {
    "x": x,
    "y": y,
    "rows": int(df.height),
    "points": int(subset.height),
    "plotted": int(plotted.height),
    "sampled": sampled,
    "correlation": (
        float(correlation)
        if correlation is not None and math.isfinite(float(correlation))
        else None
    ),
}
if color_by:
    summary["color_by"] = color_by
if size_by:
    summary["size_by"] = size_by

return summary: METRICS
