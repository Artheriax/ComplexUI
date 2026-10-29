# /// script
# dependencies = ["numpy>=1.26", "plotly>=6.0"]
# ///
import numpy as np
import plotly.graph_objects as go

history: TENSOR
metric_name: str = "loss"
smooth_window: int = 1
show_raw: bool = True
max_points: int = 5000
log_y: bool = False
lower_is_better: bool = True

if not 1 <= int(smooth_window) <= 500:
    raise ValueError("smooth_window must be between 1 and 500")
if not 100 <= int(max_points) <= 50_000:
    raise ValueError("max_points must be between 100 and 50000")

if hasattr(history, "detach"):
    values = history.detach().cpu().reshape(-1).tolist()
else:
    values = np.asarray(history, dtype=float).reshape(-1).tolist()
values = [float(value) for value in values]
if not values:
    raise ValueError("history is empty")
if not np.all(np.isfinite(values)):
    raise ValueError("history contains non-finite values")
if log_y and any(value <= 0 for value in values):
    raise ValueError("log_y requires strictly positive history values")

epochs = len(values)
window = min(int(smooth_window), epochs)
indices = np.linspace(0, epochs - 1, min(epochs, max_points), dtype=int)
raw_x = (indices + 1).tolist()
raw_y = [values[index] for index in indices]

figure = go.Figure()
if show_raw:
    figure.add_trace(
        go.Scatter(
            x=raw_x,
            y=raw_y,
            mode="lines",
            name=metric_name,
            line={"color": "#60a5fa", "width": 1},
            opacity=0.45,
        )
    )
if window > 1:
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    smooth_x_all = np.arange(window, epochs + 1)
    smooth_indices = np.linspace(
        0, smoothed.size - 1, min(smoothed.size, max_points), dtype=int
    )
    figure.add_trace(
        go.Scatter(
            x=smooth_x_all[smooth_indices].tolist(),
            y=smoothed[smooth_indices].tolist(),
            mode="lines",
            name=f"{metric_name} (mean {window})",
            line={"color": "#fbbf24", "width": 2.5},
        )
    )
if not figure.data:
    figure.add_trace(
        go.Scatter(
            x=raw_x,
            y=raw_y,
            mode="lines",
            name=metric_name,
            line={"color": "#60a5fa", "width": 2},
        )
    )
figure.update_layout(
    template="plotly_dark",
    title=f"{metric_name} history",
    xaxis_title="epoch",
    yaxis_title=metric_name,
    showlegend=True,
)
if log_y:
    figure.update_yaxes(type="log")
display(figure)

best_index = int(np.argmin(values) if lower_is_better else np.argmax(values))
summary = {
    "metric": metric_name,
    "epochs": int(epochs),
    "final": float(values[-1]),
    "min": float(np.min(values)),
    "max": float(np.max(values)),
    "lower_is_better": bool(lower_is_better),
    "best_epoch": best_index + 1,
    "best_value": float(values[best_index]),
    "smooth_window": int(window),
    "plotted_points": int(len(raw_x)),
}

return summary: METRICS
