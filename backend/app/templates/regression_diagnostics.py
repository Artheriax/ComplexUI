# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26", "plotly>=6.0"]
# ///
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

model: ML_MODEL
test: DATASET
max_points: int = 2000

if not 100 <= int(max_points) <= 20_000:
    raise ValueError("max_points must be between 100 and 20000")

X_test, y_test = test
actual = np.asarray(y_test, dtype=float).reshape(-1)
predictions = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
if actual.size == 0:
    raise ValueError("Test set is empty")
if predictions.size != actual.size:
    raise ValueError("Predictions and targets have different lengths")

residuals = actual - predictions
mse = float(mean_squared_error(actual, predictions))
r2 = float(r2_score(actual, predictions))
if not np.isfinite(r2):
    r2 = None

step = max(1, int(np.ceil(actual.size / max_points)))
plot_actual = actual[::step]
plot_predictions = predictions[::step]
plot_residuals = residuals[::step]
low = float(min(plot_actual.min(), plot_predictions.min()))
high = float(max(plot_actual.max(), plot_predictions.max()))
if high == low:
    high += 1.0
    low -= 1.0

figure = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Actual vs predicted", "Residuals vs predicted"),
    horizontal_spacing=0.12,
)
figure.add_trace(
    go.Scatter(
        x=plot_actual.tolist(),
        y=plot_predictions.tolist(),
        mode="markers",
        name="Predictions",
        marker={"opacity": 0.65, "color": "#60a5fa"},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=[low, high],
        y=[low, high],
        mode="lines",
        name="Ideal",
        line={"color": "#f87171", "dash": "dash"},
    ),
    row=1,
    col=1,
)
figure.add_trace(
    go.Scatter(
        x=plot_predictions.tolist(),
        y=plot_residuals.tolist(),
        mode="markers",
        name="Residuals",
        marker={"opacity": 0.65, "color": "#34d399"},
    ),
    row=1,
    col=2,
)
figure.add_trace(
    go.Scatter(
        x=[float(plot_predictions.min()), float(plot_predictions.max())],
        y=[0.0, 0.0],
        mode="lines",
        name="Zero",
        line={"color": "#f87171", "dash": "dash"},
    ),
    row=1,
    col=2,
)
figure.update_xaxes(title_text="Actual", row=1, col=1)
figure.update_yaxes(title_text="Predicted", row=1, col=1)
figure.update_xaxes(title_text="Predicted", row=1, col=2)
figure.update_yaxes(title_text="Residual", row=1, col=2)
figure.update_layout(
    template="plotly_dark",
    title="Regression diagnostics",
    showlegend=False,
    height=460,
)
display(figure)

summary = {
    "mae": float(mean_absolute_error(actual, predictions)),
    "mse": mse,
    "rmse": float(np.sqrt(mse)),
    "r2": r2,
    "n_test_samples": int(actual.size),
    "residual_mean": float(np.mean(residuals)),
    "residual_std": float(np.std(residuals)),
    "max_abs_error": float(np.max(np.abs(residuals))),
    "plotted_points": int(plot_actual.size),
}

return summary: METRICS
