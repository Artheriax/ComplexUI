# /// script
# dependencies = ["numpy>=1.26", "plotly>=6.0"]
# ///
import numpy as np
import plotly.graph_objects as go

model: ML_MODEL
top_k: int = 20
feature_names: str = ""
absolute: bool = True

if not 1 <= int(top_k) <= 100:
    raise ValueError("top_k must be between 1 and 100")

if hasattr(model, "feature_importances_"):
    values = np.asarray(model.feature_importances_, dtype=float).reshape(-1)
    importance = "feature_importances_"
elif hasattr(model, "coef_"):
    coefficients = np.asarray(model.coef_, dtype=float)
    if coefficients.ndim == 1:
        values = coefficients
    elif absolute:
        values = np.abs(coefficients).mean(axis=0)
    else:
        values = coefficients.mean(axis=0)
    if absolute:
        values = np.abs(values)
    importance = "coefficient"
else:
    raise ValueError(
        f"Model type '{type(model).__name__}' exposes neither "
        "feature_importances_ nor coef_"
    )
values = np.nan_to_num(values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
if values.size == 0:
    raise ValueError("Model exposes no feature importance values")

provided = [name.strip() for name in feature_names.split(",") if name.strip()]
if provided:
    if len(provided) != values.size:
        raise ValueError(
            f"feature_names has {len(provided)} entries but the model has "
            f"{values.size} features"
        )
    names = provided
else:
    names = [f"feature_{index}" for index in range(values.size)]

order = np.argsort(-np.abs(values), kind="stable")[:top_k]
selected_values = [float(values[index]) for index in order]
selected_names = [names[index] for index in order]

figure = go.Figure(
    data=[
        go.Bar(
            x=selected_values,
            y=selected_names,
            orientation="h",
            marker_color="#60a5fa",
        )
    ]
)
figure.update_layout(
    template="plotly_dark",
    title=f"Feature importance ({importance})",
    xaxis_title="importance",
    yaxis_title="feature",
)
figure.update_yaxes(autorange="reversed")
display(figure)

summary = {
    "model_type": type(model).__name__,
    "importance": importance,
    "n_features": int(values.size),
    "features_shown": int(len(order)),
    "top_features": [
        {"feature": name, "importance": value}
        for name, value in zip(selected_names, selected_values)
    ],
}

return summary: METRICS
