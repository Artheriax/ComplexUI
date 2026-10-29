# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26", "plotly>=6.0"]
# ///
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, confusion_matrix

model: ML_MODEL
test: DATASET
normalize: Literal["none", "true", "pred", "all"] = "none"
max_classes: int = 30
annotate: bool = True

if not 2 <= int(max_classes) <= 100:
    raise ValueError("max_classes must be between 2 and 100")

X_test, y_test = test
predictions = model.predict(X_test)
labels = np.unique(np.concatenate([np.asarray(y_test), np.asarray(predictions)]))
if len(labels) > max_classes:
    raise ValueError(
        f"Confusion matrix has {len(labels)} classes; increase max_classes "
        f"above {int(max_classes)} to display it"
    )

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=labels,
    normalize=None if normalize == "none" else normalize,
)
matrix = np.atleast_2d(matrix)
if normalize == "none":
    text = [[str(int(value)) for value in row] for row in matrix]
else:
    text = [[f"{value:.2f}" for value in row] for row in matrix]
class_names = [str(label) for label in labels]

figure = go.Figure(
    data=[
        go.Heatmap(
            z=matrix.tolist(),
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=text if annotate else None,
            texttemplate="%{text}" if annotate else None,
            colorbar={"title": "count" if normalize == "none" else "ratio"},
        )
    ]
)
figure.update_layout(
    template="plotly_dark",
    title="Confusion matrix",
    xaxis_title="Predicted",
    yaxis_title="Actual",
)
figure.update_yaxes(autorange="reversed")
display(figure)

summary = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "n_test_samples": int(len(y_test)),
    "n_classes": int(len(labels)),
    "labels": class_names,
    "normalize": normalize,
    "confusion_matrix": matrix.tolist(),
}

return summary: METRICS
