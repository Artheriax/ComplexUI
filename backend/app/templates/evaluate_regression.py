# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26"]
# ///
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model: ML_MODEL
test: DATASET

X_test, y_test = test
predictions = model.predict(X_test)
mse = float(mean_squared_error(y_test, predictions))

metrics = {
    "mae": float(mean_absolute_error(y_test, predictions)),
    "mse": mse,
    "rmse": float(np.sqrt(mse)),
    "r2": float(r2_score(y_test, predictions)),
    "n_test_samples": int(len(y_test)),
}

return metrics: METRICS
