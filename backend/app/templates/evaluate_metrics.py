# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26"]
# ///
from sklearn.metrics import accuracy_score, confusion_matrix

model: ML_MODEL
test: DATASET

X_test, y_test = test
predictions = model.predict(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    "n_test_samples": int(len(y_test)),
}

return metrics: METRICS
