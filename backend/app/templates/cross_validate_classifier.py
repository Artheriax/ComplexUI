# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26"]
# ///
from sklearn.model_selection import cross_val_score

model: ML_MODEL
train: DATASET
cv_folds: int = 5

X, y = train
scores = cross_val_score(model, X, y, cv=cv_folds)

metrics = {
    "cv_folds": int(cv_folds),
    "accuracy_mean": float(scores.mean()),
    "accuracy_std": float(scores.std()),
    "fold_scores": [float(score) for score in scores],
}

return metrics: METRICS
