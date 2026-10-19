# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.ensemble import GradientBoostingClassifier

train: DATASET
n_estimators: int = 100
learning_rate: float = 0.1
max_depth: int = 3

X, y = train
model = GradientBoostingClassifier(
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    random_state=42,
)
model.fit(X, y)

return model: ML_MODEL
