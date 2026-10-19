# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.ensemble import RandomForestClassifier

train: DATASET
n_estimators: int = 100
max_depth: int = 0
class_weight: str = ""

X, y = train
depth = max_depth if max_depth > 0 else None
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=depth,
    class_weight=class_weight.strip() or None,
    random_state=42,
)
model.fit(X, y)

return model: ML_MODEL
