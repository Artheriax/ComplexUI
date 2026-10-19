# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.tree import DecisionTreeClassifier

train: DATASET
max_depth: int = 0
min_samples_leaf: int = 1
class_weight: str = ""

X, y = train
depth = max_depth if max_depth > 0 else None
model = DecisionTreeClassifier(
    max_depth=depth,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight.strip() or None,
    random_state=42,
)
model.fit(X, y)

return model: ML_MODEL
