# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.tree import DecisionTreeRegressor

train: DATASET
max_depth: int = 0
min_samples_leaf: int = 1

X, y = train
depth = max_depth if max_depth > 0 else None
model = DecisionTreeRegressor(
    max_depth=depth,
    min_samples_leaf=min_samples_leaf,
    random_state=42,
)
model.fit(X, y)

return model: ML_MODEL
