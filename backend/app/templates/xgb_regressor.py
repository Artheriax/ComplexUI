# /// script
# dependencies = ["xgboost>=2.0", "numpy>=1.26"]
# ///
from xgboost import XGBRegressor

train: DATASET
n_estimators: int = 100
max_depth: int = 6
learning_rate: float = 0.3
random_state: int = 42

X, y = train
model = XGBRegressor(
    n_estimators=n_estimators,
    max_depth=max_depth,
    learning_rate=learning_rate,
    random_state=random_state,
    n_jobs=1,
    tree_method="hist",
)
model.fit(X, y)

return model: ML_MODEL
