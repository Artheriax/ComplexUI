# /// script
# dependencies = ["lightgbm>=4.0", "numpy>=1.26"]
# ///
from lightgbm import LGBMClassifier

train: DATASET
n_estimators: int = 100
num_leaves: int = 31
learning_rate: float = 0.1
random_state: int = 42

X, y = train
model = LGBMClassifier(
    n_estimators=n_estimators,
    num_leaves=num_leaves,
    learning_rate=learning_rate,
    random_state=random_state,
    n_jobs=1,
    verbosity=-1,
)
model.fit(X, y)

return model: ML_MODEL
