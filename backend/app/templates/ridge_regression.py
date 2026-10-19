# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.linear_model import Ridge

train: DATASET
alpha: float = 1.0

X, y = train
model = Ridge(alpha=alpha)
model.fit(X, y)

return model: ML_MODEL
