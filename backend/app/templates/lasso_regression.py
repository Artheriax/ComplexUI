# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.linear_model import Lasso

train: DATASET
alpha: float = 0.1

X, y = train
model = Lasso(alpha=alpha)
model.fit(X, y)

return model: ML_MODEL
