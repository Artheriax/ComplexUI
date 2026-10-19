# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.linear_model import LinearRegression

train: DATASET
fit_intercept: bool = True

X, y = train
model = LinearRegression(fit_intercept=fit_intercept)
model.fit(X, y)

return model: ML_MODEL
