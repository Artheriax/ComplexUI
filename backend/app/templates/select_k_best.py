# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26"]
# ///
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

train: DATASET
k: int = 5
score_func: str = "f_classif"

X, y = train
function = f_regression if score_func == "f_regression" else f_classif
selector = SelectKBest(score_func=function, k=min(k, X.shape[1]))
X_selected = selector.fit_transform(X, y)
reduced = (X_selected, y)

return reduced: DATASET
