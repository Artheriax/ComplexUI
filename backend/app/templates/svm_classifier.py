# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.svm import SVC

train: DATASET
kernel: str = "rbf"
C: float = 1.0

X, y = train
model = SVC(kernel=kernel, C=C)
model.fit(X, y)

return model: ML_MODEL
