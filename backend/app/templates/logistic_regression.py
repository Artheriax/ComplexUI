# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.linear_model import LogisticRegression

train: DATASET
C: float = 1.0
max_iter: int = 200
class_weight: str = ""

X, y = train
model = LogisticRegression(
    C=C,
    max_iter=max_iter,
    class_weight=class_weight.strip() or None,
)
model.fit(X, y)

return model: ML_MODEL
