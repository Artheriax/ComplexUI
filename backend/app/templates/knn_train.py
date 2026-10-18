# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.neighbors import KNeighborsClassifier

train: DATASET
n_neighbors: int = 5

X_train, y_train = train
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

return model: ML_MODEL
