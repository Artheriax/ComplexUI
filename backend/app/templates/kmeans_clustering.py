# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.cluster import KMeans

train: DATASET
n_clusters: int = 3
random_state: int = 42

X, _ = train
model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
model.fit(X)

return model: ML_MODEL
