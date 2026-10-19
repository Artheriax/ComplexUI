# /// script
# dependencies = ["scikit-learn>=1.5", "numpy>=1.26"]
# ///
from sklearn.decomposition import PCA

train: DATASET
n_components: int = 2
random_state: int = 42

X, y = train
pca = PCA(n_components=n_components, random_state=random_state)
X_reduced = pca.fit_transform(X)
reduced = (X_reduced, y)

return reduced: DATASET
