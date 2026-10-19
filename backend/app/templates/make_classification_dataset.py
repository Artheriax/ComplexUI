# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import make_classification
import polars as pl

n_samples: int = 300
n_features: int = 10
n_informative: int = 5
n_classes: int = 2
random_state: int = 42

features, labels = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=min(n_informative, n_features),
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=random_state,
)
columns = [f"feature_{index}" for index in range(n_features)]
df = pl.DataFrame(features, schema=columns).with_columns(
    pl.Series("target", labels)
)

return df: DATAFRAME
