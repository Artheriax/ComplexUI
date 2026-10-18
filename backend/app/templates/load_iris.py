# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import load_iris
import polars as pl

iris = load_iris()
df = pl.DataFrame(iris.data, schema=iris.feature_names).with_columns(
    pl.Series("target", iris.target)
)

return df: DATAFRAME
