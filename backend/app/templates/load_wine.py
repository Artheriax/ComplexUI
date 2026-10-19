# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import load_wine
import polars as pl

wine = load_wine()
df = pl.DataFrame(wine.data, schema=wine.feature_names).with_columns(
    pl.Series("target", wine.target)
)

return df: DATAFRAME
