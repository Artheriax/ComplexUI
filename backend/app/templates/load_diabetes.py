# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import load_diabetes
import polars as pl

diabetes = load_diabetes()
df = pl.DataFrame(diabetes.data, schema=list(diabetes.feature_names)).with_columns(
    pl.Series("target", diabetes.target)
)

return df: DATAFRAME
