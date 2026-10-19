# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import load_breast_cancer
import polars as pl

cancer = load_breast_cancer()
df = pl.DataFrame(cancer.data, schema=list(cancer.feature_names)).with_columns(
    pl.Series("target", cancer.target)
)

return df: DATAFRAME
