# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.datasets import load_digits
import polars as pl

digits = load_digits()
columns = [f"pixel_{index}" for index in range(digits.data.shape[1])]
df = pl.DataFrame(digits.data, schema=columns).with_columns(
    pl.Series("target", digits.target)
)

return df: DATAFRAME
