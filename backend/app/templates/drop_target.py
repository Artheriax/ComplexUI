# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
target_column: str = "target"

features = df.drop(target_column)

return features: DATAFRAME
