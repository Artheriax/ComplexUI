# /// script
# dependencies = ["polars>=1.0", "numpy>=1.26"]
# ///
import polars as pl

df: DATAFRAME
target_column: str = "target"

feature_columns = [name for name in df.columns if name != target_column]
features = df.select(feature_columns).to_numpy()
labels = df[target_column].to_numpy()
dataset = (features, labels)

return dataset: DATASET
