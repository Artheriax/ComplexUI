# /// script
# dependencies = ["torch>=2.0", "polars>=1.0", "numpy>=1.26"]
# ///
import torch
import polars as pl

df: DATAFRAME
target_column: str = "target"
device: str = "cpu"

feature_columns = [name for name in df.columns if name != target_column]
features = df.select(feature_columns).to_numpy()
labels = df[target_column].to_numpy()

X = torch.tensor(features, dtype=torch.float32, device=device)
y = torch.tensor(labels, dtype=torch.long, device=device)
data = (X, y)

return data: DATASET
