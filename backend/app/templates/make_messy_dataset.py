# /// script
# dependencies = ["polars>=1.0", "numpy>=1.26"]
# ///
import numpy as np
import polars as pl

n_rows: int = 200
seed: int = 42

rng = np.random.default_rng(seed)
ages = rng.integers(18, 75, size=n_rows).astype(float).tolist()
scores = rng.normal(50, 12, size=n_rows).tolist()
for index in rng.choice(n_rows, size=max(1, n_rows // 10), replace=False):
    ages[int(index)] = None
for index in rng.choice(n_rows, size=max(1, n_rows // 12), replace=False):
    scores[int(index)] = None
scores[0] = 250.0

base = pl.DataFrame(
    {
        "id": np.arange(n_rows),
        "name": rng.choice([" Alice ", "bob", "Carol ", " dave", "Eve"], size=n_rows),
        "age": ages,
        "score": scores,
        "city": rng.choice([" Paris", "paris ", "Lyon", "lyon", None], size=n_rows),
        "subscribed": rng.choice(["yes", "no", " maybe "], size=n_rows),
    }
)
df = pl.concat([base, base.head(5)])

return df: DATAFRAME
