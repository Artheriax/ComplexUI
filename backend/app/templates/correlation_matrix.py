# /// script
# dependencies = ["polars>=1.0", "numpy>=1.26"]
# ///
import numpy as np
import polars as pl
import polars.selectors as cs

df: DATAFRAME
method: str = "pearson"

numeric = df.select(cs.numeric())
names = numeric.columns
if numeric.width < 2:
    values = [[1.0]] if numeric.width == 1 else []
else:
    matrix = numeric.to_numpy().astype(float)
    if method == "spearman":
        ranked = np.apply_along_axis(
            lambda column: np.argsort(np.argsort(np.nan_to_num(column))), 0, matrix
        )
        values = np.corrcoef(ranked, rowvar=False)
    else:
        values = np.corrcoef(matrix, rowvar=False)
    values = [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in np.atleast_2d(values)
    ]

correlation = {"columns": names, "method": method, "matrix": values}

return correlation: METRICS
