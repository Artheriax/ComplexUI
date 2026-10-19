# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

path: str = "data.parquet"

df = pl.read_parquet(path)

return df: DATAFRAME
