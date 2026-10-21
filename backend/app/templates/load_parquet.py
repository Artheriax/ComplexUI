# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

path: File = ""

df = pl.read_parquet(path)

return df: DATAFRAME
