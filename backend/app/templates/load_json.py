# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

path: File = ""

df = pl.read_ndjson(path)

return df: DATAFRAME
