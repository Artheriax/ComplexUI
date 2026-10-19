# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

path: str = "data.ndjson"

df = pl.read_ndjson(path)

return df: DATAFRAME
