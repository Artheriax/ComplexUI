# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

path: File = ""
separator: str = ","
has_header: bool = True

df = pl.read_csv(path, separator=separator, has_header=has_header)

return df: DATAFRAME
