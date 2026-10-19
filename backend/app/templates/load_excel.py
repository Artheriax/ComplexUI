# /// script
# dependencies = ["polars>=1.0", "fastexcel>=0.12"]
# ///
import polars as pl

path: str = "data.xlsx"
sheet_id: int = 1

df = pl.read_excel(path, sheet_id=sheet_id)

return df: DATAFRAME
