# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
columns: str = ""
collapse_whitespace: bool = False
lowercase: bool = False

selected = [name.strip() for name in columns.split(",") if name.strip()]
if not selected:
    selected = [name for name, kind in df.schema.items() if kind == pl.String]

expressions = []
for name in selected:
    column = pl.col(name).str.strip_chars()
    if collapse_whitespace:
        column = column.str.replace_all(r"\s+", " ")
    if lowercase:
        column = column.str.to_lowercase()
    expressions.append(column)

trimmed = df.with_columns(expressions)

return trimmed: DATAFRAME
