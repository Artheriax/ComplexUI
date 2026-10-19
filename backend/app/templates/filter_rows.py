# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
column: str = "target"
operator: str = "=="
value: str = "1"

kind = df.schema[column]
if kind.is_numeric():
    numeric = float(value)
    literal = int(numeric) if kind.is_integer() else numeric
elif kind == pl.Boolean:
    literal = value.strip().lower() in {"1", "true", "yes"}
else:
    literal = value

if operator == "==":
    predicate = pl.col(column) == literal
elif operator == "!=":
    predicate = pl.col(column) != literal
elif operator == ">":
    predicate = pl.col(column) > literal
elif operator == ">=":
    predicate = pl.col(column) >= literal
elif operator == "<":
    predicate = pl.col(column) < literal
elif operator == "<=":
    predicate = pl.col(column) <= literal
elif operator == "contains":
    predicate = pl.col(column).cast(pl.String).str.contains(value, literal=True)
else:
    raise ValueError(f"Unsupported operator: {operator}")

filtered = df.filter(predicate)

return filtered: DATAFRAME
