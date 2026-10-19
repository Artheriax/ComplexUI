# /// script
# dependencies = ["polars>=1.0"]
# ///
import polars as pl

df: DATAFRAME
subset: str = ""
keep: str = "first"

selected = [name.strip() for name in subset.split(",") if name.strip()]
if keep not in {"first", "last", "none"}:
    raise ValueError(f"Unknown keep strategy: {keep}")

deduped = df.unique(subset=selected or None, keep=keep)

return deduped: DATAFRAME
