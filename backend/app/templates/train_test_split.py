# /// script
# dependencies = ["scikit-learn>=1.5", "polars>=1.0", "numpy>=1.26"]
# ///
from sklearn.model_selection import train_test_split

df: DATAFRAME
test_size: float = 0.2

feature_columns = [name for name in df.columns if name != "target"]
features = df.select(feature_columns).to_numpy()
labels = df["target"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=test_size, random_state=42
)

train = (X_train, y_train)
test = (X_test, y_test)

return train: DATASET, test: DATASET
