# /// script
# dependencies = ["scikit-learn>=1.5"]
# ///
from sklearn.naive_bayes import GaussianNB

train: DATASET

X, y = train
model = GaussianNB()
model.fit(X, y)

return model: ML_MODEL
