# /// script
# dependencies = ["imbalanced-learn>=0.12", "scikit-learn>=1.5", "numpy>=1.26"]
# ///
from imblearn.over_sampling import RandomOverSampler

train: DATASET
random_state: int = 42

X, y = train
sampler = RandomOverSampler(random_state=random_state)
X_resampled, y_resampled = sampler.fit_resample(X, y)
resampled = (X_resampled, y_resampled)

return resampled: DATASET
