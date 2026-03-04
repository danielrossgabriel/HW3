
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer
from scipy.stats import skew


class AutoPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Automatically applies Yeo‑Johnson transformation
    to skewed numeric features.
    """

    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.skewed_cols = []
        self.pt = PowerTransformer(method="yeo-johnson")

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        numeric_df = X.select_dtypes(include=[np.number])

        if numeric_df.empty:
            return self

        skewness = numeric_df.apply(lambda x: skew(x.dropna()))
        self.skewed_cols = skewness[abs(skewness) > self.threshold].index.tolist()

        if self.skewed_cols:
            self.pt.fit(X[self.skewed_cols])

        return self

    def transform(self, X):

        X_copy = X.copy()

        if not isinstance(X_copy, pd.DataFrame):
            X_copy = pd.DataFrame(X_copy)

        if self.skewed_cols:
            X_copy[self.skewed_cols] = self.pt.transform(X_copy[self.skewed_cols])

        return X_copy


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Removes features with too many missing values
    or very low correlation with the target.
    """

    def __init__(self, missing_threshold=0.3, corr_threshold=0.03):
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.features_to_keep = []

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Remove columns with too many missing values
        null_ratios = X.isnull().mean()
        cols_low_missing = null_ratios[null_ratios <= self.missing_threshold].index.tolist()

        X_filtered = X[cols_low_missing]

        numeric_X = X_filtered.select_dtypes(include="number")

        if y is not None and not numeric_X.empty:

            temp_df = numeric_X.copy()
            temp_df["target"] = y

            correlations = temp_df.corr()["target"].abs().drop("target")

            self.features_to_keep = correlations[correlations >= self.corr_threshold].index.tolist()

        else:

            self.features_to_keep = numeric_X.columns.tolist()

        return self

    def transform(self, X):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X[self.features_to_keep]
