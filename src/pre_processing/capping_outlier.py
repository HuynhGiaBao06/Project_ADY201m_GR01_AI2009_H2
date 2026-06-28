# Xử lý outlier và scaling
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapping(BaseEstimator, TransformerMixin):
    """
    Winsorization/Capping bằng percentile 1%-99%.
    Học ngưỡng trên Train và áp dụng cho Train/Test.
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.thresholds_ = {}

    def fit(self, X, y=None):

        X = X.copy()

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        cols = self.cols

        if cols is None:
            cols = X.select_dtypes(include=[np.number]).columns.tolist()

        self.cols_ = cols

        self.thresholds_ = {}

        for col in self.cols_:

            if col not in X.columns:
                continue

            lower = X[col].quantile(0.01)
            upper = X[col].quantile(0.99)

            self.thresholds_[col] = (lower, upper)

        return self

    def transform(self, X):

        if not self.thresholds_:
            raise ValueError(
                "OutlierCapping has not been fitted. Call fit() first."
            )

        X_out = X.copy()

        for col in self.cols_:

            if col not in X_out.columns:
                continue

            lower, upper = self.thresholds_[col]

            X_out[col] = np.clip(
                X_out[col],
                lower,
                upper
            )

        return X_out