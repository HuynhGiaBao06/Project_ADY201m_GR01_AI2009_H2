"""
Xử lý Outlier bằng phương pháp Percentile Capping.

Nguyên tắc:
- Chỉ học ngưỡng trên tập Train.
- Áp dụng cùng một ngưỡng cho cả Train và Test.
- Không tính toán lại trên tập Test nhằm tránh Data Leakage.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierCapping(BaseEstimator, TransformerMixin):
    """
    Module xử lý Outlier bằng Percentile Capping.
    """

    def __init__(
        self,
        numerical_cols=None,
        lower_quantile=0.01,
        upper_quantile=0.99,
    ):
        self.numerical_cols = numerical_cols
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

        # Lưu các ngưỡng capping đã học từ tập Train
        self.capping_bounds_ = {}

    def fit(self, X, y=None):
        """
        GIAI ĐOẠN 1: HỌC (FIT)

        Học ngưỡng capping trên tập Train.
        """

        if self.numerical_cols is None:
            self.numerical_cols = (
                X.select_dtypes(include=[np.number])
                .columns
                .tolist()
            )

        self.capping_bounds_.clear()

        for col in self.numerical_cols:

            if col in X.columns:

                self.capping_bounds_[col] = {
                    "lower": X[col].quantile(self.lower_quantile),
                    "upper": X[col].quantile(self.upper_quantile),
                }

        return self

    def transform(self, X):
        """
        GIAI ĐOẠN 2: ÁP DỤNG (TRANSFORM)

        Áp dụng các ngưỡng đã học lên dữ liệu.
        """

        X_out = X.copy()

        for col, bounds in self.capping_bounds_.items():

            if col in X_out.columns:

                X_out[col] = np.clip(
                    X_out[col],
                    bounds["lower"],
                    bounds["upper"],
                )

        return X_out

    def fit_transform(self, X, y=None):
        """
        Fit và Transform trên tập Train.
        """

        return self.fit(X, y).transform(X)