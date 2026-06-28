"""
Chuẩn hóa dữ liệu bằng StandardScaler.

Nguyên tắc:
- Chỉ học Mean và Standard Deviation trên tập Train.
- Áp dụng cùng bộ tham số cho Train/Test.
- Không tính toán lại trên tập Test nhằm tránh Data Leakage.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Module chuẩn hóa các biến số bằng StandardScaler.
    """

    def __init__(self, numerical_cols=None):

        self.numerical_cols = numerical_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        """
        GIAI ĐOẠN 1: HỌC (FIT)

        Học Mean và Standard Deviation từ tập Train.
        """

        if self.numerical_cols is None:

            self.numerical_cols = (
                X.select_dtypes(include=[np.number])
                .columns
                .tolist()
            )

        self.scaler.fit(X[self.numerical_cols])

        return self

    def transform(self, X):
        """
        GIAI ĐOẠN 2: ÁP DỤNG (TRANSFORM)

        Chuẩn hóa dữ liệu bằng các tham số đã học.
        """

        X_out = X.copy()

        X_out[self.numerical_cols] = self.scaler.transform(
            X_out[self.numerical_cols]
        )

        return X_out

    def fit_transform(self, X, y=None):
        """
        Fit và Transform trên tập Train.
        """

        return self.fit(X, y).transform(X)