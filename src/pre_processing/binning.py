"""
Rời rạc hóa (binning) biến độ tuổi person_age thành các khoảng cố định
7 năm/khoảng (ví dụ: 22-28, 29-35). 
Giúp các mô hình thuật toán tuyến tính bắt được tính chu kỳ rủi ro theo từng nhóm tuổi vay vốn.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AgeBinner(BaseEstimator, TransformerMixin):
    """
    Chia tuổi thành các nhóm nhưng trả về mã số (ordinal bins)
    để tương thích với StandardScaler và các mô hình ML.
    """

    def __init__(self,
                 col_name='person_age',
                 bin_size=7,
                 start_age=14,
                 end_age=119):
        self.col_name = col_name
        self.bin_size = bin_size
        self.start_age = start_age
        self.end_age = end_age

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        bins = np.arange(
            self.start_age,
            self.end_age + self.bin_size,
            self.bin_size
        )

        age = X_out[self.col_name]

        age = age.clip(lower=self.start_age, upper=self.end_age-1)
        
        X_out[self.col_name] = pd.cut(
            age,
            bins=bins,
            labels=False,
            include_lowest=True
        )

        X_out[self.col_name] = X_out[self.col_name].astype(float)

        return X_out

