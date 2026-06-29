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
    Rời rạc hóa biến person_age thành các khoảng 7 năm cố định.
    Cung cấp category 'Unknown' để bắt các ngoại lệ không lọt vào nhóm nào.
    """
    def __init__(self, col_name='person_age', bin_size=7, start_age=14, end_age=119):
        self.col_name = col_name
        self.bin_size = bin_size
        self.start_age = start_age
        self.end_age = end_age
        
        # Tự động tạo mốc cắt: [14, 21, 28, 35, 42, ...]
        self.bins = list(range(self.start_age, self.end_age + self.bin_size + 1, self.bin_size))
        # Tự động tạo nhãn: ['15-21', '22-28', '29-35', ...]
        self.labels = [f"{self.bins[i]+1}-{self.bins[i+1]}" for i in range(len(self.bins)-1)]

    def fit(self, X, y=None):
        # Không có yếu tố thống kê cần học, triệt tiêu Data Leakage
        return self

    def transform(self, X):
        X_transformed = X.copy()
        
        if self.col_name in X_transformed.columns:
            binned_series = pd.cut(
                X_transformed[self.col_name], 
                bins=self.bins, 
                labels=self.labels, 
                right=True, 
                include_lowest=True
            )
            # Ép kiểu an toàn, tránh NaN cho ngoại lệ
            binned_series = binned_series.cat.add_categories('Unknown').fillna('Unknown')
            # Cập nhật và chuyển sang String để module Target Encoding nhận diện
            X_transformed[self.col_name] = binned_series.astype(str)
            
        return X_transformed