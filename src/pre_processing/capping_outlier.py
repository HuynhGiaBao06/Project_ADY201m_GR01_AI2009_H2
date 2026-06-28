# Xử lý outlier và scaling
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierCapping:
    def __init__(self, cols=None):
        """
        Khởi tạo module Capping.
        
        Parameters:
        cols (list or None): Danh sách các cột cần xử lý ngoại lai. 
                             Nếu None, module sẽ tự động áp dụng cho tất cả các cột dạng số.
        """
        self.cols = cols
        self.thresholds_ = {}  # Dictionary để lưu trữ ngưỡng 1% và 99% của từng cột

    def fit(self, X, y=None):
        """
        Học các mốc phân vị 1% và 99% từ tập Train.
        
        Parameters:
        X (pd.DataFrame): Dữ liệu đầu vào (bắt buộc là tập Train).
        y (pd.Series, optional): Biến mục tiêu (không sử dụng, giữ nguyên để tương thích Pipeline).
        
        Returns:
        self
        """
        # Đảm bảo dữ liệu đầu vào là DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Nếu không truyền danh sách cột cụ thể, tự động lấy toàn bộ các cột có kiểu dữ liệu số
        if self.cols is None:
            self.cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Tính toán và lưu trữ ngưỡng phân vị 1% và 99% cho từng cột
        for col in self.cols:
            lower_bound = X[col].quantile(0.01)
            upper_bound = X[col].quantile(0.99)
            self.thresholds_[col] = (lower_bound, upper_bound)

        return self

    def transform(self, X):
        """
        Áp dụng cắt cụt dữ liệu dựa trên các ngưỡng đã học ở bước fit.
        
        Parameters:
        X (pd.DataFrame): Dữ liệu cần biến đổi (Train hoặc Test).
        
        Returns:
        pd.DataFrame: Dữ liệu đã được cắt cụt ngoại lai.
        """
        # Copy dữ liệu để không làm biến đổi DataFrame gốc
        X_transformed = X.copy()
        
        if not isinstance(X_transformed, pd.DataFrame):
            X_transformed = pd.DataFrame(X_transformed)

        # Thực hiện ép (clip) giá trị về các ngưỡng đã lưu trong self.thresholds_
        for col in self.cols:
            if col in self.thresholds_:
                lower_bound, upper_bound = self.thresholds_[col]
                # Sử dụng np.clip để ép các giá trị ngoài giới hạn về đúng ngưỡng an toàn
                X_transformed[col] = np.clip(X_transformed[col], lower_bound, upper_bound)

        return X_transformed

