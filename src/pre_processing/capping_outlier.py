# Xử lý outlier và scaling
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierCapping(BaseEstimator, TransformerMixin):
    """
    Module xử lý ngoại lai bằng Winsorization (cắt cụt 1% và 99%).
    """
    def __init__(self, cols=None):
        self.cols = cols
        self.thresholds_ = {}  # Lưu trữ bộ quy luật đã học (Từ điển Fit)

    def fit(self, X, y=None):
        """Học các mốc phân vị 1% và 99% từ tập Train."""
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        if self.cols is None:
            self.cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in self.cols:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            self.thresholds_[col] = (lower_bound, upper_bound)
            
        return self

    def transform(self, X):
        """Áp dụng cắt cụt (Clip) dựa trên ngưỡng Train, không tính lại."""
        X_transformed = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        
        for col in self.cols:
            if col in self.thresholds_ and col in X_transformed.columns:
                lower, upper = self.thresholds_[col]
                X_transformed[col] = np.clip(X_transformed[col], lower, upper)
                
        return X_transformed