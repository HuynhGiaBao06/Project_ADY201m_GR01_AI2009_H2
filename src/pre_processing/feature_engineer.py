"""Feature engineering"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Module tự động hóa việc Học quy luật thống kê trên tập Train (Fit Phase)
    và Áp dụng để xử lý dữ liệu trên cả hai tập Train/Test (Transform Phase).
    """
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        
        # Khởi tạo bộ "từ điển" chứa các quy luật thống kê đã học từ tập Train
        self.imputation_values_ = {}
        self.capping_bounds_ = {}
        
    def fit(self, X, y=None):
        """
        GIAI ĐOẠN 1: "HỌC" (FIT PHASE) — CHỈ THỰC HIỆN TRÊN TẬP TRAIN.
        Khóa chặt tập Test để rút trích thông số khách quan, chống Data Leakage tuyệt đối.
        """
        # Tự động phân loại cột nếu danh sách chưa được định nghĩa tường minh
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
            
        # 1.1. Học thông số đối với Biến số (Numerical Variables)
        for col in self.numerical_cols:
            if col in X.columns:
                # Tiêu chí 1: Học Trung vị (Median) để chuẩn bị điền khuyết
                self.imputation_values_[col] = X[col].median()
                
                # Tiêu chí 2: Tính mốc phân vị (1% và 99%) thiết lập ranh giới an toàn cho Capping
                self.capping_bounds_[col] = {
                    'lower': X[col].quantile(0.01),
                    'upper': X[col].quantile(0.99)
                }
                
        # 1.2. Học thông số đối với Biến phân loại (Categorical Variables)
        for col in self.categorical_cols:
            if col in X.columns:
                # Tiêu chí 1: Học Yếu vị (Mode) để chuẩn bị điền khuyết
                mode_series = X[col].mode()
                # Tránh lỗi nếu cột trống hoàn toàn, mặc định gán nhãn dự phòng
                self.imputation_values_[col] = mode_series[0] if not mode_series.empty else "unknown"
                
        return self
        
    def transform(self, X):
        """
        GIAI ĐOẠN 2: "ÁP DỤNG" (TRANSFORM PHASE) — THỰC THI TRÊN CẢ TRAIN VÀ TEST.
        Dùng chính xác bộ thông số từ bước Fit, tuyệt đối không tính toán lại.
        """
        # NGUYÊN TẮC SỐNG CÒN 1: Tuyệt đối không fillna(inplace=True). 
        # Phải dùng cơ chế trả về một bản sao rõ ràng (.copy()) để tránh side-effects.
        X_out = X.copy()
        
        # Tiêu chí 3: Áp dụng các giá trị đã học để thực hiện Điền khuyết (Imputation)
        for col, value in self.imputation_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(value)
                
        return X_out

    def apply_capping(self, X):
        """
        HÀM RẼ NHÁNH TỐI ƯU MÔ HÌNH (Dành riêng cho Nhánh A - Ridge Classifier).
        Ép các điểm ngoại lai dị biệt về lại ranh giới 1% và 99% đã học từ tập Train.
        """
        X_out = X.copy()
        for col, bounds in self.capping_bounds_.items():
            if col in X_out.columns:
                X_out[col] = np.clip(X_out[col], bounds['lower'], bounds['upper'])
        return X_out