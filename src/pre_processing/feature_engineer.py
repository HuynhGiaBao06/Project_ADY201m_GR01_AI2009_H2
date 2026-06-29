"""Feature engineering"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Điền khuyết dựa trên thống kê Median/Mode và tính toán 3 biến phái sinh.
    """
    def __init__(self, numerical_cols=None, categorical_cols=None):
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.imputation_values_ = {}

    def fit(self, X, y=None):
        """Học Median cho biến số và Mode cho biến phân loại trên tập Train."""
        if self.numerical_cols is None:
            self.numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if self.categorical_cols is None:
            self.categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in self.numerical_cols:
            if col in X.columns:
                self.imputation_values_[col] = X[col].median()

        for col in self.categorical_cols:
            if col in X.columns:
                mode_series = X[col].mode()
                self.imputation_values_[col] = mode_series[0] if not mode_series.empty else "unknown"
                
        return self

    def transform(self, X):
        """Áp dụng điền khuyết và tạo công thức phái sinh."""
        X_out = X.copy()
        
        # 1. Điền khuyết (Imputation)
        for col, value in self.imputation_values_.items():
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(value)

        # 2. Đặc trưng: Độ chín muồi tài chính
        if 'person_age' in X_out.columns and 'cb_person_cred_hist_length' in X_out.columns:
            X_out['age_at_first_credit'] = X_out['person_age'] - X_out['cb_person_cred_hist_length']
            # Bắt buộc XÓA cột gốc để tránh đa cộng tuyến hoàn hảo
            X_out = X_out.drop(columns=['cb_person_cred_hist_length'])

        # 3. Đặc trưng: Áp lực chi phí vốn
        if 'loan_amnt' in X_out.columns and 'loan_int_rate' in X_out.columns:
            X_out['Estimated_Annual_Interest'] = X_out['loan_amnt'] * (X_out['loan_int_rate'] / 100.0)

        # 4. Đặc trưng: Năng lực tạo dòng tiền (Sử dụng epsilon tránh chia 0)
        if 'person_income' in X_out.columns and 'person_emp_length' in X_out.columns:
            X_out['Income_per_Employment_Year'] = X_out['person_income'] / (X_out['person_emp_length'] + 1e-6)  # Thêm epsilon để tránh chia cho 0
    

        

        # ==========================================
        # ĐOẠN CODE DEBUG TẠM THỜI (THÊM VÀO ĐÂY)
        # ==========================================
        nan_counts = X_out.isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]

        # 3. Chốt chặn Kiểm định tự động (QA Assertions)
        assert X_out.isna().sum().sum() == 0, \
            "🚨 PIPELINE ERROR: Pandas calculation unexpectedly generated NaN values!"

        return X_out
