"""
K-Fold: Tập Train được chia thành K=5. Giá trị mã hóa của một dòng chỉ được tính từ tỷ lệ 
rủi ro của 4 fold còn lại, chặn đứng rò rỉ nhãn cục bộ.
Smoothing: Cứu vớt các nhóm thiểu số có dữ liệu mỏng bằng cách kéo tỷ lệ 
vỡ nợ của chúng về gần tỷ lệ trung bình của toàn bộ dữ liệu.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

class KFoldTargetEncoder:
    """
    Module mã hóa rủi ro danh mục (Risk Target Encoding) kết hợp OOF và M-estimate Smoothing.
    Chuyên trách bởi: An Duy (Tuần 2)
    """
    def __init__(self, cols, target_col='loan_status', n_splits=5, m=100):
        self.cols = cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.m = m
        self.encoding_maps = {}
        self.global_mean = None

    def fit_transform(self, train_df):
        """
        GIAI ĐOẠN 1: Học từ tập Train bằng cơ chế Out-of-Fold (OOF).
        Tuyệt đối không chạm vào tập Test.
        """
        df = train_df.copy()
        # Tính tỷ lệ vỡ nợ trung bình toàn cục trên toàn bộ tập Train
        self.global_mean = df[self.target_col].mean()
        
        # Khởi tạo các cột mã hóa mới
        encoded_cols = []
        for col in self.cols:
            encoded_name = f'{col}_encoded'
            df[encoded_name] = np.nan
            encoded_cols.append(encoded_name)
            
        # Triển khai Stratified K-Fold để giữ cân bằng tỷ lệ nợ xấu giữa các fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for train_idx, val_idx in skf.split(df, df[self.target_col]):
            # Chia tách nội bộ tập Train thành 4 folds (X_tr) và 1 fold validation (X_val)
            X_tr = df.iloc[train_idx]
            X_val = df.iloc[val_idx]
            
            for col in self.cols:
                # Tính toán thống kê trên 4 folds nền
                stats = X_tr.groupby(col)[self.target_col].agg(['count', 'sum'])
                fold_global_mean = X_tr[self.target_col].mean()
                
                # Áp dụng công thức M-estimate Smoothing
                n_group = stats['count']
                sum_y = stats['sum']
                alpha = n_group / (n_group + self.m)
                smoothed_values = alpha * (sum_y / n_group) + (1 - alpha) * fold_global_mean
                
                # Ánh xạ giá trị đã làm mượt vào fold validation hiện tại
                encoded_name = f'{col}_encoded'
                df.loc[df.index[val_idx], encoded_name] = X_val[col].map(smoothed_values).fillna(fold_global_mean)

        # HỌC CUỐI CÙNG: Tính toán bản đồ ánh xạ trên TOÀN BỘ tập Train để dùng cho tập Test về sau
        for col in self.cols:
            stats = df.groupby(col)[self.target_col].agg(['count', 'sum'])
            n_group = stats['count']
            sum_y = stats['sum']
            alpha = n_group / (n_group + self.m)
            
            # Lưu bảng ánh xạ chính thức vào từ điển của Object
            final_smoothed = alpha * (sum_y / n_group) + (1 - alpha) * self.global_mean
            self.encoding_maps[col] = final_smoothed.to_dict()
            
        return df

    def transform(self, test_df):
        """
        GIAI ĐOẠN 2: Chỉ ánh xạ (Map) từ điển đã học sang tập Test.
        Tuyệt đối không tính toán lại hay sử dụng nhãn của tập Test.
        """
        df = test_df.copy()
        
        for col in self.cols:
            encoded_name = f'{col}_encoded'
            # Map bằng từ điển đã fit từ tập Train. Nếu xuất hiện danh mục lạ -> điền Global Mean
            df[encoded_name] = df[col].map(self.encoding_maps[col]).fillna(self.global_mean)
            
        return df

    