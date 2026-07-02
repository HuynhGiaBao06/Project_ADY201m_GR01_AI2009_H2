"""
K-Fold: Tập Train được chia thành K=5. Giá trị mã hóa của một dòng chỉ được tính từ tỷ lệ 
rủi ro của 4 fold còn lại, chặn đứng rò rỉ nhãn cục bộ.
Smoothing: Cứu vớt các nhóm thiểu số có dữ liệu mỏng bằng cách kéo tỷ lệ 
vỡ nợ của chúng về gần tỷ lệ trung bình của toàn bộ dữ liệu.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        cols=None,
        target_col="loan_status",
        n_splits=5,
        m=100,
        random_state=42
    ):
        self.cols = cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.m = m
        self.random_state = random_state

        self.global_mean_ = None
        self.encoding_maps_ = {}

 
    # FIT
 
    def fit(self, X, y=None):
        """
        Học Risk Table trên toàn bộ Train.
        Được sử dụng khi Pipeline gọi fit().
        """

        df = X.copy()

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' không tồn tại trong DataFrame."
            )

        if self.cols is None:
            self.cols = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            if self.target_col in self.cols:
                self.cols.remove(self.target_col)

        self.global_mean_ = df[self.target_col].mean()

        self.encoding_maps_ = {}

        for col in self.cols:

            stats = (
                df.groupby(col)[self.target_col]
                .agg(["count", "sum"])
            )

            n_group = stats["count"]
            sum_y = stats["sum"]

            alpha = n_group / (n_group + self.m)

            smoothed = (
                alpha * (sum_y / n_group)
                + (1 - alpha) * self.global_mean_
            )

            self.encoding_maps_[col] = smoothed.to_dict()

        return self

 
    # FIT_TRANSFORM (OOF TRAIN)
 
    def fit_transform(self, X, y=None):
        """
        Train:
        - Tạo OOF Encoding.
        - Sau đó học Global Risk Table cho Test.
        """

        df = X.copy()

        if self.target_col not in df.columns:
            raise ValueError(
                f"Target column '{self.target_col}' không tồn tại trong DataFrame."
            )

        if self.cols is None:
            self.cols = df.select_dtypes(
                include=["object", "string", "category"]
            ).columns.tolist()

            if self.target_col in self.cols:
                self.cols.remove(self.target_col)

        self.global_mean_ = df[self.target_col].mean()

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

         
        # Khởi tạo cột OOF
         
        for col in self.cols:
            df[f"{col}_encoded"] = np.nan

         
        # OOF Encoding
         
        for train_idx, val_idx in skf.split(
            df,
            df[self.target_col]
        ):

            train_fold = df.iloc[train_idx]
            valid_fold = df.iloc[val_idx]

            fold_global_mean = train_fold[
                self.target_col
            ].mean()

            for col in self.cols:

                stats = (
                    train_fold.groupby(col)[self.target_col]
                    .agg(["count", "sum"])
                )

                n_group = stats["count"]
                sum_y = stats["sum"]

                alpha = n_group / (n_group + self.m)

                smoothed = (
                    alpha * (sum_y / n_group)
                    + (1 - alpha) * fold_global_mean
                )

                encoded_values = (
                    valid_fold[col]
                    .map(smoothed)
                    .fillna(fold_global_mean)
                )

                df.loc[
                    df.index[val_idx],
                    f"{col}_encoded"
                ] = encoded_values

         
        # Học Risk Table cuối cùng trên TOÀN TRAIN
         
        self.fit(df)

         
        # Thay thế biến categorical bằng biến encoded
         
        for col in self.cols:

            df[col] = df[f"{col}_encoded"]

            df.drop(
                columns=[f"{col}_encoded"],
                inplace=True
            )

        return df

 
    # TRANSFORM (TEST)
 
    def transform(self, X):
        """
        Test:
        Chỉ map theo Risk Table đã học.
        Không được tính toán lại.
        """

        df = X.copy()

        for col in self.cols:

            if col not in df.columns:
                continue

            df[col] = (
                df[col]
                .map(self.encoding_maps_[col])
                .fillna(self.global_mean_)
            )

        return df
