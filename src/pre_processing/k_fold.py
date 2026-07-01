# ==========================================================
# k_fold.py
#
# K-Fold Target Encoding + M-estimate Smoothing
# Credit Risk Classification
#
# FIT:
#   Chỉ học trên TRAIN
#
# TRANSFORM:
#   Áp dụng TRAIN / TEST
#   Không dùng target TEST
#
# Triết lý:
# "Thà từ chối nhầm khách tốt,
#  còn hơn cho vay nhầm khách xấu"
# ==========================================================


import sys

from pathlib import Path

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold



# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)



# ==========================================================
# K-FOLD TARGET ENCODER
# ==========================================================


class KFoldTargetEncoder(
    BaseEstimator,
    TransformerMixin
):


    def __init__(
        self,
        cols=None,
        n_splits=5,
        m=100
    ):

        self.cols = cols

        self.n_splits = n_splits

        self.m = m


        # Mapping dùng cho TEST

        self.encoding_maps_ = {}


        # Tỷ lệ nợ xấu toàn TRAIN

        self.global_mean_ = None



    # ======================================================
    # FIT
    #
    # Học encoding từ TRAIN
    # ======================================================


    def fit(
        self,
        X,
        y
    ):


        assert isinstance(
            X,
            pd.DataFrame
        ), "Input fit phải là pandas.DataFrame"


        assert self.cols is not None, (
            "Chưa truyền danh sách cột encode"
        )


        assert len(X) == len(y), (
            "X và y không cùng số dòng"
        )


        for col in self.cols:

            assert col in X.columns, (
                f"Không tồn tại cột {col}"
            )



        # Global bad rate TRAIN

        self.global_mean_ = y.mean()



        # ==================================================
        # OOF ENCODING
        #
        # Mỗi dòng chỉ nhìn thấy 4 fold còn lại
        # ==================================================


        oof_encoded = pd.DataFrame(
            index=X.index
        )


        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )



        for train_idx, valid_idx in skf.split(
            X,
            y
        ):


            X_train = X.iloc[train_idx]

            y_train = y.iloc[train_idx]


            X_valid = X.iloc[valid_idx]


            fold_mean = y_train.mean()



            for col in self.cols:


                stats = (

                    pd.DataFrame(
                        {
                            "category": X_train[col],
                            "target": y_train
                        }
                    )

                    .groupby("category")
                    ["target"]

                    .agg(
                        [
                            "count",
                            "sum"
                        ]
                    )

                )



                count = stats["count"]

                bad = stats["sum"]



                alpha = (

                    count /

                    (
                        count + self.m
                    )

                )



                mapping = (

                    alpha *

                    (
                        bad / count
                    )

                    +

                    (1 - alpha)

                    *

                    fold_mean

                )



                oof_encoded.loc[
                    X_valid.index,
                    col + "_encoded"
                ] = (

                    X_valid[col]

                    .map(mapping)

                    .fillna(
                        fold_mean
                    )

                )



        # ==================================================
        # FINAL MAPPING
        #
        # Học toàn bộ TRAIN
        # Dùng cho TEST
        # ==================================================


        for col in self.cols:


            stats = (

                pd.DataFrame(
                    {
                        "category": X[col],
                        "target": y
                    }
                )

                .groupby("category")
                ["target"]

                .agg(
                    [
                        "count",
                        "sum"
                    ]
                )

            )



            count = stats["count"]

            bad = stats["sum"]



            alpha = (

                count /

                (
                    count + self.m
                )

            )



            mapping = (

                alpha *

                (
                    bad / count
                )

                +

                (1 - alpha)

                *

                self.global_mean_

            )



            self.encoding_maps_[col] = (
                mapping.to_dict()
            )



        return self



    # ======================================================
    # TRANSFORM
    #
    # Map theo rule đã học
    # ======================================================


    def transform(
        self,
        X
    ):


        assert isinstance(
            X,
            pd.DataFrame
        ), "Input transform phải là pandas.DataFrame"



        X_out = X.copy()



        for col in self.cols:


            encoded_col = (
                col +
                "_encoded"
            )


            X_out[encoded_col] = (

                X_out[col]

                .map(
                    self.encoding_maps_[col]
                )

                .fillna(
                    self.global_mean_
                )

            )



        return X_out