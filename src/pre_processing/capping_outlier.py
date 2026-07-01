# ==========================================================
# capping_outlier.py
#
# Module xử lý ngoại lai bằng Percentile Capping
# Dự án: Credit Risk Classification
#
# Nguyên tắc:
# - FIT chỉ trên TRAIN
# - TRANSFORM trên TRAIN + TEST
# - Không Data Leakage
#
# Triết lý:
# "Thà từ chối nhầm khách tốt,
#  còn hơn cho vay nhầm khách xấu"
# ==========================================================


import pandas as pd
import numpy as np

import sys

from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin



# ==========================================================
# PROJECT ROOT CONFIG
# ==========================================================


# src/pre_processing/capping_outlier.py

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
# DATA PATH - CHỜ PIPELINE NẠP DATA
# ==========================================================


DATA_PATH = (
    PROJECT_ROOT
    /
    "data"
    /
    "checkpoint_data"
    /
    "train_df.csv"
)



def load_data(path=DATA_PATH):

    """
    Hàm hỗ trợ kiểm tra dữ liệu.

    Pipeline thật sẽ truyền DataFrame
    trực tiếp vào fit() và transform()
    """

    print("==============================")
    print("LOAD DATA")
    print("==============================")

    print("PATH:")
    print(path)


    assert Path(path).exists(), (
        f"Không tìm thấy file dữ liệu: {path}"
    )


    df = pd.read_csv(path)


    print("Shape:")
    print(df.shape)


    print("\nColumns:")
    print(df.columns.tolist())


    return df





# ==========================================================
# OUTLIER CAPPING TRANSFORMER
# ==========================================================


class OutlierCapping(
    BaseEstimator,
    TransformerMixin
):


    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        columns: list = None
    ):


        self.lower_percentile = lower_percentile

        self.upper_percentile = upper_percentile

        self.columns = columns


        # Lưu ngưỡng học từ TRAIN
        #
        # {
        #   "person_income":
        #       {
        #          "lower": value,
        #          "upper": value
        #       }
        # }
        #

        self.thresholds_ = {}





    # ======================================================
    # FIT
    # ======================================================


    def fit(
        self,
        X: pd.DataFrame,
        y=None
    ):


        assert isinstance(
            X,
            pd.DataFrame
        ), (
            "Lỗi: Dữ liệu đầu vào phải là pandas.DataFrame"
        )



        assert (
            self.columns is not None
        ), (
            "Lỗi: Chưa truyền danh sách cột cần xử lý"
        )



        for col in self.columns:


            assert (
                col in X.columns
            ), (
                f"Lỗi: Không tồn tại cột {col}"
            )



        # Học percentile từ TRAIN
        for col in self.columns:


            lower_limit = np.percentile(
                X[col].dropna(),
                self.lower_percentile
            )


            upper_limit = np.percentile(
                X[col].dropna(),
                self.upper_percentile
            )



            self.thresholds_[col] = {


                "lower":
                    lower_limit,


                "upper":
                    upper_limit

            }



        return self





    # ======================================================
    # TRANSFORM
    # ======================================================


    def transform(
        self,
        X: pd.DataFrame
    ):


        assert isinstance(
            X,
            pd.DataFrame
        ), (
            "Lỗi: Dữ liệu transform phải là pandas.DataFrame"
        )



        X_out = X.copy()



        for col in self.columns:


            assert (
                col in X_out.columns
            ), (
                f"Lỗi: Không tồn tại cột {col}"
            )



            lower_limit = (
                self.thresholds_[col]
                ["lower"]
            )


            upper_limit = (
                self.thresholds_[col]
                ["upper"]
            )



            # Cắt giá trị ngoại lai
            #
            # nhỏ hơn 1%  -> đưa về 1%
            # lớn hơn 99% -> đưa về 99%
            #

            X_out[col] = (
                X_out[col]
                .clip(
                    lower=lower_limit,
                    upper=upper_limit
                )
            )



        # ==================================================
        # ASSERT KIỂM TRA SAU CAPPING
        # ==================================================


        for col in self.columns:


            lower_limit = (
                self.thresholds_[col]
                ["lower"]
            )


            upper_limit = (
                self.thresholds_[col]
                ["upper"]
            )


            assert not (
                (
                    X_out[col] < lower_limit
                )
                |
                (
                    X_out[col] > upper_limit
                )

            ).any(), (

                f"Lỗi: {col} vẫn còn giá trị ngoài vùng capping"
            )



        return X_out





# ==========================================================
# TEST CHẠY ĐỘC LẬP
# ==========================================================


if __name__ == "__main__":


    df = load_data()



    capper = OutlierCapping(

        columns=[

            "person_income",

            "loan_amnt"

        ]

    )



    result = capper.fit_transform(
        df
    )


    print(result.head())