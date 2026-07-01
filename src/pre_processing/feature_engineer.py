# ==========================================================
# feature_creator.py
#
# Module Feature Engineering cho dự án
# Phân loại Rủi ro Tín dụng (30.000 hồ sơ)
#
# Nhiệm vụ:
# - Học quy luật xử lý dữ liệu từ tập TRAIN
# - Áp dụng lại cho TRAIN và TEST
#
# Nguyên tắc:
# FIT  : Chỉ học từ Train
# TRANSFORM : Không học lại, chỉ áp dụng
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


# src/pre_processing/feature_creator.py

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
    Hàm hỗ trợ test module.

    Trong pipeline thật:
    DataFrame sẽ truyền trực tiếp vào:
        fit()
        transform()
    """

    print("==============================")
    print("LOAD DATA")
    print("==============================")

    print("PATH:")
    print(path)


    assert Path(path).exists(), (
        f"Không tìm thấy file: {path}"
    )


    df = pd.read_csv(path)


    print("Shape:")
    print(df.shape)


    print("Columns:")
    print(df.columns.tolist())


    return df





# ==========================================================
# FEATURE CREATOR CLASS
# ==========================================================


class FeatureCreator(
    BaseEstimator,
    TransformerMixin
):


    def __init__(
        self,
        numerical_cols=None,
        categorical_cols=None
    ):


        # Danh sách cột số
        self.numerical_cols = numerical_cols


        # Danh sách cột category
        self.categorical_cols = categorical_cols



        # Các giá trị học từ TRAIN

        # Numerical:
        # median dùng để điền missing

        self.numeric_imputation_values_ = {}



        # Categorical:
        # mode dùng để điền missing

        self.category_imputation_values_ = {}




    # ======================================================
    # FIT PHASE
    #
    # Học quy luật từ TRAIN
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
            "Lỗi: Input phải là pandas.DataFrame"
        )



        # Nếu chưa truyền danh sách cột
        # tự động phân loại

        if self.numerical_cols is None:


            self.numerical_cols = (

                X
                .select_dtypes(
                    include=np.number
                )
                .columns
                .tolist()

            )



        if self.categorical_cols is None:


            self.categorical_cols = (

                X
                .select_dtypes(
                    exclude=np.number
                )
                .columns
                .tolist()

            )




        # ==================================================
        # HỌC MISSING VALUE CHO BIẾN SỐ
        # ==================================================


        for col in self.numerical_cols:


            assert (
                col in X.columns
            ), (
                f"Không tồn tại cột {col}"
            )



            # Median ổn định hơn Mean
            # khi dữ liệu tín dụng có outlier

            self.numeric_imputation_values_[col] = (

                X[col]
                .median()

            )





        # ==================================================
        # HỌC MISSING VALUE CHO BIẾN CATEGORY
        # ==================================================


        for col in self.categorical_cols:


            assert (
                col in X.columns
            ), (
                f"Không tồn tại cột {col}"
            )



            mode_value = (
                X[col]
                .mode()
            )



            if not mode_value.empty:


                self.category_imputation_values_[col] = (

                    mode_value.iloc[0]

                )


            else:


                # trường hợp cột rỗng hoàn toàn

                self.category_imputation_values_[col] = (

                    "unknown"

                )




        return self





    # ======================================================
    # TRANSFORM PHASE
    #
    # Áp dụng quy luật đã học
    # ======================================================


    def transform(
        self,
        X: pd.DataFrame
    ):



        assert isinstance(
            X,
            pd.DataFrame
        ), (
            "Lỗi: Input transform phải là pandas.DataFrame"
        )



        # Không sửa dữ liệu gốc

        X_out = X.copy()





        # ==================================================
        # IMPUTATION NUMERICAL
        # ==================================================


        for col, value in (

            self.numeric_imputation_values_
            .items()

        ):


            if col in X_out.columns:


                X_out[col] = (

                    X_out[col]
                    .fillna(value)

                )





        # ==================================================
        # IMPUTATION CATEGORY
        # ==================================================


        for col, value in (

            self.category_imputation_values_
            .items()

        ):


            if col in X_out.columns:


                X_out[col] = (

                    X_out[col]
                    .fillna(value)

                )





        # ==================================================
        # ASSERTION KIỂM TRA
        # ==================================================


        assert (

            X_out.shape[1]

            ==

            X.shape[1]

        ), (

            "Lỗi: Số lượng cột bị thay đổi"
        )



        return X_out





# ==========================================================
# TEST MODULE
# ==========================================================


if __name__ == "__main__":



    df = load_data()



    creator = FeatureCreator()



    result = creator.fit_transform(
        df
    )



    print("\nKết quả sau Feature Engineering:")

    print(
        result.head()
    )