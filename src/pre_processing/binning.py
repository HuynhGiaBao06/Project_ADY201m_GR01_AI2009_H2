# ==========================================================
# binning.py
# Module xử lý binning cho dự án Credit Risk Classification
# ==========================================================


import pandas as pd
import numpy as np
import sys

from pathlib import Path
from typing import List

from sklearn.base import BaseEstimator, TransformerMixin



# ==========================================================
# PROJECT PATH CONFIG
# ==========================================================

# src/preprocessing/binning.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

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
    Hàm load dữ liệu.

    Không dùng trong sklearn Pipeline thật.
    Chỉ hỗ trợ:
    - test module
    - chạy debug
    - pipeline gọi từ bên ngoài
    """

    print("==============================")
    print("LOADING DATA")
    print("==============================")

    print("DATA PATH:")
    print(path)


    df = pd.read_csv(path)


    print("\nDATA SHAPE:")
    print(df.shape)


    print("\nCOLUMNS:")
    print(df.columns.tolist())


    print("\nSAMPLE:")
    print(df.head())


    return df




# ==========================================================
# AGE BINNING TRANSFORMER
# ==========================================================


class AgeBinner(
    BaseEstimator,
    TransformerMixin
):


    def __init__(
        self,
        cols: List[str],
        bin_width: int = 7,
        age_cap: int = 62,
        epsilon: float = 0.5
    ):


        self.cols = cols

        self.bin_width = bin_width

        self.age_cap = age_cap

        self.epsilon = epsilon



        self.bins_ = {}

        self.labels_ = {}

        self.woe_dicts_ = {}

        self.global_woe_ = 0.0




    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ):


        self.log.info(
            "FIT START - TRAIN DATA"
        )


        assert isinstance(
            X,
            pd.DataFrame
        ), "Input phải là DataFrame"



        assert (
            y is not None
        ), "Thiếu target y"



        total_good = (
            y == 0
        ).sum()


        total_bad = (
            y == 1
        ).sum()



        self.global_woe_ = np.log(
            (
                total_good + self.epsilon
            )
            /
            (
                total_bad + self.epsilon
            )
        )



        for col in self.cols:


            min_age = int(
                X[col].min()
            )



            # Tạo bins
            core_bins = list(
                range(
                    min_age,
                    self.age_cap,
                    self.bin_width
                )
            )



            if core_bins[-1] != self.age_cap:

                core_bins.append(
                    self.age_cap
                )



            bins = (
                [-np.inf]
                +
                core_bins
                +
                [np.inf]
            )



            labels = []



            for i in range(
                len(bins)-1
            ):

                if i == 0:

                    labels.append(
                        f"<{core_bins[0]}"
                    )


                elif i == len(bins)-2:

                    labels.append(
                        f">={core_bins[-1]}"
                    )


                else:

                    labels.append(
                        f"{bins[i]}-{bins[i+1]-1}"
                    )



            self.bins_[col] = bins

            self.labels_[col] = labels



        return self




    def transform(
        self,
        X: pd.DataFrame
    ):


        assert isinstance(
            X,
            pd.DataFrame
        ), "Input transform phải là DataFrame"



        X_out = X.copy()



        for col in self.cols:


            binned = pd.cut(

                X_out[col],

                bins=self.bins_[col],

                labels=self.labels_[col],

                right=False
            )


            X_out[col] = (

                binned

                .map(
                    self.woe_dicts_.get(
                        col,
                        {}
                    )
                )

                .fillna(
                    self.global_woe_
                )

            )



        assert (

            X_out.isna()
            .sum()
            .sum()

            ==

            0

        ), "Transform sinh NaN"

        return X_out

# ==========================================================
# TEST MODULE
# ==========================================================

if __name__ == "__main__":
    df = load_data()
    print(
        "\nModule binning.py ready"
    )