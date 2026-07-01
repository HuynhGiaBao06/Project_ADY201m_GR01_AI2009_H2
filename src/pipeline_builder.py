# ==========================================================
# pipeline_builder.py
#
# Xây dựng Pipeline cho dự án Credit Risk Classification
#
# Base:
# 1. Linear Pipeline
#    - RidgeClassifier
#
# 2. Tree Pipeline
#    - AdaBoost
#    - LightGBM
#
# FIT:
#   Chỉ học trên TRAIN
#
# TRANSFORM:
#   Áp dụng TRAIN / TEST
# ==========================================================


import sys

from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeClassifier

from sklearn.ensemble import AdaBoostClassifier

from lightgbm import LGBMClassifier



# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)



# ==========================================================
# IMPORT PREPROCESSING
# ==========================================================

from src.pre_processing.feature_engineer import FeatureCreator
from src.pre_processing.capping_outlier import OutlierCapping
from src.pre_processing.binning import AgeBinner
from src.pre_processing.k_fold import KFoldTargetEncoder



# ==========================================================
# LINEAR BASE PIPELINE
#
# Dành cho RidgeClassifier
#
# Vì mô hình tuyến tính:
# - nhạy với scale
# - nhạy với outlier
# ==========================================================


def build_linear_pipeline(model):

    return Pipeline([

        (
            "feature_creator",
            FeatureCreator()
        ),

        (
            "outlier_capping",
            OutlierCapping(

                columns=[

                    "person_income",

                    "loan_amnt"

                ]

            )
        ),

        (
            "age_binning",
            AgeBinner(
               cols = ["person_age"],
                bin_width=7,
                age_cap=62,
                epsilon=0.5

            )
        ),

        (
            "target_encoder",
            KFoldTargetEncoder(

                cols=[

                    "person_home_ownership",

                    "loan_intent"

                ],

            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "model",
            model
        )

    ])





# ==========================================================
# TREE BASE PIPELINE
#
# Dành cho:
# - AdaBoost
# - LightGBM
#
# Không cần:
# - Scaling
# - Capping
# ==========================================================


def build_tree_pipeline(model):

    return Pipeline([

        (
            "feature_creator",
            FeatureCreator()
        ),

        (
            "target_encoder",
            KFoldTargetEncoder(

                cols=[

                    "person_home_ownership",

                    "loan_intent"

                ],

            )
        ),

        (
            "model",
            model
        )

    ])





# ==========================================================
# MODEL BUILDERS
# ==========================================================


def build_ridge_pipeline():

    return build_linear_pipeline(

        RidgeClassifier()

    )




def build_adaboost_pipeline():

    return build_tree_pipeline(

        AdaBoostClassifier(

            random_state=42

        )

    )




def build_lightgbm_pipeline():

    return build_tree_pipeline(

        LGBMClassifier(

            random_state=42,

            n_estimators=200,

            learning_rate=0.05

        )

    )





# ==========================================================
# TEST
# ==========================================================


if __name__ == "__main__":


    ridge = build_ridge_pipeline()

    ada = build_adaboost_pipeline()

    lgbm = build_lightgbm_pipeline()


    print("LINEAR PIPELINE")

    print(ridge)


    print("\nTREE PIPELINE - ADABOOST")

    print(ada)


    print("\nTREE PIPELINE - LIGHTGBM")

    print(lgbm)