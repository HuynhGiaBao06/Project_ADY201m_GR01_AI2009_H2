from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.path import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE,
)

from src.pre_processing.feature_engineer import FeatureCreator
from src.pre_processing.capping_outlier import OutlierCapping
from src.pre_processing.feature_scaling import FeatureScaler
#from src.pre_processing.binning import AgeBinner
#from src.pre_processing.k_fold import KFoldTargetEncoder
''' k_fold bị thiếu 1 tham số '''


class PipelineBuilder:

    def __init__(self, model_type="ridge", target_col="loan_status"):

        self.model_type = model_type.lower()
        self.target_col = target_col

        self.pipeline = None

        self.numerical_cols = None
        self.categorical_cols = None

    def _detect_columns(self, df):

        num_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if self.target_col in num_cols:
            num_cols.remove(self.target_col)

        cat_cols = df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        return num_cols, cat_cols

    def _common_steps(self):

        return [

            (
                "feature_creator",
                FeatureCreator(
                    numerical_cols=self.numerical_cols,
                    categorical_cols=self.categorical_cols,
                ),
            ),

            #(
            #    "age_binning",
            #    AgeBinner(),
            #),

            #(
            #    "target_encoding",
            #    KFoldTargetEncoder(...)
            #),

        ]

    def _build_pipeline(self):

        steps = self._common_steps()

        if self.model_type == "ridge":

            steps.extend(

                [

                    (
                        "outlier_capping",
                        OutlierCapping(
                            numerical_cols=None
                        ),
                    ),

                    (
                        "feature_scaling",
                        FeatureScaler(
                            numerical_cols=None
                        ),
                    ),

                ]

            )

        elif self.model_type in (

            "adaboost",
            "lightgbm",

        ):

            pass

        else:

            raise ValueError(

                f"Unsupported model: {self.model_type}"

            )

        self.pipeline = Pipeline(steps)

    def fit(self, train_df):

        self.numerical_cols, self.categorical_cols = \
            self._detect_columns(train_df)

        self._build_pipeline()

        self.pipeline.fit(train_df)

        return self

    def transform(self, df):

        return self.pipeline.transform(df)

    def fit_transform(self, train_df):

        self.fit(train_df)

        return self.transform(train_df)


if __name__ == "__main__":

    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    ridge = PipelineBuilder("ridge")

    ridge.fit(train_df)

    train_ridge = ridge.transform(train_df)
    test_ridge = ridge.transform(test_df)

    print("\nRIDGE")
    print(train_ridge.shape)
    print(test_ridge.shape)

    assert train_ridge.isna().sum().sum() == 0
    assert test_ridge.isna().sum().sum() == 0

    print("\nALL TEST PASSED")

