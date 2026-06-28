"""
pipeline_builder.py

Trái tim điều phối toàn bộ luồng tiền xử lý cho dự án
Credit Risk Modeling.

Nguyên tắc:
- Fit trên Train
- Transform trên Test
- Không Data Leakage
- Tách riêng pipeline cho:
    + Ridge Classifier
    + AdaBoost / LightGBM
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# ======================================================
# ADD PROJECT ROOT
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.path import (
    TRAIN_DATA_FILE,
    TEST_DATA_FILE
)

from src.pre_processing import (
    FeatureCreator,
    AgeBinner,
    OutlierCapping,
    KFoldTargetEncoder
)


# ======================================================
# DERIVED FEATURES
# ======================================================

class DerivedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Tạo biến phái sinh.

    age_at_first_credit
    Estimated_Annual_Interest
    Income_per_Employment_Year

    Đồng thời loại bỏ:
    cb_person_cred_hist_length
    """

    def __init__(self):
        self.epsilon = 1e-6

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        df = X.copy()

        required_cols = [
            "person_age",
            "cb_person_cred_hist_length",
            "loan_amnt",
            "loan_int_rate",
            "person_income",
            "person_emp_length"
        ]

        missing = [
            c for c in required_cols
            if c not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # ==========================
        # FEATURE 1
        # ==========================

        df["age_at_first_credit"] = (
            df["person_age"]
            - df["cb_person_cred_hist_length"]
        )

        # ==========================
        # FEATURE 2
        # ==========================

        df["Estimated_Annual_Interest"] = (
            df["loan_amnt"]
            * (df["loan_int_rate"] / 100)
        )

        # ==========================
        # FEATURE 3
        # ==========================

        df["Income_per_Employment_Year"] = (
            df["person_income"]
            / (
                df["person_emp_length"]
                + self.epsilon
            )
        )

        # ==========================
        # MULTICOLLINEARITY
        # ==========================

        df.drop(
            columns=["cb_person_cred_hist_length"],
            inplace=True,
            errors="ignore"
        )

        return df


# ======================================================
# PIPELINE BUILDER
# ======================================================

class PipelineBuilder:

    def __init__(
        self,
        target_col="loan_status"
    ):

        self.target_col = target_col

        self.pipeline_a = None
        self.pipeline_b = None

        self.scaler = StandardScaler()

    # --------------------------------------------------

    def _detect_columns(self, df):

        numeric_cols = (
            df.select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)

        categorical_cols = (
            df.select_dtypes(exclude=np.number)
            .columns
            .tolist()
        )

        return numeric_cols, categorical_cols

    # --------------------------------------------------

    def fit(self, train_df):

        num_cols, cat_cols = self._detect_columns(train_df)

        self.pipeline_a = Pipeline([
            (
                "imputation",
                FeatureCreator(
                    numerical_cols=num_cols,
                    categorical_cols=cat_cols,
                    target_col=self.target_col
                )
            ),

            (
                "derived",
                DerivedFeatureCreator()
            ),

            (
                "binning",
                AgeBinner(
                    col_name="person_age"
                )
            ),

            (
                "target_encoding",
                KFoldTargetEncoder(
                    cols=cat_cols,
                    target_col=self.target_col,
                    n_splits=5,
                    m=100
                )
            ),

            (
                "capping",
                OutlierCapping()
            )
        ])

        self.pipeline_b = Pipeline([
            (
                "imputation",
                FeatureCreator(
                    numerical_cols=num_cols,
                    categorical_cols=cat_cols,
                    target_col=self.target_col
                )
            ),

            (
                "derived",
                DerivedFeatureCreator()
            ),

            (
                "binning",
                AgeBinner(
                    col_name="person_age"
                )
            ),

            (
                "target_encoding",
                KFoldTargetEncoder(
                    cols=cat_cols,
                    target_col=self.target_col,
                    n_splits=5,
                    m=100
                )
            )
        ])

        train_a = self.pipeline_a.fit_transform(train_df)

        train_b = self.pipeline_b.fit_transform(train_df)

        numeric_after = train_a.select_dtypes(
            include=np.number
        ).columns.tolist()

        if self.target_col in numeric_after:
            numeric_after.remove(self.target_col)

        self.scaler.fit(
            train_a[numeric_after]
        )

        return train_a, train_b

    # --------------------------------------------------

    def transform(self, test_df):

        test_a = self.pipeline_a.transform(test_df)

        test_b = self.pipeline_b.transform(test_df)

        numeric_after = test_a.select_dtypes(
            include=np.number
        ).columns.tolist()

        if self.target_col in numeric_after:
            numeric_after.remove(self.target_col)

        test_a[numeric_after] = self.scaler.transform(
            test_a[numeric_after]
        )

        return test_a, test_b

    # --------------------------------------------------

    def fit_transform(self, train_df):

        train_a, train_b = self.fit(train_df)

        numeric_after = train_a.select_dtypes(
            include=np.number
        ).columns.tolist()

        if self.target_col in numeric_after:
            numeric_after.remove(self.target_col)

        train_a[numeric_after] = self.scaler.transform(
            train_a[numeric_after]
        )

        return train_a, train_b


# ======================================================
# MAIN TEST
# ======================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LOAD DATA")
    print("=" * 60)

    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    builder = PipelineBuilder()

    train_a, train_b = builder.fit_transform(
        train_df
    )

    test_a, test_b = builder.transform(
        test_df
    )

    print("\nBranch A (Ridge)")
    print(train_a.shape)
    print(test_a.shape)

    print("\nBranch B (Tree Models)")
    print(train_b.shape)
    print(test_b.shape)

    # ====================================
    # ASSERTIONS
    # ====================================

    assert train_a.isna().sum().sum() == 0
    assert test_a.isna().sum().sum() == 0

    assert train_b.isna().sum().sum() == 0
    assert test_b.isna().sum().sum() == 0

    assert (
        list(train_a.columns)
        == list(test_a.columns)
    )

    assert (
        list(train_b.columns)
        == list(test_b.columns)
    )

    print("\nALL ASSERTIONS PASSED")