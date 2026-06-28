import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureCreator(BaseEstimator, TransformerMixin):
    """
    Imputation:
    - Numerical -> Median
    - Categorical -> Mode
    """

    def __init__(
        self,
        numerical_cols=None,
        categorical_cols=None,
        target_col="loan_status"
    ):

        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.target_col = target_col

        self.imputation_values_ = {}

    def fit(self, X, y=None):

        X = X.copy()

        if self.numerical_cols is None:

            self.numerical_cols_ = (
                X.select_dtypes(include=[np.number])
                .columns
                .tolist()
            )

            if self.target_col in self.numerical_cols_:
                self.numerical_cols_.remove(self.target_col)

        else:
            self.numerical_cols_ = self.numerical_cols

        if self.categorical_cols is None:

            self.categorical_cols_ = (
                X.select_dtypes(exclude=[np.number])
                .columns
                .tolist()
            )

        else:
            self.categorical_cols_ = self.categorical_cols

        self.imputation_values_ = {}

        # Numerical
        for col in self.numerical_cols_:

            if col not in X.columns:
                continue

            self.imputation_values_[col] = X[col].median()

        # Categorical
        for col in self.categorical_cols_:

            if col not in X.columns:
                continue

            mode_series = X[col].mode()

            self.imputation_values_[col] = (
                mode_series.iloc[0]
                if not mode_series.empty
                else "unknown"
            )

        return self

    def transform(self, X):

        if not self.imputation_values_:
            raise ValueError(
                "FeatureCreator has not been fitted. Call fit() first."
            )

        X_out = X.copy()

        for col, value in self.imputation_values_.items():

            if col not in X_out.columns:
                continue

            X_out[col] = X_out[col].fillna(value)

        return X_out