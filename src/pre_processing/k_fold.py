import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    KFold Target Encoding + M-estimate Smoothing
    """

    def __init__(
        self,
        cols,
        target_col="loan_status",
        n_splits=5,
        m=100
    ):

        self.cols = cols
        self.target_col = target_col
        self.n_splits = n_splits
        self.m = m

        self.encoding_maps = {}
        self.global_mean = None

    def fit(self, X, y=None):

        df = X.copy()

        self.global_mean = df[self.target_col].mean()

        self.encoding_maps = {}

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
                + (1 - alpha) * self.global_mean
            )

            self.encoding_maps[col] = smoothed.to_dict()

        return self

    def fit_transform(self, X, y=None):

        df = X.copy()

        self.global_mean = df[self.target_col].mean()

        encoded_df = df.copy()

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42
        )

        for col in self.cols:

            encoded_name = f"{col}_encoded"

            encoded_df[encoded_name] = np.nan

        for train_idx, val_idx in skf.split(
            df,
            df[self.target_col]
        ):

            train_fold = df.iloc[train_idx]
            val_fold = df.iloc[val_idx]

            fold_mean = train_fold[self.target_col].mean()

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
                    + (1 - alpha) * fold_mean
                )

                encoded_name = f"{col}_encoded"

                encoded_df.loc[
                    encoded_df.index[val_idx],
                    encoded_name
                ] = (
                    val_fold[col]
                    .map(smoothed)
                    .fillna(fold_mean)
                )

        self.fit(df)
        encoded_df.drop(columns=self.cols, inplace=True)

        return encoded_df

    def transform(self, X):

        if not self.encoding_maps:
            raise ValueError(
                "KFoldTargetEncoder has not been fitted."
            )

        df = X.copy()

        for col in self.cols:

            encoded_name = f"{col}_encoded"

            df[encoded_name] = (
                df[col]
                .map(self.encoding_maps[col])
                .fillna(self.global_mean)
            )

        df.drop(columns=self.cols, inplace=True)

        return df