#Lắp ráp các class trên thành luồng Pipeline hoàn chỉnh
"""
Xây dựng Pipeline tiền xử lý dữ liệu.

Module này đóng vai trò Factory, chịu trách nhiệm kết hợp
các bước tiền xử lý phù hợp với từng nhóm mô hình.
"""

from sklearn.pipeline import Pipeline

from pre_processing import (
    FeatureCreator,
    OutlierCapping,
    FeatureScaler,
    AgeBinner,
)


class PipelineBuilder:
    """
    Factory xây dựng các Pipeline tiền xử lý.

    Hiện hỗ trợ:

    - Linear Pipeline (Ridge Classifier)
    - Tree Pipeline (AdaBoost, LightGBM)
    """

    @staticmethod
    def build_ridge_pipeline(
        numerical_cols=None,
        categorical_cols=None,
    ):
        """
        Pipeline dành cho các mô hình tuyến tính.

        Quy trình:

            Missing Value
                    ↓
            Feature Engineering
                    ↓
            Outlier Capping
                    ↓
            Standard Scaling
                    ↓
            Age Binning
        """

        return Pipeline(
            steps=[
                (
                    "feature_creator",
                    FeatureCreator(
                        numerical_cols=numerical_cols,
                        categorical_cols=categorical_cols,
                    ),
                ),
                (
                    "outlier_capping",
                    OutlierCapping(
                        numerical_cols=numerical_cols,
                    ),
                ),
                (
                    "feature_scaling",
                    FeatureScaler(
                        numerical_cols=numerical_cols,
                    ),
                ),
                (
                    "age_binning",
                    AgeBinner(),
                ),
            ]
        )

    @staticmethod
    def build_tree_pipeline(
        numerical_cols=None,
        categorical_cols=None,
    ):
        """
        Pipeline dành cho các mô hình Tree-based.

        Áp dụng cho:

        - AdaBoost
        - LightGBM

        Không thực hiện:

        - Outlier Capping
        - Feature Scaling
        """

        return Pipeline(
            steps=[
                (
                    "feature_creator",
                    FeatureCreator(
                        numerical_cols=numerical_cols,
                        categorical_cols=categorical_cols,
                    ),
                ),
                (
                    "age_binning",
                    AgeBinner(),
                ),
            ]
        )

    @staticmethod
    def get_support_models():
     """
     Trả về danh sách các mô hình hiện được hỗ trợ.
     """
     return (
         "ridge",
         "adaboost",
         "lightgbm",
         )

    @staticmethod
    def build_pipeline(
        model_type,
        numerical_cols=None,
        categorical_cols=None,
    ):
        """
        Tự động lựa chọn Pipeline theo tên mô hình.
        """

        model_type = model_type.lower()

        if model_type == "ridge":
            return PipelineBuilder.build_ridge_pipeline(
                numerical_cols,
                categorical_cols,
            )

        if model_type in (
            "adaboost",
            "lightgbm",
        ):
            return PipelineBuilder.build_tree_pipeline(
                numerical_cols,
                categorical_cols,
            )

        raise ValueError(
            f"Unsupported model: {model_type}"
        )