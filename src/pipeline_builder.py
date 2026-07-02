from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pre_processing.feature_engineer import FeatureCreator
from pre_processing.binning import AgeBinner
from pre_processing.k_fold import KFoldTargetEncoder
from pre_processing.capping_outlier import OutlierCapping


class CreditRiskPipelineBuilder:
    def __init__(self, target_col="loan_status"):
        self.target_col = target_col

        self.feature_creator = FeatureCreator()
        self.age_binner = AgeBinner()
        self.kfold_encoder = KFoldTargetEncoder(
            target_col=self.target_col
        )
        self.outlier_capper = OutlierCapping()
        self.scaler = StandardScaler()

    def build_and_run(self, train_df, test_df):
        print("🚀 Bắt đầu chạy Data Pipeline...")

        # =====================================================
        # 1. Tách Target
        # =====================================================
        y_train = train_df[self.target_col].copy()
        y_test = test_df[self.target_col].copy()

        X_train = train_df.drop(columns=[self.target_col]).copy()
        X_test = test_df.drop(columns=[self.target_col]).copy()

        # =====================================================
        # 2. Feature Engineering
        # =====================================================
        print("⏳ Feature Engineering...")

        self.feature_creator.fit(X_train)

        X_train = self.feature_creator.transform(X_train)
        X_test = self.feature_creator.transform(X_test)

        # =====================================================
        # 3. Binning
        # =====================================================
        print("⏳ Age Binning...")

        self.age_binner.fit(X_train)

        X_train = self.age_binner.transform(X_train)
        X_test = self.age_binner.transform(X_test)

        # =====================================================
        # 4. K-Fold Target Encoding
        # =====================================================
        print("⏳ Risk Target Encoding...")

        X_train[self.target_col] = y_train

        X_train = self.kfold_encoder.fit_transform(X_train)
        X_test = self.kfold_encoder.transform(X_test)

        X_train.drop(columns=[self.target_col], inplace=True)

        train_cols = list(X_train.columns)

        # =====================================================
        # NHÁNH B
        # AdaBoost + LightGBM
        # =====================================================
        print("🌿 Tạo Nhánh B (Tree Models)...")

        X_train_B = X_train.copy()
        X_test_B = X_test.copy()

        # =====================================================
        # NHÁNH A
        # Ridge Classifier
        # =====================================================
        print("📈 Tạo Nhánh A (Linear Models)...")

        X_train_A = X_train.copy()
        X_test_A = X_test.copy()

        # Outlier Capping
        self.outlier_capper.fit(X_train_A)

        X_train_A = self.outlier_capper.transform(X_train_A)
        X_test_A = self.outlier_capper.transform(X_test_A)

        # Scaling
        num_cols = (
            X_train_A
            .select_dtypes(include=[np.number])
            .columns
            .tolist()
        )

        self.scaler.fit(X_train_A[num_cols])

        X_train_A[num_cols] = self.scaler.transform(
            X_train_A[num_cols]
        )

        X_test_A[num_cols] = self.scaler.transform(
            X_test_A[num_cols]
        )

        # =====================================================
        # ASSERTIONS
        # =====================================================
        print("🛡️ Kiểm định dữ liệu...")

        datasets = [
            ("Train A", X_train_A),
            ("Test A", X_test_A),
            ("Train B", X_train_B),
            ("Test B", X_test_B),
        ]

        for name, df in datasets:

            assert (
                df.isna().sum().sum() == 0
            ), f"LỖI: Phát hiện NaN trong {name}"

            assert (
                df.shape[1] == len(train_cols)
            ), f"LỖI: Sai số lượng cột tại {name}"

            assert (
                list(df.columns) == train_cols
            ), f"LỖI: Khác tên/thứ tự cột tại {name}"

        print("✅ Pipeline hoàn tất!")

        return (
            (X_train_A, X_test_A, y_train, y_test),
            (X_train_B, X_test_B, y_train, y_test),
        )


if __name__ == "__main__":

    from path import TRAIN_DATA_FILE, TEST_DATA_FILE

    print("\n=======================================================")
    print("🛠️ KIỂM THỬ PIPELINE")
    print("=======================================================")

    if not TRAIN_DATA_FILE.exists():
        print(f"❌ Không tìm thấy: {TRAIN_DATA_FILE}")

    elif not TEST_DATA_FILE.exists():
        print(f"❌ Không tìm thấy: {TEST_DATA_FILE}")

    else:
        train_df = pd.read_csv(TRAIN_DATA_FILE)
        test_df = pd.read_csv(TEST_DATA_FILE)

        print(
            f"✅ Train: {train_df.shape} | "
            f"Test: {test_df.shape}"
        )

        builder = CreditRiskPipelineBuilder()

        (
            X_tr_A,
            X_te_A,
            y_tr_A,
            y_te_A,
        ), (
            X_tr_B,
            X_te_B,
            y_tr_B,
            y_te_B,
        ) = builder.build_and_run(
            train_df,
            test_df,
        )
        
        #đoạn này thêm để test
        print("\n===== TRAIN A COLUMNS =====")
        for i, col in enumerate(X_tr_A.columns, 1):
            print(f"{i}. {col}")

        print("\n===== TRAIN B COLUMNS =====")
        for i, col in enumerate(X_tr_B.columns, 1):
            print(f"{i}. {col}")

        print("\nSố cột A:", len(X_tr_A.columns))
        print("Số cột B:", len(X_tr_B.columns))
        #kết thúc đoạn thêm để test
        
        
        print("\n===== OBJECT COLUMNS TRAIN A =====")
        print( X_tr_A.select_dtypes( include=["object"]).columns.tolist())

        print("\n===== OBJECT COLUMNS TRAIN B =====")
        print( X_tr_B.select_dtypes( include=["object"]).columns.tolist())

        print("\n===== DTYPES TRAIN A =====")
        print(X_tr_A.dtypes)

        print("\n===== DTYPES TRAIN B =====")
        print(X_tr_B.dtypes)
        

        print("\n===== CHECK NAN =====")
        print(X_tr_A.isna().sum().sum())
        print(X_tr_B.isna().sum().sum())

        print("\n===== CHECK INF =====")
        print(np.isinf(X_tr_A).sum().sum())
        print(np.isinf(X_tr_B).sum().sum())


        print("\n📊 KẾT QUẢ")

        print(
            f"Nhánh A - Train: {X_tr_A.shape} | "
            f"Test: {X_te_A.shape}"
        )

        print(
            f"Nhánh B - Train: {X_tr_B.shape} | "
            f"Test: {X_te_B.shape}"
        )

        print(
            f"Label   - Train: {y_tr_A.shape} | "
            f"Test: {y_te_A.shape}"
        )

        