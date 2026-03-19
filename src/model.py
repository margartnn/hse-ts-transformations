import catboost as cb
import numpy as np
import pandas as pd
from typing import List, Optional

class CatBoostRecursive:

    def __init__(
        self,
        horizon: int,
        history: int,
        use_gpu: bool = False,
        random_seed: int = 42,
    ):
        self.horizon = horizon
        self.history = history
        self.use_gpu = use_gpu
        self.random_seed = random_seed

        self.model = None
        self.feature_columns = None
        self.categorical_features_idx = None

    def fit(
        self,
        train_data: pd.DataFrame,
        id_col: str = "series_name",
        timestamp_col: str = "timestamp",
        value_col: str = "series_value",
        cat_features: Optional[List[str]] = None,
    ):
        train_data = train_data.sort_values([id_col, timestamp_col]).reset_index(drop=True)
        exclude_cols = [id_col, timestamp_col, value_col]
        self.feature_columns = [col for col in train_data.columns if col not in exclude_cols]
        if cat_features:
            self.categorical_features_idx = [
                i for i, col in enumerate(self.feature_columns) if col in cat_features
            ]
        else:
            self.categorical_features_idx = None

        train_clean = train_data.dropna(subset=self.feature_columns + [value_col]).copy()

        X_train = train_clean[self.feature_columns]
        y_train = train_clean[value_col]

        self.model = cb.CatBoostRegressor(
            task_type="GPU" if self.use_gpu else "CPU",
            loss_function="RMSE",
            random_seed=self.random_seed,
            verbose=100,
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            cat_features=self.categorical_features_idx,
        )

        train_pool = cb.Pool(
            data=X_train,
            label=y_train,
            cat_features=self.categorical_features_idx,
        )

        self.model.fit(train_pool, plot=False)

    def predict(
        self,
        test_data: pd.DataFrame,
        id_col: str = "series_name",
        timestamp_col: str = "timestamp",
        value_col: str = "series_value",
    ) -> pd.DataFrame:

        test_df = test_data.copy()
        test_df = test_df.sort_values([id_col, timestamp_col]).reset_index(drop=True)

        predictions = []
        for series_id, group in test_df.groupby(id_col):

            group = group.copy().reset_index()
            original_index = group["index"].values

            for step in range(self.horizon):

                mask = group[value_col].isna()

                if not mask.any():
                    break

                predict_idx = group[mask].index[0]

                X = group.loc[predict_idx, self.feature_columns].values.reshape(1, -1)

                pred = self.model.predict(X)[0]

                group.loc[predict_idx, value_col] = pred

                predictions.append({
                    id_col: series_id,
                    timestamp_col: group.loc[predict_idx, timestamp_col],
                    "predicted_value": pred,
                })

        result_df = pd.DataFrame(predictions)

        if len(result_df) > 0:
            result_df = result_df.sort_values([id_col, timestamp_col]).reset_index(drop=True)

        return result_df