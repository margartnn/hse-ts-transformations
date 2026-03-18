import pandas as pd
import numpy as np
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd, RollingMin, RollingMax
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class DummyModel(BaseEstimator, TransformerMixin):
    """
    Заглушка для генерации признаков без обучения
    """
    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        return np.zeros(len(X)) if len(X) > 0 else np.array([])


def generate_features(
        df, 
        id_column='series_name', 
        date_column='timestamp', 
        target_column='series_value', 
        freq='MS', 
        lags=[1, 2, 3, 6, 12], 
        rolling_windows=[3, 6, 12]
        ):
    
    df_ml = df.rename(columns={
        id_column: 'unique_id',
        date_column: 'ds',
        target_column: 'y'
    })
    
    lag_transforms = {}
    for lag in lags:
        transforms = []
        for window in rolling_windows:
            transforms.append(RollingMean(window))
            transforms.append(RollingStd(window))
            transforms.append(RollingMin(window))
            transforms.append(RollingMax(window))
        lag_transforms[lag] = transforms

    mlf = MLForecast(
        models=DummyModel(),
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        num_threads=-1
    )

    mlf.fit(df_ml)
    
    df_features = mlf.preprocess(df_ml)
    
    df_features = df_features.rename(columns={
        'unique_id': id_column,
        'ds': date_column,
        'y': target_column
    })
    
    return df_features