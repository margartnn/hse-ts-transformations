from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive

def evaluate_baseline(df_train, df_test): 
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()

    df_train_copy = df_train_copy.rename(columns={
        "series_name": "unique_id",
        "timestamp": "ds",
        "series_value": "y"
    })

    df_test_copy = df_test_copy.rename(columns={
        "series_name": "unique_id",
        "timestamp": "ds",
        "series_value": "y"
    })

    sf = StatsForecast(
        models = [
            Naive(),
            SeasonalNaive(season_length=12)
        ],
        freq='MS',
        verbose=True
    )
    sf.fit(df_train_copy)
    preds = sf.predict(h=20)

    results = preds.merge(df_test_copy, on=["unique_id", "ds"])

    return results