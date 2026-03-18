import pandas as pd
import zipfile 
from sktime.datasets import load_tsf_to_dataframe

def load_data():
    data, metadata = load_tsf_to_dataframe("data/m4_monthly_dataset.tsf")

    return data


def select_n_series(df, n=200):
    df = df.reset_index()
    series = df["series_name"].unique()[:n]
    selected = df[df["series_name"].isin(series)].copy()

    return selected


def split_data(df, horizon=20): 
    t = df["series_name"].unique()
    train_list = []
    test_list = []
    for series in range(len(t)): 
        subset = df[df['series_name'] == t[series]]
        test = subset[-horizon:]
        train = subset[:-horizon]
        train_list.append(train)
        test_list.append(test)
    
    train = pd.concat(train_list)
    test = pd.concat(test_list)

    return train, test 