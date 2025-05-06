import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv('./data/raw/train.csv')
test_data = pd.read_csv('./data/raw/test.csv')

def standard_df(df):
    df = df.select_dtypes(include=['int64', 'float64'])
    scaler = StandardScaler()
    scaler_array = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaler_array, columns = df.columns, index = df.index)
    return scaled_df

train_processed_data = standard_df(train_data)   
test_processed_data = standard_df(test_data)

data_path  = os.path.join("data", "processed")

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"))
test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"))
