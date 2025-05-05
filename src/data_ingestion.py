import numpy as np
import pandas as pd
import os
from  sklearn.model_selection import train_test_split
import yaml
import logging


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('error.log')
file_handler.setLevel("ERROR")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.error.ParserError as e:    
        print(f"error:  failed to parse the csv file from {data_url}.")
        print(e)
        raise
    except Exception as e:
        print(f"error: an unexpeted error occurred whild loading the data.")
        print(e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['gender'] = df['gender'].replace({
            'Female': 0, 
            'Male': 1,
            'Other': 2})
        df['gender'] = df['gender'].astype(int)
        df['smoking_history'] = df['smoking_history'].replace({
            'No Info': 1,
            'never': 2,'former': 3,
            'not current': 3,
            'current': 4,
            'ever': 4})
        df['smoking_history'] = df['smoking_history'].astype(int)
        return df
    except KeyError as e:
        print(f"Error: Missing columns {e} in the dataframe.")
        raise
    except Exception  as e:
        print(f"error: an unexpected error occurred during preprocessing. ") 
        print(e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        data_path = os.path.join(data_path, 'raw')
        os.makedirs(data_path, exist_ok = True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index = False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index = False)
    except Exception as e:
        print(f"error: an unexpected error occured while saving the data.")
        print(e)
        raise

def main():
    try:
        df = load_data(data_url="C:/Users/Sande/Desktop/Datasets/diabetes_prediction_dataset.csv")   
        df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size = 0.2, random_state = 42)
        save_data(train_data, test_data, data_path="data")
    except Exception as e:
        print(f"error {e}")
        print('failed to complete the data ingestion precess') 

if __name__ == '__main__':
    main()           

