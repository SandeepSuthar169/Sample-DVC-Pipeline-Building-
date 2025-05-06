import numpy as np
import pandas as pd
import os
import pickle

from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv('./data/processed/train_processed.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# X_test = test_data.iloc[:,0:-1].values
# y_test = test_data.iloc[:,-1].values

rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)

pickle.dump(rfc, open('model.pkl', 'wb'))
