import numpy as np
import pandas as pd
import pickle
import json 
from sklearn.metrics import accuracy_score
from sklearn.metrics import  precision_score, f1_score, recall_score

rfc = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('./data/processed/test_processed.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values


y_pred = rfc.predict(X_test)
y_test = y_test.astype(int)
y_pred = y_pred.astype(int)

accuracy = accuracy_score(y_test, y_pred)
#precision  = precision_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#f1 = f1_score(y_test, y_pred)

metrics_dict = {
    'accuracy':accuracy,
    #'precision':precision,
    #'recall':recall,
    #'f1':f1
}

with open('metrics.json', 'w') as file:  
    json.dump(metrics_dict, file, indent=4)
