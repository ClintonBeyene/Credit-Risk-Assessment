# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:26:41 2024

@author: clinton
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from CreditScoring import CreditScoring
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Credit Scoring Model': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Credit Score with the confidence
@app.post('/predict')
def predict_credit_scoring(data:CreditScoring):
    data = data.dict()
    TransactionStartTime_Day=data['TransactionStartTime_Day']
    TransactionCount=data['TransactionCount']
    StdTransactionAmount=data['StdTransactionAmount']
    TransactionStartTime_Month=data['TransactionStartTime_Month']
    AverageTransactionAmount=data['AverageTransactionAmount']
    Value=data['Value']
    TransactionStartTime_Hour=data['TransactionStartTime_Hour']
    FraudResult=data['FraudResult']
    Amount=data['Amount']
    PricingStrategy=data['PricingStrategy']
    MonetaryTotal_woe=data['MonetaryTotal_woe']
    MonetaryAvg_woe=data['MonetaryAvg_woe']
    Frequency_woe=data['Frequency_woe']
    Recency_woe=data['Recency_woe']
    
    prediction = classifier.predict(np.array([[TransactionStartTime_Day, TransactionCount, StdTransactionAmount, TransactionStartTime_Month, 
                                             AverageTransactionAmount, Value, TransactionStartTime_Hour, FraudResult, Amount, 
                                             PricingStrategy, MonetaryTotal_woe, MonetaryAvg_woe, Frequency_woe, Recency_woe]]))
    if(prediction[0]==0):
        prediction="Bad"
    else:
        prediction="Good"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload # First parameter is app file second parameter is object