# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 15:26:41 2024

@author: clinton
"""
from pydantic import BaseModel
from typing import Optional

# 2. Class which describes Credit Score measurements
class CreditScoring(BaseModel):
    TransactionStartTime_Day: int
    TransactionCount: int
    StdTransactionAmount: float
    TransactionStartTime_Month: int  
    AverageTransactionAmount: float
    Value: int  
    TransactionStartTime_Hour:  int  
    FraudResult: int  
    Amount: float
    PricingStrategy: int  
    MonetaryTotal_woe: float
    MonetaryAvg_woe: float
    Frequency_woe: float
    Recency_woe: float