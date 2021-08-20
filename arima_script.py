# A Trading bot to be deployed on quantconnect

"""
@author: Niraj

"""

# This script describes the ARIMA model, and its methods for training predicting

import numpy as np
import pandas as pd

from statsmodels.tsa.arima_model import ARIMA

class MyARIMA:
    
    def __init__(self):
        self.model = None

    def Create_ARIMA(self, data):
    
        # Create ARIMA Model
        self.model = ARIMA(data, order=(3,1,5))
        
    def Fit_ARIMA(self, model):
        
        # Training the ARIMA model
        result = model.fit(disp=0)
        return result
        
    def Forecast_ARIMA(self, result):
        
        # Forecasting from the ARIMA model
        prediction_arima, _ , _ = result.forecast(1)
        return prediction_arima