# A Trading bot to be deployed on quantconnect
"""
Created on Fri Aug 10 12:11:18 2021

@author: Niraj

"""

# This script describes the XGBOOST model, and its methods for training and prediction

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

class XGBoost:
    
    def __init__(self):
        
        # Initializing the model
        self.model = None
        

    def ProcessData_XGBoost(self, data):
        
        # Creating sequential dataset, with upto 3 time-stamps back, for training model
        temp_1 = pd.DataFrame(data.shift(1))
        temp_1.columns = ['t-1']
        temp_2 = pd.DataFrame(data.shift(2))
        temp_2.columns = ['t-2']
        temp_3 = pd.DataFrame(data.shift(3))
        temp_3.columns = ['t-3']
        
        total_price_df = pd.DataFrame(data)
        total_price_df.columns = ['close']
        
        train_data_df = total_price_df.join([temp_1, temp_2, temp_3])
        train_data_df.dropna(inplace=True)
        
        train_data_X = train_data_df.iloc[:, 1:].values
        train_data_Y = train_data_df.iloc[:, 0].values
        
        return train_data_X, train_data_Y
    
    
    def Create_XGBoost(self):
    
        # Create Model
        self.model = XGBRegressor(objective='reg:squarederror', n_estimators = 1000)
        
    def Fit_XGBoost(self, model, train_data_X, train_data_Y):
        
        # Training the model 
        model.fit(train_data_X, train_data_Y)
    
    def Fit_XGBoost(self, train_data_X, train_data_Y):
        
        # Training the model 
        self.model.fit(train_data_X, train_data_Y)
    
    
        
    def Forecast_XGBoost(self, train_data_X, train_data_Y):
        
        # Forecasting for one-step ahead
        test_input = np.hstack((train_data_Y[-1], train_data_X[-1,:2]))
        test_input = np.expand_dims(test_input, axis=0)
        
        prediction_XGBoost = float(self.model.predict(test_input))
        return prediction_XGBoost