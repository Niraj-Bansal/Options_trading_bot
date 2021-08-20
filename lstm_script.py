# A Trading bot to be deployed on quantconnect

"""
@author: Niraj

"""

# This script describes the LSTM model, and its methods for training and forecasting

import numpy as np
import pandas as pd

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

class MyLSTM:
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range = (0, 1))

    def ProcessData_LSTM(self, data):
        
        # Transform data
        training_data_array = np.array(data).reshape((len(data), 1))
        training_data_scaled = self.scaler.fit_transform(training_data_array)
        
        # Get features and labels
        features_set = []
        labels = []
        
        for i in range(20, training_data_array.shape[0]):
            features_set.append(training_data_scaled[i-20:i, 0])
            labels.append(training_data_scaled[i, 0])
            
        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
        return features_set, labels, training_data_scaled
    
    
    def Create_LSTM(self, features_set, labels):
    
        # Create Model
        self.model = Sequential()
        self.model.add(LSTM(units = 50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units = 1))
        self.model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        
    def Fit_LSTM(self, model, features_set, labels):
        
        # Training the LSTM model
        model.fit(features_set, labels, epochs = 50, batch_size = 32)
    
        
    def Forecast_LSTM(self, model, training_data_scaled):
        
        # Forecasting for 1-step using LSTM model
        test_input = training_data_scaled[-20:]
        test_input = test_input.reshape(-1,1)
        
        prediction = model.predict(np.expand_dims(test_input, axis=0))
        prediction_LSTM = float(self.scaler.inverse_transform(prediction))
        
        return prediction_LSTM