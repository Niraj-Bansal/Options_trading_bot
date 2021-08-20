# A Trading bot to be deployed on quantconnect

"""
@author: Niraj

"""
# This is the main script for the trading

from lstm_script import MyLSTM
from xgboost_script import XGBoost
from arima_script import MyARIMA
from datetime import timedelta, datetime

class MultidimensionalHorizontalFlange(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2017, 1, 4)  # Set Start Date
        self.SetCash(100000)  # Set Strategy Cash
        self.SetEndDate(2021, 8, 1)
        
        """ Check for them """
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        
        self.SetExecution(ImmediateExecutionModel())

        # self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())

        self.UniverseSettings.Resolution = Resolution.Daily
        self.SetUniverseSelection(LiquidETFUniverse())
        
        # Helper dictionaries
        self.models = {"Arima": None, "Arima_results": None, "XGBoost": None, "LSTM": None}
        
        # Initializing the expiry date
        self.lastest_expiry = datetime.min
        
        # Initializing the Securities
        equity = self.AddEquity('SPY')
        option = self.AddOption("SPY", Resolution.Minute)
        self.symbol_option = option.Symbol
        self.symbol_equity = equity.Symbol
        
        
        # Adding a consolidator as the options data is available in minute scale, and we need daily scale
        self.Consolidate(self.symbol_option, Resolution.Daily, self.OnOptionConsolidated)
        
        # Schedule prediction, and trading 
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.At(6, 0), Action(self.Rebalance))
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.At(7, 0), Action(self.PredictModels))
        self.Schedule.On(self.DateRules.EveryDay('SPY'), self.TimeRules.AfterMarketOpen('SPY', 5), self.TradeOptions)
    
    def OnData(self,slice):
        
        # We wont use OnData, and rather used our rebalanced timeline
        pass
            
    def OnOptionConsolidated(self, consolidated):
        
        if self.lastest_expiry.date() == self.Time.date():
            self.symbol_consolidated = consolidated.Symbol
            self.optionchain = self.CurrentSlice.OptionChains
        
    def Rebalance(self):
        
        # Getting the Date for the nearest expiry
        calendar = self.TradingCalendar.GetDaysByType(TradingDayType.OptionExpiration, self.Time, self.EndDate)
        expiries = [i.Date for i in calendar]
        if len(expiries) == 0: return
        self.lastest_expiry = expiries[0]
        
            
        
    def PredictModels(self):
        
        if self.lastest_expiry.date() == self.Time.date():
        
            # Fetching the dataset, and performing training and prediction on it
            qb = self
            history = qb.History([self.symbol_equity], 1280, Resolution.Daily)
            data = history.loc[self.symbol_equity].close

            # (1) ... Creating the LSTM object and model
            lstm_obj = MyLSTM()
            self.features_set, self.labels, self.training_data_scaled = lstm_obj.ProcessData_LSTM(data)
            lstm_model = lstm_obj.Create_LSTM(self.features_set, self.labels)
            self.models["LSTM"] = lstm_model
            
            # (2) ... Creating the Arima object and model
            arima_obj = MyARIMA()
            arima_model = arima_obj.Create_ARIMA(data)
            self.models["Arima"] = arima_model
            
            # (3) ... Creating the XGBoost object and model
            xgboost_obj = XGBoost()
            self.train_data_X, self.train_data_Y = xgboost_obj.ProcessData_XGBoost(data)
            xgboost_model = xgboost_obj.Create_XGBoost()
            self.models['XGBoost'] = xgboost_model
            
            # Using self.Train as it avoids runtime error
            self.Train(self.TrainModels)
            
            # Gathering the predictions from models individually 
            prediction_lstm =  lstm_obj.Forecast_LSTM(self.models["LSTM"], self.training_data_scaled)
            prediction_arima = arima_obj.Forecast_ARIMA(self.models["Arima_results"])
            prediction_xgboost = xgboost_obj.Forecast_XGBoost(self.models['XGBoost'], self.train_data_X, self.train_data_Y)
            
            # Storing the ensemble prediction
            self.prediction = 0.7*prediction_arima + 0.1*prediction_xgboost + 0.2*prediction_lstm
            self.prediction = prediction_xgboost
            self.change  = ((self.prediction - float(data[-1]))/self.prediction)*100

    
    def TrainModels(self):
        
        # Function to train models, and avoiding runtime error
        self.models['Arima'] = arima_obj.Fit_ARIMA(self.models['Arima'])
        self.models['LSTM'] = lstm_obj.Fit_LSTM(self.models['LSTM'], self.features_set, self.labels)
        self.models['XGBoost'] = xgboost_obj.Fit_XGBoost(self.models['XGBoost'], self.train_data_X, self.train_data_Y)
        
        # Function to train models, and avoiding runtime error
        self.model['Arima_results'] = self.model['Arima'].fit(disp=0)
        self.model['LSTM'].fit(self.features_set, self.labels, epochs = 50, batch_size = 32)
        self.model['XGBoost'].fit(self.train_data_X, self.train_data_Y)
      
    
    def TradeOptions(self):
    
        """ TradeOptions function when triggered, identifies the best spread based option trading stratergy
            using the underlying price, and market forecast and Executes it """
        
        
        if self.lastest_expiry.date() == self.Time.date():
        
            """ Bull Spread """
            
            if self.change > float(5):
                
                for i in self.optionchain:
                    
                    # Using only the optionchain with symbol as SPY options Symbol 
                    if i.Key != self.symbol_consolidated: continue
                    chain = i.Value
                    
                    # Sorting the optionchain according to expiry, and choosing the nearest one (on latest_expiry)
                    expiry = sorted(chain,key = lambda x: x.Expiry, reverse=False)[0].Expiry
                    
                    # Filtering the call options from the contracts that expires on that date
                    call = [i for i in chain if i.Expiry == expiry and i.Right == 0]
                    
                    # Sorting the call options, with strike greater than prediction
                    call_contracts_buy = [i for i in call if i.Strike < self.prediction]
                    call_contratcs_sell = [i for i in call if i.Strike > self.prediction]
                    
                    if len(call_contracts_buy) == 0: continue
                    if len(call_contracts_sell) == 0: continue
                    
                    # getting the call option with the just low and higher strike price acc to prediction
                    self.call_buy = sorted(call_contracts_buy, key = lambda x: x.Strike, reverse=True)[0]
                    self.call_sell = sorted(call_contratcs_sell, key = lambda x: x.Strike, reverse=False)[0]
                    
                    # Placing the orders according to bull spread
                    self.Buy(self.call_buy.Symbol, 10)
                    self.Sell(self.call_sell.Symbol ,10)
                    
            
            elif self.change < float(5):
                
                """ Bear Spread """
                
                for i in self.optionchain:
                    
                    # Using only the optionchain with symbol as SPY options Symbol 
                    if i.Key != self.symbol_consolidated: continue
                    chain = i.Value
                    
                    # Sorting the optionchain according to expiry, and choosing the nearest one (on latest_expiry)
                    expiry = sorted(chain,key = lambda x: x.Expiry, reverse=False)[0].Expiry
                    
                    # Filtering the put options from the contracts that expires on that date
                    put = [i for i in chain if i.Expiry == expiry and i.Right == 1]
                    
                    # Sorting the put options, with strike greater than prediction
                    put_contracts_sell = [i for i in put if i.Strike < self.prediction]
                    put_contratcs_buy = [i for i in put if i.Strike > self.prediction]
                    
                    if len(put_contratcs_buy) == 0: continue
                    if len(put_contracts_sell) == 0: continue
                    
                    # getting the put option with the just low and higher strike price a/c to prediction
                    self.put_buy = sorted(put_contratcs_buy, key = lambda x: x.Strike, reverse=False)[0]
                    self.put_sell = sorted(put_contracts_sell, key = lambda x: x.Strike, reverse=True)[0]
                    
                    # Placing the orders according to bear spread
                    self.Buy(self.put_buy.Symbol, 10)
                    self.Sell(self.put_sell.Symbol ,10)
                    
            
            else:
                
                """ Butterfly Spread """
                
                for i in self.optionchain:
                    
                    # Using only the optionchain with symbol as SPY options Symbol
                    if i.Key != self.symbol_consolidated: continue
                    chain = i.Value
                    
                    # Sorting the optionchain according to expiry, and choosing the nearest ones (on latest_expiry)
                    expiry = sorted(chain,key = lambda x: x.Expiry, reverse=True)[0].Expiry
                    
                    # Filtering the call options from the contracts that expires on that date
                    call = [i for i in chain if i.Expiry == expiry and i.Right == 0]
                    
                    # Sorting the call options using strike prices
                    call_contracts = sorted(call,key = lambda x: x.Strike, reverse=False) 
                    
                    if len(call_contracts) == 0: continue
                    
                    # choose OTM call 
                    self.otm_call = [i for i in call_contracts if i.Strike > self.prediction][0]
                    # choose ITM call 
                    self.itm_call = [i for i in call_contracts if i.Strike < self.prediction][-1]
                    # choose ATM call
                    self.atm_call = sorted(call_contracts,key = lambda x: abs(chain.Underlying.Price - x.Strike))[0]
        
                    # Placing orders according to butterfly spread
                    self.Sell(self.atm_call.Symbol ,20)
                    self.Buy(self.itm_call.Symbol ,10)
                    self.Buy(self.otm_call.Symbol ,10)
                                    
                                        
                    