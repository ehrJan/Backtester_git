from strategies.base import Strategy
import pandas as pd
import numpy as np  

class SimpleBreakOut(Strategy):
    def __init__(self, asset_name, window=20, long_only=True):
        self.asset_name = asset_name
        self.window = window
        self.long_only = long_only

    def generate_signals(self, data):
        max = data[self.asset_name].shift(1).rolling(window=self.window).max()
        min = data[self.asset_name].shift(1).rolling(window=self.window).min()
        if self.long_only:
            return (data[self.asset_name] > max).astype(int)
        else:
            signals = pd.Series(0, index=data.index)
            signals[data[self.asset_name] > max] = 1
            signals[data[self.asset_name] < min] = -1
            return signals
        

class SimpleBreakOutMAOut(Strategy):
    def __init__(self, asset_name, window=20, long_only=True, ma_window=50):
        
        self.asset_name = asset_name
        self.window = window
        self.long_only = long_only
        self.ma_window = ma_window 
       

    def generate_signals(self, data):
        max_price = data[self.asset_name].shift(1).rolling(window=self.window).max()
        min_price = data[self.asset_name].shift(1).rolling(window=self.window).min()
        ma = data[self.asset_name].shift(1).rolling(window=self.ma_window).mean()

        if self.long_only:
            signals = pd.Series(0, index=data.index)
            signals[data[self.asset_name] > max_price] = 1
            # Adjust signals based on MA crossover
            for i in range(1, len(data)):
                if signals.iloc[i-1] == 1 and data[self.asset_name].iloc[i] < ma.iloc[i]:
                    signals.iloc[i] = 0
            
            return signals
        else:
            signals = pd.Series(0, index=data.index)
            signals[data[self.asset_name] > max_price] = 1
            signals[data[self.asset_name] < min_price] = -1
            # Adjust signals based on MA crossover
            for i in range(1, len(data)):
                if signals.iloc[i-1] == 1 and data[self.asset_name].iloc[i] < ma.iloc[i]:
                    signals.iloc[i] = 0
            return signals
        

