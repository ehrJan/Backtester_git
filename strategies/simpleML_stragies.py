from strategies.base import Strategy
import pandas as pd
import numpy as np  

class SimpleCatBoost(Strategy):
    def __init__(self,long_only=True):
        self.long_only = long_only

    def generate_signals(self, data):
        signals=(data['prediction_2']+data['prediction_1'])/2
        return signals
        