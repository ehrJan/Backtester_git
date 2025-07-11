from strategies.base import Strategy
import pandas as pd

class TrendFollowingStrategy(Strategy):
    def __init__(self,asset_name, window=20):
        self.asset_name = asset_name
        self.window = window

    def generate_signals(self, data):
        ma = data[self.asset_name].rolling(window=self.window).mean()
        return (data[self.asset_name] > ma).astype(int)  # 1 if price > MA, else 0
