from strategies.base import Strategy
import pandas as pd

class MeanReversionStrategy(Strategy):
    def __init__(self,asset_name, window=20, threshold=0.02):
        self.window = window
        self.threshold = threshold
        self.asset_name = asset_name

    def generate_signals(self, data):
        ma = data[self.asset_name].rolling(window=self.window).mean()
        deviation = (data[self.asset_name] - ma) / ma
        signals = (deviation < -self.threshold).astype(int)
        return signals
