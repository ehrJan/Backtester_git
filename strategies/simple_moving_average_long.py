from strategies.base import Strategy
import pandas as pd
import numpy as np  

class TrendFollowingStrategy(Strategy):
    def __init__(self,asset_name, window=20,long_only=True):
        self.asset_name = asset_name
        self.window = window
        self.long_only = long_only

    def generate_signals(self, data):
        ma = data[self.asset_name].rolling(window=self.window).mean()
        if self.long_only:
            return (data[self.asset_name] > ma).astype(int)
        else:
            # For long and short strategy, return 1 for long, -1 for short, 0 for hold
            signals = pd.Series(0, index=data.index)
            signals[data[self.asset_name] > ma] = 1
            signals[data[self.asset_name] < ma] = -1
            return signals
        

class TrendFollowingStrategyVolPossitioning(TrendFollowingStrategy):
    def __init__(self, asset_name,long_only=True, window=20,target_annual_volatility=0.5):
        super().__init__(asset_name, window, long_only)
        self.target_annual_volatility = target_annual_volatility

    def generate_signals(self, data):
        ma = data[self.asset_name].rolling(window=self.window).mean()

        log_returns = np.log(data[self.asset_name] / data[self.asset_name].shift(1))
        volatility = log_returns.rolling(window=self.window).std() * np.sqrt(360)
        signals = pd.Series(0.0, index=data.index)

        vol_position = self.target_annual_volatility / volatility
        vol_position = vol_position.fillna(0)

        if self.long_only:
            signals.loc[data[self.asset_name] > ma] = vol_position
        else:
            signals.loc[data[self.asset_name] > ma] = vol_position
            signals.loc[data[self.asset_name] < ma] = -vol_position

        return signals
