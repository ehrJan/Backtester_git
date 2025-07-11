from strategies.base import Strategy
import pandas as pd

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
    def __init__(self, asset_name, window=20,target_volatility=0.5):
        super().__init__(asset_name, window, long_only=True)
        self.target_volatility = target_volatility

    def generate_signals(self, data):
        ma = data[self.asset_name].rolling(window=self.window).mean()
        volatility = data[self.asset_name].rolling(window=self.window).std()
        signals = pd.Series(0, index=data.index)
        if self.long_only:
            signals[(data[self.asset_name] > ma)] = volatility/self.target_volatility
        
        else:
            signals[(data[self.asset_name] > ma)] = volatility/self.target_volatility
            signals[(data[self.asset_name] < ma) & (volatility < volatility.quantile(0.25))] = -1*(volatility/self.target_volatility)    
    
        return signals