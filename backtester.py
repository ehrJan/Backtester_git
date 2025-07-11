import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Backtester:
    def __init__(self, data:pd.DataFrame, strategy, initial_cash=10000):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.results = None
        self.asset_name = data.columns[0] if isinstance(data, pd.DataFrame) and not data.empty else None
        self.clean_data()

    def clean_data(self):
        if self.asset_name:
            self.data = self.data[[self.asset_name]].dropna()
        else:
            self.data = self.data.dropna()

        self.data.sort_index(inplace=True)

    def run(self):
        self.data["Signal"] = self.strategy.generate_signals(self.data)
        self.data["Returns"] = np.log(self.data[self.asset_name] / self.data[self.asset_name].shift(1))
        self.data["Strategy"] = self.data["Signal"].shift(1) * self.data["Returns"]
        self.data.dropna(inplace=True)
        self.results = self.data.copy()

    def evaluate(self, silent=False):
        metrics = self.get_performance_metrics()
        if not silent:
            for k, v in metrics.items():
                print(f"{k}: {v}")

    def plot(self,params=None):
        (1 + self.results[["Returns", "Strategy"]]).cumprod().plot(figsize=(12, 6))

        if self.results is None:
            plt.title(f"Buy & Hold vs Strategy Performance for {self.asset_name}")
        else:
            plt.title(f"Buy & Hold vs Strategy Performance for {self.asset_name} and {params}")
        plt.grid()
        plt.show()

    def get_performance_metrics(self):
        total_return = (1 + self.results["Strategy"]).prod() - 1
        sharpe = self.results["Strategy"].mean() / self.results["Strategy"].std() * (252**0.5)
        annualized_return = (1 + total_return) ** (252 / len(self.results)) - 1

        return {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe": round(sharpe, 2)
        }