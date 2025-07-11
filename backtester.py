import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Backtester:
    def __init__(self, data:pd.DataFrame, strategy, initial_cash=10000,trading_fee_percent=0.0004):
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.results = None
        self.asset_name = data.columns[0] if isinstance(data, pd.DataFrame) and not data.empty else None
        self.trading_fee_percent = trading_fee_percent
        
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

        self.data["TradeChange"] = self.data["Signal"].diff().fillna(0)
        
        def trade_direction(change):
            if change == 1: return 1  # open long
            elif change == -1: return -1  # close long or open short
            elif change == 2: return 2  # flip short to long
            elif change == -2: return -2  # flip long to short
            else: return 0

        self.data["TradeDirection"] = self.data["TradeChange"].apply(trade_direction)

        # Apply fees
        self.data["Fee"] = self.data["TradeChange"].abs() * self.trading_fee_percent
        self.data["Strategy"] -= self.data["Fee"]

        self.data.dropna(inplace=True)
        self.results = self.data.copy()
    
    def evaluate(self, silent=False):
        metrics = self.get_performance_metrics()
        if not silent:
            for k, v in metrics.items():
                print(f"{k}: {v}")

    def plot(self, params=None):
        fig, ax = plt.subplots(figsize=(14, 6))

        # Equity curves
        (1 + self.results[["Returns", "Strategy"]]).cumprod().plot(ax=ax, lw=1.5)

        # Plot entry/exit points
        trades = self.results[self.results["TradeDirection"] != 0]

        for idx, row in trades.iterrows():
            if row["TradeDirection"] == 1:  # open long
                ax.plot(idx, (1 + self.results.loc[idx, "Strategy"]).cumprod(), marker="^", color="green", markersize=8, label="Open Long" if 'Open Long' not in ax.get_legend_handles_labels()[1] else "")
            elif row["TradeDirection"] == -1:  # close long or open short
                ax.plot(idx, (1 + self.results.loc[idx, "Strategy"]).cumprod(), marker="v", color="red", markersize=8, label="Open Short" if 'Open Short' not in ax.get_legend_handles_labels()[1] else "")
            elif row["TradeDirection"] in [-2, 2]:
                ax.plot(idx, (1 + self.results.loc[idx, "Strategy"]).cumprod(), marker="x", color="black", markersize=8, label="Position Flip" if 'Position Flip' not in ax.get_legend_handles_labels()[1] else "")
        
        ax.set_title(f"Strategy vs Buy & Hold: {self.asset_name} {params}")
        ax.grid()
        ax.legend()
        plt.show()

    def get_performance_metrics(self):
        gross_return = (1 + self.results["Strategy"] + self.results["Fee"]).prod() - 1
        net_return = (1 + self.results["Strategy"]).prod() - 1

        sharpe = self.results["Strategy"].mean() / self.results["Strategy"].std() * np.sqrt(252)
        annualized_return = (1 + net_return) ** (252 / len(self.results)) - 1

        total_fees = self.results["Fee"].sum()
        num_trades = int(self.results["TradeDirection"].abs().sum())
        fee_share = total_fees / gross_return if gross_return != 0 else np.nan

        return {
            "total_return": round(net_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe": round(sharpe, 2),
            "num_trades": num_trades,
            "total_fees_paid": round(total_fees, 4),
            "fees_as_pct_of_gross": round(fee_share * 100, 2) if pd.notnull(fee_share) else "N/A"
        }
