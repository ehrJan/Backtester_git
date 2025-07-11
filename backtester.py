import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

        self.data["Strategy"] = self.data["Signal"].shift(1) * self.data["Returns"]#no look ahead bias
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

        # --- Equity curve (strategy & buy-and-hold) ---
        equity_curves = np.exp(self.results[["Returns", "Strategy"]].cumsum())
        equity_curves.plot(ax=ax, lw=1.5)

        # --- Mark Trades ---
        trades = self.results[self.results["TradeDirection"] != 0]
        for idx, row in trades.iterrows():
            y = np.exp(self.results.loc[:idx, "Strategy"].cumsum())[-1]
            if row["TradeDirection"] == 1:
                ax.plot(idx, y, marker="^", color="green", markersize=8, label="Open Long" if 'Open Long' not in ax.get_legend_handles_labels()[1] else "")
            elif row["TradeDirection"] == -1:
                ax.plot(idx, y, marker="v", color="red", markersize=8, label="Open Short" if 'Open Short' not in ax.get_legend_handles_labels()[1] else "")
            elif row["TradeDirection"] in [-2, 2]:
                ax.plot(idx, y, marker="x", color="black", markersize=8, label="Flip" if 'Flip' not in ax.get_legend_handles_labels()[1] else "")

        # --- Max drawdown area ---
        equity_curve = np.exp(self.results["Strategy"].cumsum())
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        ax.fill_between(drawdown.index, equity_curve, peak, where=drawdown < 0, color='red', alpha=0.2, label='Drawdown')

        ax.set_title(f"Strategy vs Buy & Hold: {self.asset_name} {params}")
        ax.set_yscale("log")
        ax.grid()
        ax.legend()
        plt.show()

    
    #PERFORMANCE METRICS of the strategy

    def get_performance_metrics(self):
        # --- Log Returns and Total Returns ---
        gross_log_return = self.results["Strategy"].sum() + self.results["Fee"].sum()
        net_log_return = self.results["Strategy"].sum()

        gross_return = np.exp(gross_log_return) - 1
        net_return = np.exp(net_log_return) - 1

        sharpe = self.results["Strategy"].mean() / self.results["Strategy"].std() * np.sqrt(252)
        annualized_return = np.exp(net_log_return * (252 / len(self.results))) - 1

        total_fees = self.results["Fee"].sum()
        num_trades = int(self.results["TradeDirection"].abs().sum())
        fee_share = total_fees / gross_log_return if gross_log_return != 0 else np.nan

        # --- Max Drawdown ---
        equity_curve = np.exp(self.results["Strategy"].cumsum())
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()

        # --- Trade-by-trade returns ---
        trade_df = self.extract_trades(plot_pdf=False)
        if trade_df.empty:
            return {
                "total_net_return": net_return,
                "annualized_return": annualized_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "avg_trade_return": np.nan,
                "median_trade_return": np.nan,
                "avg_trade_return_long": np.nan,
                "median_trade_return_long": np.nan,
                "avg_trade_return_short": np.nan,
                "median_trade_return_short": np.nan,
                "skewness": np.nan,
                "num_trades": num_trades,
                "total_fees_paid": total_fees,
                "fees_as_pct_of_gross": fee_share * 100 if pd.notnull(fee_share) else np.nan
            }
        long_trades = trade_df[trade_df["direction"] == "Long"]
        short_trades = trade_df[trade_df["direction"] == "Short"]

        avg_win = trade_df["net_return_pct"].mean() / 100
        median_win = trade_df["net_return_pct"].median() / 100

        avg_win_long = long_trades["net_return_pct"].mean() / 100 if not long_trades.empty else np.nan
        median_win_long = long_trades["net_return_pct"].median() / 100 if not long_trades.empty else np.nan

        avg_win_short = short_trades["net_return_pct"].mean() / 100 if not short_trades.empty else np.nan
        median_win_short = short_trades["net_return_pct"].median() / 100 if not short_trades.empty else np.nan

        num_trades = len(trade_df)
        # --- Skewness ---
        skew = self.results["Strategy"].skew()

        return {
            "total_net_return": round(net_return, 4),
            "annualized_return": round(annualized_return, 4),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_drawdown, 4),
            "avg_trade_return": round(avg_win, 5),
            "median_trade_return": round(median_win, 5),
            "avg_trade_return_long": round(avg_win_long, 5) if pd.notna(avg_win_long) else "N/A",
            "median_trade_return_long": round(median_win_long, 5) if pd.notna(avg_win_long) else "N/A",
            "avg_trade_return_short": round(avg_win_short, 5) if pd.notna(avg_win_short) else "N/A",
            "median_trade_return_short": round(median_win_short, 5)if pd.notna(avg_win_long) else "N/A",
            "skewness": round(skew, 3),
            "num_trades": num_trades,
            "total_fees_paid": round(total_fees, 4),
            "fees_as_pct_of_gross": round(fee_share * 100, 2) if pd.notnull(fee_share) else "N/A"
        }

    def get_drawdown_table(self):
        """
        Identifies all local drawdowns and returns them as a DataFrame with:
        - start date
        - end date
        - drawdown depth
        - drawdown length
        - recovery time
        """
        equity_curve = np.exp(self.results["Strategy"].cumsum())
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak

        in_drawdown = False
        start = None
        trough = None
        max_dd = 0
        drawdowns = []

        for i in range(len(drawdown)):
            if not in_drawdown and drawdown.iloc[i] < 0:
                in_drawdown = True
                start = drawdown.index[i - 1] if i > 0 else drawdown.index[i]
                trough = drawdown.index[i]
                max_dd = drawdown.iloc[i]
            elif in_drawdown:
                if drawdown.iloc[i] < max_dd:
                    max_dd = drawdown.iloc[i]
                    trough = drawdown.index[i]
                if drawdown.iloc[i] == 0:
                    end = drawdown.index[i]
                    duration = (end - start).days
                    recovery_time = (end - trough).days
                    drawdowns.append({
                        "start": start,
                        "trough": trough,
                        "end": end,
                        "length_days": duration,
                        "depth_pct": round(max_dd * 100, 2),
                        "recovery_days": recovery_time
                    })

    def extract_trades(self, plot_pdf=True):
        """
        Extracts all trades and optionally plots a cumulative distribution (CDF) of returns per trade.
        Returns a DataFrame with trade stats.
        """
        trades = []
        in_trade = False
        entry_idx = None
        entry_signal = 0

        for i in range(1, len(self.results)):
            current_signal = self.results["Signal"].iloc[i]
            previous_signal = self.results["Signal"].iloc[i - 1]

            if not in_trade and current_signal != 0 and previous_signal == 0:
                in_trade = True
                entry_idx = i
                entry_signal = current_signal

            elif in_trade and (current_signal == 0 or np.sign(current_signal) != np.sign(entry_signal)):
                exit_idx = i
                trade_data = self.results.iloc[entry_idx:exit_idx]

                trade_return_log = trade_data["Strategy"].sum()
                trade_fees = trade_data["Fee"].sum()
                trade_return = np.exp(trade_return_log) - 1

                trades.append({
                    "start": trade_data.index[0],
                    "end": trade_data.index[-1],
                    "direction": "Long" if entry_signal > 0 else "Short",
                    "log_return": trade_return_log,
                    "net_return_pct": trade_return * 100,
                    "length": len(trade_data),
                    "total_fees": trade_fees
                })

                in_trade = current_signal != 0
                if in_trade:
                    entry_idx = i
                    entry_signal = current_signal

        trade_df = pd.DataFrame(trades)

        # Plot CDF if requested and trades exist
        if plot_pdf and not trade_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.histplot(trade_df["net_return_pct"], bins=30, kde=True, ax=ax, color='skyblue')
            ax.axvline(0, color='red', linestyle="--", label="Break-even")
            ax.axvline(trade_df["net_return_pct"].mean(), color='green', linestyle="--", label=f"Mean: {trade_df['net_return_pct'].mean():.2f}%")
            ax.set_title("Distribution of Trade Returns (PDF)")
            ax.set_xlabel("Trade Return (%)")
            ax.set_ylabel("Frequency")
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.show()


        return trade_df


