from backtester import Backtester
from strategies.simple_moving_average_long import TrendFollowingStrategy
from strategies.mean_reversion import MeanReversionStrategy

if __name__ == "__main__":
    strategy = TrendFollowingStrategy(window=20)
    # strategy = MeanReversionStrategy(window=20)

    bt = Backtester(
        data_path="data/cleaned_crypto_7_cleaned.csv",
        strategy=strategy,
        initial_cash=10000
    )

    bt.run()
    bt.evaluate()
    bt.plot()
