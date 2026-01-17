import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

ticker_a = "GOOG"
ticker_b = "MSFT"
start_date = "2020-01-01"
end_date = "2024-01-01"

print(f"--- Fetching Data for {ticker_a} and {ticker_b} ---")

df_a = yf.download(ticker_a, start=start_date, end=end_date, progress=False)
df_b = yf.download(ticker_b, start=start_date, end=end_date, progress=False)

if df_a.empty or df_b.empty:
    print("ERROR: No data downloaded. Check your internet connection or ticker names.")
    exit()

try:
    series_a = df_a['Adj Close'] if 'Adj Close' in df_a.columns else df_a['Close']
    series_b = df_b['Adj Close'] if 'Adj Close' in df_b.columns else df_b['Close']
except KeyError:
    series_a = df_a.iloc[:, 0] 
    series_b = df_b.iloc[:, 0]

if isinstance(series_a, pd.DataFrame): series_a = series_a.iloc[:, 0]
if isinstance(series_b, pd.DataFrame): series_b = series_b.iloc[:, 0]

data = pd.concat([series_a, series_b], axis=1)
data.columns = [ticker_a, ticker_b]
data = data.dropna()

print(f"Successfully aligned {len(data)} rows of data.")

stock_a = data[ticker_a]
stock_b = data[ticker_b]

# STATISTICAL ANALYSIS
# Calculate the Hedge Ratio using Linear Regression (OLS)
# How many shares of B do we need to hedge 1 share of A?
stock_b_const = sm.add_constant(stock_b)
result = sm.OLS(stock_a, stock_b_const).fit()
hedge_ratio = result.params[ticker_b]

print(f"Hedge Ratio: {hedge_ratio:.4f}")

# Calculate the "Spread" (The Leash)
# Spread = Stock_A - (Hedge_Ratio * Stock_B)
spread = stock_a - (hedge_ratio * stock_b)

# ADF Test
# If p-value < 0.05, the spread is "Mean Reverting" (Good for trading)
# If p-value > 0.05, it's a random walk (Bad for trading)
adf_result = adfuller(spread)
p_value = adf_result[1]

print(f"ADF P-Value: {p_value:.5f}")
if p_value < 0.05:
    print(">> SUCCESS: The pair is Cointegrated. Mean Reversion strategy applies.")
else:
    print(">> WARNING: The pair is NOT Cointegrated. Risks are high.")

window = 30 
spread_mean = spread.rolling(window=window).mean()
spread_std = spread.rolling(window=window).std()
z_score = (spread - spread_mean) / spread_std

signals = pd.DataFrame(index=spread.index)
signals['price_a'] = stock_a
signals['price_b'] = stock_b
signals['z'] = z_score
signals['position'] = 0

signals.loc[signals.z > 2.0, 'position'] = -1
signals.loc[signals.z < -2.0, 'position'] = 1
signals.loc[abs(signals.z) < 0.5, 'position'] = 0 
signals['position'] = signals['position'].ffill().fillna(0) 

spread_daily_return = spread.diff()
signals['pnl'] = signals['position'].shift(1) * spread_daily_return
signals['cumulative_pnl'] = signals['pnl'].cumsum()

total_profit = signals['cumulative_pnl'].iloc[-1]
sharpe_ratio = np.sqrt(252) * (signals['pnl'].mean() / signals['pnl'].std())

print(f"--- BACKTEST RESULTS ---")
print(f"Total Profit (per share base): ${total_profit:.2f}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(z_score, label="Spread Z-Score", alpha=0.7)
plt.axhline(2.0, color='r', linestyle='--', label="Sell Threshold (+2)")
plt.axhline(-2.0, color='g', linestyle='--', label="Buy Threshold (-2)")
plt.axhline(0, color='black', linewidth=1)
plt.legend()
plt.title(f"Z-Score Mean Reversion Signal ({ticker_a} vs {ticker_b})")

plt.subplot(2, 1, 2)
plt.plot(signals['cumulative_pnl'], color='green', label="Strategy Profit")
plt.legend()
plt.title("Cumulative PnL (USD)")

plt.tight_layout()
plt.show()