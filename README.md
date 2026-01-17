# Statistical Pairs Trading Engine

## Performance
![Strategy Result](demo_result.png)
*Backtest showing Spread Z-Score (Top) and Cumulative PnL (Bottom).*

## Overview
This project implements a market-neutral **Mean Reversion Strategy** based on statistical cointegration. It identifies asset pairs that historically move together (e.g., GOOG vs MSFT) and executes trades when the spread deviates significantly from its historical mean.

## Key Features
- **Cointegration Testing:** Uses the **Augmented Dickey-Fuller (ADF)** test to mathematically validate if a pair is mean-reverting (p-value < 0.05).
- **Hedge Ratio Calculation:** Uses **Ordinary Least Squares (OLS)** regression to calculate the dynamic hedge ratio, creating a stationarity-optimized spread.
- **Signal Generation:** Generates Entry/Exit signals based on **Z-Score** thresholds (+/- 2.0 standard deviations).
- **Backtesting Engine:** Simulates trade execution over 5 years of historical data to calculate Sharpe Ratio and Cumulative PnL.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the engine: `python main.py`
