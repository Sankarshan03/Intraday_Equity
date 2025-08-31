# Intraday EMA-RSI Stock Strategy

This repository contains an implementation of an intraday trading strategy based on EMA and RSI indicators for NSE stocks.

## Data Format

The strategy expects CSV files in the `stock_data` folder with the following naming convention:
- `dataNSE_YYYYMMDD.csv`

Each CSV file should have the following columns:
- ticker: Stock symbol
- time: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
- open: Opening price
- high: High price
- low: Low price
- close: Closing price
- volume: Trading volume

## Setup

1. Place your CSV files in the `stock_data` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run the backtest: `python main.py`

## Strategy Details

The strategy uses:
- EMA(3) and EMA(10) on 10-minute data
- EMA(50) on 1-hour data
- RSI(14) on 10-minute data

Entry signals:
- Long: EMA(3) > EMA(10), RSI > 60, Close > EMA(50)
- Short: EMA(3) < EMA(10), RSI < 30, Close < EMA(50)

Exit rules:
- Stop loss: 0.5%
- Profit target: 2%
- Trailing stop: 0.75% after 0.5% profit

## Output

The backtest will generate:
- Trade log with all executed trades
- Performance metrics (return, drawdown, win rate, Sharpe ratio)
- Equity curve data