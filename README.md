# Intraday EMA-RSI Trading Strategy (Streamlit Only)

## Updated README.md

```markdown
# Intraday EMA-RSI Trading Strategy

A comprehensive intraday trading strategy implementation using Exponential Moving Averages (EMA) and Relative Strength Index (RSI) for the Indian stock market (NSE).

## Features

- **Automated Stock Selection**: Selects top 10 stocks by turnover at 9:25 AM
- **Dual Strategy**: Supports both long and short positions
- **Technical Indicators**: EMA(3), EMA(10), EMA(50), and RSI(14)
- **Risk Management**: Stop loss, profit targets, and trailing stops
- **Backtesting Engine**: Comprehensive performance metrics
- **Streamlit Dashboard**: Interactive web interface for analysis

## Strategy Rules

### Stock Selection
- At 9:25 AM, select top 10 stocks by turnover
- Turnover = Σ(Volume × Close price) for first 10 minutes (9:15-9:25)

### Long Entry
- EMA(3) > EMA(10)
- RSI(14) > 60
- Close > EMA(50)
- Buy at high of entry candle

### Short Entry
- EMA(3) < EMA(10)
- RSI(14) < 30
- Close < EMA(50)
- Sell at low of last 5 minutes

### Exit Rules
- Stop loss: 0.5%
- Profit target: 2%
- Trailing stop: 0.75% after 0.5% profit

## Project Structure

```
Intraday_Equity/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── indicators.py      # Technical indicator calculations
│   ├── backtesting.py     # Backtesting engine
│   ├── performance.py     # Performance metrics calculation
│   └── utils.py           # Utility functions
├── app.py                 # Streamlit dashboard
├── stock_data/            # Directory for CSV files
├── results/               # Backtest results output
├── config/                # Configuration files
│   └── config.yaml        # Strategy configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8+

### Installation Steps
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your CSV files in the `stock_data/` directory with the naming convention `dataNSE_YYYYMMDD.csv`

### CSV Format
Expected CSV columns:
- ticker: Stock symbol
- time: Timestamp (YYYY-MM-DD HH:MM:SS)
- open: Opening price
- high: High price
- low: Low price
- close: Closing price
- volume: Trading volume

## Usage

### Running the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Backtesting Options
1. **Use existing data**: Place CSV files in the `stock_data/` directory
2. **Upload files**: Use the file uploader in the sidebar
3. **Adjust parameters**: Customize strategy parameters in the sidebar

### Performance Metrics
The dashboard provides:
- Equity curve visualization
- Performance metrics (Return, Win Rate, Sharpe Ratio, etc.)
- Trade analysis with heatmaps and scatter plots
- Detailed trade logs
- Strategy configuration summary

## Configuration

Modify `config/config.yaml` to adjust strategy parameters:

```yaml
strategy:
  base_capital: 1000000
  risk_per_trade: 0.005
  stop_loss: 0.005
  profit_target: 0.02
  trailing_stop: 0.0075
  profit_trail_start: 0.005

indicators:
  ema_fast: 3
  ema_slow: 10
  ema_trend: 50
  rsi_period: 14
  rsi_overbought: 60
  rsi_oversold: 30

data:
  timeframe_primary: 10min
  timeframe_secondary: 1h
  market_start: "09:15"
  market_end: "15:30"
  selection_time: "09:25"
  selection_minutes: 10
  top_stocks_count: 10
```

## Results

Backtest results are saved in the `results/` directory:
- `trade_log.csv`: Detailed log of all trades
- `trades.csv`: Summary of all trades
- `equity_curve.csv`: Daily equity values

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all files are in the correct directory structure
2. **Data format issues**: Verify CSV files have the correct columns and format
3. **Memory issues**: Close other applications if you encounter memory errors

### Debug Mode

Enable debug mode by setting the environment variable:
```bash
export DEBUG=true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used for live trading without proper testing and validation. The authors are not responsible for any financial losses incurred through the use of this software.
```

## Simplified Project Structure

```
Intraday_Equity/
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── indicators.py
│   ├── backtesting.py
│   ├── performance.py
│   └── utils.py
├── app.py
├── stock_data/
├── results/
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your CSV files in the `stock_data` directory

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Key Features of the Streamlit Application

1. **Interactive Parameter Tuning**: Adjust all strategy parameters through the sidebar
2. **File Upload**: Upload CSV files directly through the web interface
3. **Visual Analytics**: Interactive charts and visualizations of backtest results
4. **Performance Metrics**: Comprehensive performance reporting with key metrics
5. **Export Functionality**: Download results in CSV format

This simplified setup focuses on the Streamlit application for easy local deployment and usage without the complexity of Docker containers.