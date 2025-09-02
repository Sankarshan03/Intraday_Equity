# Enhanced Intraday EMA-RSI Trading Strategy

## Updated README.md

```markdown
# Intraday EMA-RSI Trading Strategy

A comprehensive intraday trading strategy implementation using Exponential Moving Averages (EMA) and Relative Strength Index (RSI) for the Indian stock market (NSE).

## 🚀 Features

- **Automated Stock Selection**: Selects top 10 stocks by turnover at 9:25 AM
- **Dual Strategy**: Supports both long and short positions with optimized entry conditions
- **Technical Indicators**: EMA(3), EMA(10), EMA(50), and RSI(14) on consistent timeframes
- **Advanced Risk Management**: Stop loss, profit targets, and trailing stops with tolerance buffers
- **Comprehensive Backtesting Engine**: Detailed performance metrics and analytics
- **Interactive Streamlit Dashboard**: User-friendly web interface for strategy analysis
- **Flexible Data Handling**: Supports both file uploads and existing data directory

## 📊 Strategy Rules (Enhanced)

### Stock Selection
- At 9:25 AM, select top 10 stocks by turnover
- Turnover = Σ(Volume × Close price) for first 10 minutes (9:15-9:25)

### Long Entry (Optimized)
- EMA(3) > EMA(10) (with 0.1% buffer)
- RSI(14) > 55 (reduced from 60 for more opportunities)
- Close > EMA(50) (with 0.2% buffer)
- Buy at high of entry candle

### Short Entry (Optimized)
- EMA(3) < EMA(10) (with 0.1% buffer)
- RSI(14) < 35 (increased from 30 for more opportunities)
- Close < EMA(50) (with 0.2% buffer)
- Sell at low of last 5 minutes of 1-minute candles

### Exit Rules
- Stop loss: 0.5%
- Profit target: 2%
- Trailing stop: 0.75% after 0.5% profit
- Risk per trade: 0.5% of allocated capital

## 🗂️ Project Structure

```
Intraday_Equity/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── indicators.py      # Technical indicator calculations
│   ├── backtesting.py     # Backtesting engine (optimized)
│   ├── performance.py     # Performance metrics calculation
│   └── utils.py           # Utility functions
├── app.py                 # Streamlit dashboard
├── stock_data/            # Directory for CSV files (create if doesn't exist)
├── results/               # Backtest results output (auto-created)
├── config/                # Configuration files
│   └── config.yaml        # Strategy configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps
1. Clone or download the repository
2. Navigate to the project directory:
   ```bash
   cd Intraday_Equity
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Create a `stock_data` directory and place your CSV files with the naming convention `dataNSE_YYYYMMDD.csv`

### CSV Format
Expected CSV columns:
- ticker: Stock symbol (e.g., "RELIANCE", "HDFCBANK")
- time: Timestamp in format 'YYYY-MM-DD HH:MM:SS'
- open: Opening price (numeric)
- high: High price (numeric)
- low: Low price (numeric)
- close: Closing price (numeric)
- volume: Trading volume (numeric)

Example:
```
ticker,time,open,high,low,close,volume
RELIANCE,2025-08-01 09:15:00,2500.0,2505.5,2498.0,2502.5,10000
RELIANCE,2025-08-01 09:16:00,2503.0,2507.0,2501.5,2505.0,8500
```

## 🎯 Usage

### Running the Streamlit Dashboard
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

### Backtesting Options
1. **Use existing data**: Place CSV files in the `stock_data/` directory
2. **Upload files**: Use the file uploader in the sidebar to add CSV files
3. **Adjust parameters**: Customize all strategy parameters in the sidebar

### Performance Metrics
The dashboard provides comprehensive analytics:
- **Equity curve visualization**: Interactive chart of portfolio performance
- **Key metrics**: Total return, win rate, profit factor, Sharpe ratio, max drawdown
- **Trade analysis**: Heatmaps, scatter plots, and detailed trade statistics
- **Strategy configuration**: Summary of all parameters used in the backtest
- **Export functionality**: Download all results as CSV files

## ⚡ Configuration

The strategy parameters can be modified in two ways:
1. Directly in `config/config.yaml`
2. Through the interactive sidebar in the Streamlit app

Default configuration:
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
  rsi_overbought: 55    # Reduced from 60 for more opportunities
  rsi_oversold: 35      # Increased from 30 for more opportunities

data:
  timeframe_primary: 10min
  market_start: "09:15"
  market_end: "15:30"
  selection_time: "09:25"
  selection_minutes: 10
  top_stocks_count: 10
```

## 📈 Results

Backtest results are automatically saved in the `results/` directory:
- `trade_log.csv`: Detailed log of all executed trades
- `trades.csv`: Summary of all trades with performance metrics
- `equity_curve.csv`: Daily equity values for drawdown analysis

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: 
   - Ensure all files are in the correct directory structure
   - Run `pip install -r requirements.txt` to install all dependencies

2. **Data format issues**:
   - Verify CSV files have the correct columns and format
   - Check that numeric columns contain valid numbers

3. **Memory issues**:
   - Close other applications if you encounter memory errors
   - Consider using smaller datasets for initial testing

4. **No trades executed**:
   - Adjust strategy parameters to be less restrictive
   - Check that your data contains the required time periods

### Debug Mode

Enable debug mode by setting the environment variable:
```bash
export DEBUG=true
```

For Windows Command Prompt:
```cmd
set DEBUG=true
```

For Windows PowerShell:
```powershell
$env:DEBUG="true"
```

## 🔄 Recent Optimizations

The strategy has been enhanced with several optimizations:

1. **Consistent Timeframes**: All indicators now use the same 10-minute timeframe for better synchronization
2. **Tolerance Buffers**: Added small buffers to entry conditions to capture more opportunities
3. **Improved Short Entry**: Fixed short entry price calculation using 1-minute data
4. **Enhanced Risk Management**: Better position sizing and risk calculation

## 📝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests if applicable
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

**Important**: This software is for educational and research purposes only. It should not be used for live trading without proper testing and validation. The authors are not responsible for any financial losses incurred through the use of this software.

Past performance is not indicative of future results. Always test strategies thoroughly with historical data and paper trading before considering live deployment.
```

## Key Enhancements in the Updated README:

1. **Clearer Structure**: Improved organization with emojis and sections for better readability
2. **Updated Strategy Rules**: Reflected the optimized parameters (RSI thresholds with buffers)
3. **Detailed Installation Instructions**: Added specific commands for different operating systems
4. **Enhanced Configuration Section**: Explained the optimized parameters and their rationale
5. **Troubleshooting Guide**: Expanded with OS-specific instructions and common solutions
6. **Recent Optimizations Section**: Highlighted the key improvements made to the strategy
7. **Clear Disclaimer**: Emphasized the educational purpose and risks involved

This updated README provides comprehensive documentation for users to understand, install, and use the enhanced intraday trading strategy effectively.