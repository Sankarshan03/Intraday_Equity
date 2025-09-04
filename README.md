# Intraday EMA-RSI Trading Strategy

A comprehensive intraday trading strategy implementation using Exponential Moving Averages (EMA) and Relative Strength Index (RSI) for the Indian stock market (NSE). This system provides automated backtesting, performance analysis, and an interactive Streamlit dashboard for strategy optimization.

## 🚀 Key Features

- **Automated Stock Selection**: Dynamically selects top 10 stocks by turnover at 9:25 AM
- **Multi-Timeframe Analysis**: EMA(3) & EMA(10) on 10-minute, EMA(50) on 1-hour, RSI(14) on 10-minute
- **Dual Strategy Support**: Both long and short positions with optimized entry/exit conditions
- **Advanced Risk Management**: Dynamic position sizing, stop loss, profit targets, and trailing stops
- **Comprehensive Backtesting**: Detailed trade logging and performance analytics
- **Interactive Dashboard**: Feature-rich Streamlit web interface with advanced visualizations
- **Flexible Data Input**: Support for CSV uploads and existing data directories
- **Real-time Performance Metrics**: Sharpe ratio, drawdown analysis, win rate tracking

## 📊 Trading Strategy Rules

### Stock Selection Criteria
- **Selection Time**: 9:25 AM daily
- **Selection Period**: First 10 minutes (9:15-9:25 AM)
- **Criteria**: Top 10 stocks by turnover (Volume × Close price)
- **Trading Start**: 9:26 AM onwards (post-selection)

### Entry Conditions

#### Long Entry
- **EMA Signal**: EMA(3) > EMA(10) with 0.1% tolerance buffer
- **RSI Signal**: RSI(14) > 60 (configurable, optimized to 55 with buffer)
- **Trend Filter**: Close > EMA(50) with 0.2% tolerance buffer
- **Entry Price**: High of the entry candle

#### Short Entry
- **EMA Signal**: EMA(3) < EMA(10) with 0.1% tolerance buffer
- **RSI Signal**: RSI(14) < 30 (configurable, optimized to 35 with buffer)
- **Trend Filter**: Close < EMA(50) with 0.2% tolerance buffer
- **Entry Price**: Low of last 5 minutes (1-minute candles)

### Risk Management
- **Base Capital**: ₹10,00,000 (configurable)
- **Risk per Trade**: 0.5% of current capital
- **Stop Loss**: 0.5% from entry price
- **Profit Target**: 2% from entry price
- **Trailing Stop**: 0.75% trailing after 0.5% profit
- **Position Sizing**: Dynamic based on risk amount and stop loss distance

## 🗂️ Project Structure

```
Intraday_Equity/
├── src/                          # Core source code modules
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading, preprocessing & stock selection
│   ├── indicators.py            # EMA and RSI technical indicator calculations
│   ├── backtesting.py           # Multi-timeframe backtesting engine
│   ├── performance.py           # Performance metrics & analytics
│   └── utils.py                 # Configuration loading & result saving utilities
├── config/
│   └── config.yaml              # Strategy parameters configuration
├── results/                     # Auto-generated backtest outputs
│   ├── trade_log.csv           # Detailed trade execution log
│   ├── trades.csv              # Trade summary with P&L analysis
│   └── equity_curve.csv        # Portfolio equity progression
├── stock_data_aug_2025/        # Sample NSE data directory
├── temp_uploaded_data/         # Temporary storage for uploaded files
├── app.py                      # Interactive Streamlit dashboard
├── main.py                     # Command-line backtesting interface
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
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

### Option 1: Interactive Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
- Access the web interface at `http://localhost:8501`
- Upload CSV files or use existing data in `stock_data_aug_2025/` folder
- Adjust strategy parameters via interactive sidebar controls
- View real-time results with advanced visualizations

### Option 2: Command Line Interface
```bash
python main.py
```
- Runs backtest using configuration from `config/config.yaml`
- Uses data from `stock_data_aug_2025/` directory
- Outputs results to console and saves to `results/` folder

### Dashboard Features
The Streamlit interface provides comprehensive analytics:

#### 📊 Performance Analysis
- **Real-time Metrics**: Total return, max drawdown, win rate, Sharpe ratio
- **Equity Curve**: Interactive portfolio performance visualization with drawdown overlay
- **Trade Distribution**: P&L histograms and win/loss analysis

#### 📈 Advanced Analytics
- **Time Analysis**: Performance by hour, day of week, and monthly heatmaps
- **Symbol Performance**: Top/worst performing stocks with detailed breakdowns
- **Trade Patterns**: Hold time vs returns, streak analysis, exit reason breakdown
- **Selected Stocks Display**: Shows the top 10 stocks selected by turnover at 9:25 AM

#### 🔧 Interactive Controls
- **Parameter Adjustment**: Real-time strategy parameter modification
- **Data Upload**: Drag-and-drop CSV file support
- **Export Functionality**: Download trade logs and detailed results as CSV

## ⚡ Configuration

The strategy parameters can be modified in two ways:
1. Directly in `config/config.yaml`
2. Through the interactive sidebar in the Streamlit app

### Default Configuration (`config/config.yaml`):
```yaml
strategy:
  base_capital: 1000000        # ₹10 Lakh starting capital
  risk_per_trade: 0.005        # 0.5% risk per trade
  stop_loss: 0.005             # 0.5% stop loss
  profit_target: 0.02          # 2% profit target
  trailing_stop: 0.0075        # 0.75% trailing stop
  profit_trail_start: 0.005    # Start trailing after 0.5% profit

indicators:
  ema_fast: 3                  # Fast EMA period
  ema_slow: 10                 # Slow EMA period
  ema_trend: 50                # Trend EMA period (1-hour timeframe)
  rsi_period: 14               # RSI calculation period
  rsi_overbought: 60           # RSI overbought threshold
  rsi_oversold: 30             # RSI oversold threshold

data:
  timeframe_primary: 10T       # 10-minute primary timeframe
  timeframe_secondary: 1H      # 1-hour secondary timeframe
  market_start: "09:15"        # Market opening time
  market_end: "15:30"          # Market closing time
  selection_time: "09:25"      # Stock selection time
  selection_minutes: 10        # Selection period duration
  top_stocks_count: 10         # Number of stocks to select
```

### Parameter Optimization
The backtesting engine includes tolerance buffers for more robust signal generation:
- **EMA conditions**: 0.1% buffer to reduce noise
- **RSI thresholds**: 5-point buffer (55 instead of 60 for long, 35 instead of 30 for short)
- **Trend filter**: 0.2% buffer for EMA(50) comparison

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

## 🔄 Technical Implementation Highlights

### Multi-Timeframe Architecture
- **Primary Indicators**: EMA(3), EMA(10), RSI(14) calculated on 10-minute OHLC data
- **Trend Filter**: EMA(50) calculated on 1-hour OHLC data, forward-filled to 10-minute intervals
- **Entry Precision**: Short entries use 1-minute data for precise low-of-last-5-minutes calculation

### Advanced Features
- **Dynamic Stock Selection**: Real-time turnover calculation during 9:15-9:25 AM window
- **Tolerance Buffers**: Noise reduction through percentage-based signal buffers
- **Position Sizing**: Risk-based quantity calculation with minimum 1-share constraint
- **Comprehensive Logging**: Detailed trade entry/exit tracking with reason codes

### Performance Optimizations
- **Memory Efficient**: Processes data in chunks by symbol and date
- **Vectorized Calculations**: Pandas-based indicator computation for speed
- **Modular Design**: Separate modules for data loading, indicators, backtesting, and performance analysis

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

## 📚 Additional Resources

- **Strategy Documentation**: Based on memory requirements for ₹10 lakh base capital with 0.5% risk management
- **Data Source**: Designed for NSE August 2025 stock data with 1-minute resolution
- **Performance Targets**: Optimized for intraday trading with 2% profit targets and 0.5% stop losses
- **Indicator Settings**: EMA(3), EMA(10) on 10-min + EMA(50) on 1-hour + RSI(14) configuration

---

*This implementation follows the intraday trading strategy requirements with automated stock selection, multi-timeframe analysis, and comprehensive risk management for the Indian equity market.*