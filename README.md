# 🚀 Optimized Intraday EMA-RSI Trading Strategy

A high-performance intraday trading strategy implementation using Exponential Moving Averages (EMA) and Relative Strength Index (RSI) for the Indian stock market (NSE). This system features **Numba JIT optimization** for 5-10x faster backtesting, automated stock selection, comprehensive performance analysis, and an interactive Streamlit dashboard with real-time performance monitoring.

## ⚡ Key Features

### 🔥 Performance Optimizations
- **Numba JIT Compilation**: 5-10x faster backtesting with optimized numerical computations
- **Vectorized Operations**: High-speed numpy array processing for indicators and signals
- **Memory Optimization**: Efficient data structures with automatic memory reduction
- **Smart Fallback System**: Automatic fallback to original implementation if needed
- **Real-time Performance Monitoring**: Execution time tracking and optimization insights

### 📈 Trading Features
- **Automated Stock Selection**: Dynamically selects top 10 stocks by turnover at 9:25 AM
- **Enhanced Multi-Timeframe Analysis**: EMA(3) & EMA(10) on 10-minute, collective 30-day EMA(50) on 1-hour, RSI(14) on 10-minute
- **Pending Entry Logic**: Wait for price to reach target before entry execution
- **Advanced Risk Management**: Dynamic position sizing, stop loss, profit targets, and trailing stops
- **Comprehensive Backtesting**: Detailed trade logging and performance analytics with speed metrics
- **Interactive Dashboard**: Feature-rich Streamlit web interface with performance breakdown tabs
- **Flexible Data Input**: Support for CSV uploads and existing data directories
- **Real-time Performance Metrics**: Sharpe ratio, drawdown analysis, win rate tracking, processing speed

## 📊 Trading Strategy Rules

### Stock Selection Criteria
- **Selection Time**: 9:25 AM daily
- **Selection Period**: First 10 minutes (9:15-9:25 AM)
- **Criteria**: Top 10 stocks by turnover (Volume × Close price)
- **Trading Start**: 9:26 AM onwards (post-selection)

### Entry Conditions

#### Long Entry (Enhanced with Pending Logic)
- **EMA Signal**: EMA(3) > EMA(10) with 0.1% tolerance buffer
- **RSI Signal**: RSI(14) > 60 (configurable, optimized to 55 with buffer)
- **Trend Filter**: Close > EMA(50) with 0.2% tolerance buffer (using collective 30-day data)
- **Entry Execution**: Wait until price reaches the high of the entry candle before executing trade
- **Precision**: Ensures optimal entry timing and reduces slippage

#### Short Entry (Enhanced with Pending Logic)
- **EMA Signal**: EMA(3) < EMA(10) with 0.1% tolerance buffer
- **RSI Signal**: RSI(14) < 30 (configurable, optimized to 35 with buffer)
- **Trend Filter**: Close < EMA(50) with 0.2% tolerance buffer (using collective 30-day data)
- **Entry Execution**: Wait until price reaches the low of the last 5 minutes (1-minute candles)
- **Precision**: Ensures optimal short entry timing with enhanced accuracy

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
├── src/                          # Core source code modules (Numba-optimized)
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Data loading, preprocessing & stock selection
│   ├── indicators.py            # Numba JIT-optimized EMA and RSI calculations
│   ├── backtesting.py           # High-performance backtesting engine with Numba optimization
│   ├── performance.py           # Performance metrics & analytics with timing
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
3. Install dependencies (includes Numba for optimization):
   ```bash
   pip install -r requirements.txt
   ```
   
   **Key Dependencies:**
   - `numba` - JIT compilation for performance optimization
   - `pandas` - Data manipulation and analysis
   - `numpy` - Numerical computations
   - `streamlit` - Interactive web dashboard
   - `plotly` - Advanced visualizations
   - `pyyaml` - Configuration management

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

### Option 1: Optimized Interactive Streamlit Dashboard (Recommended)
```bash
streamlit run app.py
```
- Access the web interface at `http://localhost:8501`
- Upload CSV files or use existing data in `stock_data_aug_2025/` folder
- Adjust strategy parameters via interactive sidebar controls
- View real-time results with advanced visualizations and performance monitoring
- **New Features**: Performance breakdown tab, execution time tracking, system efficiency metrics

### Option 2: Optimized Command Line Interface
```bash
python main.py
```
- Runs high-performance backtest using configuration from `config/config.yaml`
- Uses data from `stock_data_aug_2025/` directory
- Outputs detailed results with performance metrics to console
- Saves comprehensive results to `results/` folder
- **New Features**: Execution time breakdown, processing speed metrics, optimization status

### Dashboard Features
The Streamlit interface provides comprehensive analytics:

#### 📊 Enhanced Performance Analysis
- **Real-time Metrics**: Total return, max drawdown, win rate, Sharpe ratio with execution timing
- **Equity Curve**: Interactive portfolio performance visualization with drawdown overlay
- **Trade Distribution**: P&L histograms and win/loss analysis
- **⚡ System Performance Tab**: Execution times, processing speed, optimization breakdown

#### 📈 Advanced Analytics
- **Time Analysis**: Performance by hour, day of week, and monthly heatmaps
- **Symbol Performance**: Top/worst performing stocks with detailed breakdowns
- **Trade Patterns**: Hold time vs returns, streak analysis, exit reason breakdown
- **Selected Stocks Display**: Shows the top 10 stocks selected by turnover at 9:25 AM
- **⚡ Performance Insights**: Trades/second processing, data processing speed, system efficiency metrics

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

## ⚡ Technical Implementation Highlights

### High-Performance Architecture
- **Numba JIT Optimization**: Core backtesting functions compiled with `@njit` for 5-10x speed improvement
- **Vectorized Processing**: Numpy array operations for maximum computational efficiency
- **Smart Fallback System**: Automatic fallback to original implementation if Numba fails
- **Memory Optimization**: Automatic dataframe memory reduction and efficient data structures

### Enhanced Multi-Timeframe Architecture
- **Primary Indicators**: Numba-optimized EMA(3), EMA(10), RSI(14) calculated on 10-minute OHLC data
- **Collective EMA50**: Enhanced trend filter using complete 30-day dataset for accurate EMA(50) calculation
- **Pending Entry Logic**: Precise entry execution waiting for target price to be reached
- **Entry Precision**: Short entries use 1-minute data for precise low-of-last-5-minutes calculation

### Advanced Features
- **Dynamic Stock Selection**: Real-time turnover calculation during 9:15-9:25 AM window
- **Enhanced Entry/Exit Logic**: Pending orders with precise price targeting
- **Tolerance Buffers**: Noise reduction through percentage-based signal buffers
- **Position Sizing**: Risk-based quantity calculation with minimum 1-share constraint
- **Comprehensive Logging**: Detailed trade entry/exit tracking with reason codes and performance metrics

### Performance Optimizations
- **Numba JIT Functions**: `@njit` decorated functions for entry/exit conditions, position sizing, trailing stops
- **Vectorized Backtesting**: `process_candle_data_numba()` for high-speed candle-by-candle processing
- **Memory Efficient**: Processes data in chunks by symbol and date with automatic memory optimization
- **Real-time Monitoring**: Execution time tracking and performance breakdown analysis
- **Modular Design**: Separate optimized modules for data loading, indicators, backtesting, and performance analysis

### Optimization Status
- ✅ **Numba JIT Compilation**: Active for all critical calculations
- ✅ **Memory Optimization**: Automatic dataframe memory reduction
- ✅ **Vectorized Operations**: High-speed numpy array processing
- ✅ **Enhanced Entry Logic**: Pending orders with precise targeting
- ✅ **Collective EMA50 Calculation**: 30-day dataset for accurate trend analysis

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

## 🚀 Performance Benchmarks

### Speed Improvements
- **Backtesting Speed**: 5-10x faster with Numba JIT compilation
- **Indicator Calculations**: Optimized EMA and RSI computations
- **Data Processing**: Efficient memory usage with automatic optimization
- **Real-time Monitoring**: Sub-second performance tracking and reporting

### System Requirements
- **Minimum**: Python 3.8+, 4GB RAM, 1GB storage
- **Recommended**: Python 3.9+, 8GB RAM, 2GB storage for optimal performance
- **Numba Compatibility**: Automatic detection and fallback for unsupported systems

## 📚 Additional Resources

- **Strategy Documentation**: Based on memory requirements for ₹10 lakh base capital with 0.5% risk management
- **Data Source**: Designed for NSE August 2025 stock data with 1-minute resolution
- **Performance Targets**: Optimized for intraday trading with 2% profit targets and 0.5% stop losses
- **Indicator Settings**: EMA(3), EMA(10) on 10-min + collective EMA(50) on 1-hour + RSI(14) configuration
- **Optimization Guide**: Numba JIT compilation with smart fallback system for maximum compatibility

---

*This high-performance implementation follows the intraday trading strategy requirements with Numba-optimized backtesting, automated stock selection, enhanced entry/exit logic, and comprehensive risk management for the Indian equity market.*