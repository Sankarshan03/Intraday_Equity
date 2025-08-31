# Complete Fix for Intraday EMA-RSI Strategy with Docker Deployment

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
- **Docker Support**: Containerized deployment

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
├── Dockerfile             # Docker configuration
├── docker-compose.yml     # Docker compose configuration
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized deployment)

### Local Installation
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

## Docker Deployment

### Using Docker Compose (Recommended)

1. Build and run the application:
   ```bash
   docker-compose up -d
   ```

2. Access the application at `http://localhost:8501`

3. To stop the application:
   ```bash
   docker-compose down
   ```

### Manual Docker Build

1. Build the Docker image:
   ```bash
   docker build -t intraday-strategy .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 -v $(pwd)/stock_data:/app/stock_data -v $(pwd)/results:/app/results intraday-strategy
   ```

### Docker Compose Configuration

The `docker-compose.yml` file includes:
- Port mapping (8501:8501)
- Volume mounts for data and results
- Environment variables
- Resource limits

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
3. **Docker permissions**: Use `chmod` to set appropriate permissions on mounted volumes
4. **Memory issues**: Increase Docker memory allocation if needed

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

## Docker Deployment Files

### Dockerfile

```dockerfile
# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p stock_data results

# Expose port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  intraday-strategy:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./stock_data:/app/stock_data
      - ./results:/app/results
    environment:
      - PYTHONPATH=/app
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### .dockerignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
stock_data/*
!stock_data/.gitkeep
results/*
!results/.gitkeep

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# Git
.git/
.gitignore
```

## Deployment Instructions

### 1. Build and Run with Docker Compose

```bash
# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

### 2. Manual Docker Deployment

```bash
# Build the image
docker build -t intraday-strategy .

# Run the container
docker run -d \
  -p 8501:8501 \
  -v $(pwd)/stock_data:/app/stock_data \
  -v $(pwd)/results:/app/results \
  --name intraday-app \
  intraday-strategy

# View running container
docker ps

# View logs
docker logs -f intraday-app

# Stop the container
docker stop intraday-app
```

### 3. Environment Variables

You can customize the deployment using environment variables:

```bash
# Set custom port
export STREAMLIT_SERVER_PORT=8502

# Enable debug mode
export DEBUG=true

# Set memory limits
export DOCKER_MEMORY_LIMIT=4G
```

### 4. Data Persistence

To ensure your data persists between container restarts:

1. Create the required directories:
   ```bash
   mkdir -p stock_data results
   ```

2. Place your CSV files in the `stock_data` directory

3. The `results` directory will contain backtest outputs

### 5. Updating the Application

To update the application:

```bash
# Pull latest changes
git pull

# Rebuild the container
docker-compose build

# Restart the service
docker-compose up -d
```

## Troubleshooting Docker Deployment

### Common Issues

1. **Port already in use**:
   ```bash
   # Change the port in docker-compose.yml
   ports:
     - "8502:8501"
   ```

2. **Permission denied**:
   ```bash
   # Change ownership of directories
   sudo chown -R $USER:$USER stock_data results
   ```

3. **Out of memory**:
   ```bash
   # Increase memory limits in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 4G
   ```

4. **Docker daemon not running**:
   ```bash
   # Start Docker service
   sudo systemctl start docker
   ```

### Debugging

To debug container issues:

```bash
# Run in foreground mode
docker-compose up

# Access container shell
docker exec -it intraday-strategy /bin/bash

# View container logs
docker logs intraday-strategy
```

This complete setup provides a robust, containerized deployment of your intraday trading strategy with a user-friendly web interface for backtesting and analysis.