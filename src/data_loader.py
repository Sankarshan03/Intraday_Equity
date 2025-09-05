import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime
from numba import njit
import warnings
warnings.filterwarnings('ignore')

def load_data_from_folder(folder_path):
    """
    Load all CSV files from the specified folder - optimized version
    """
    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")
    
    # Read and concatenate all CSV files with optimizations
    dfs = []
    for file in csv_files:
        try:
            print(f"Loading file: {file}")
            # Use optimized pandas settings for faster loading
            df = pd.read_csv(file, 
                           engine='c',  # Use C engine for faster parsing
                           low_memory=False,  # Read entire file into memory
                           dtype={'volume': 'float32', 'open': 'float32', 
                                 'high': 'float32', 'low': 'float32', 'close': 'float32'})
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid CSV files could be read")
    
    # Use more efficient concatenation
    combined_df = pd.concat(dfs, ignore_index=True, copy=False)
    return combined_df

def preprocess_data(df):
    """
    Preprocess the raw data from CSV files
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Check if the expected columns exist
    required_cols = ['ticker', 'time', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert time column to datetime
    df['datetime'] = pd.to_datetime(df['time'])
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    
    # Rename columns to match our expected format
    df.rename(columns={
        'ticker': 'Symbol',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Ensure numeric columns are properly formatted
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['Symbol', 'Close'])
    
    return df

def filter_market_hours(df, market_start="09:15", market_end="15:30"):
    """
    Filter data to only include market hours
    """
    # Convert to time for filtering
    time_filter = (df.index.time >= pd.to_datetime(market_start).time()) & \
                  (df.index.time <= pd.to_datetime(market_end).time())
    return df[time_filter]

@njit
def calculate_turnover_numba(volumes, closes):
    """
    Fast turnover calculation using Numba
    """
    return np.sum(volumes * closes)

def select_top_stocks_by_turnover(df, selection_time="09:25", selection_minutes=10):
    """
    Select top stocks by turnover for the first 10 minutes - optimized version
    """
    if df.empty:
        return []
    
    # Get the date from the data
    date = df.index[0].date()
    selection_start = pd.to_datetime(f"{date} {selection_time}") - pd.Timedelta(minutes=selection_minutes)
    selection_end = pd.to_datetime(f"{date} {selection_time}")
    
    # Filter data for the selection period
    selection_data = df[(df.index >= selection_start) & (df.index <= selection_end)]
    
    if selection_data.empty:
        return []
    
    # Calculate turnover for each stock using optimized method
    turnover_dict = {}
    for symbol, group in selection_data.groupby('Symbol'):
        volumes = group['Volume'].values
        closes = group['Close'].values
        turnover_dict[symbol] = calculate_turnover_numba(volumes, closes)
    
    # Convert to series and select top 10 stocks
    turnover = pd.Series(turnover_dict)
    top_stocks = turnover.nlargest(10).index.tolist()
    
    print(f"Top 10 stocks by turnover: {top_stocks}")
    return top_stocks

def prepare_data_for_backtesting(folder_path, market_start="09:15", market_end="15:30"):
    """
    Complete data preparation pipeline
    """
    # Load data from folder
    df = load_data_from_folder(folder_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Filter market hours
    df = filter_market_hours(df, market_start, market_end)
    
    # Select top stocks by turnover
    top_stocks = select_top_stocks_by_turnover(df)
    
    # Filter data to only include top stocks
    if top_stocks:
        df = df[df['Symbol'].isin(top_stocks)]
    
    return df, folder_path