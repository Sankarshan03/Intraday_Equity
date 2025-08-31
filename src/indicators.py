import pandas as pd
import numpy as np

def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average
    """
    return data.ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_all_indicators(df, config):
    """
    Calculate all required indicators for the strategy
    """
    # Group by symbol and calculate indicators
    results = []
    
    for symbol, group in df.groupby('Symbol'):
        group = group.copy()
        
        # Calculate EMAs and RSI
        group['EMA3'] = calculate_ema(group['Close'], config['ema_fast'])
        group['EMA10'] = calculate_ema(group['Close'], config['ema_slow'])
        group['RSI14'] = calculate_rsi(group['Close'], config['rsi_period'])
        
        results.append(group)
    
    # Combine all symbols
    result_df = pd.concat(results)
    
    return result_df