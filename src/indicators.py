import pandas as pd
import numpy as np
from numba import jit, njit
import warnings
warnings.filterwarnings('ignore')

@njit
def calculate_ema_numba(prices, period):
    """
    Fast EMA calculation using Numba JIT compilation
    """
    alpha = 2.0 / (period + 1.0)
    ema = np.empty_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def calculate_ema(data, period):
    """
    Calculate Exponential Moving Average - optimized version
    """
    if isinstance(data, pd.Series):
        # Use numba for better performance
        prices = data.values
        ema_values = calculate_ema_numba(prices, period)
        return pd.Series(ema_values, index=data.index)
    else:
        # Fallback to pandas
        return data.ewm(span=period, adjust=False).mean()

@njit
def calculate_rsi_numba(prices, period):
    """
    Fast RSI calculation using Numba JIT compilation
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    rsi = np.empty(len(prices))
    rsi[0] = 50.0  # Initial RSI value
    
    if len(gains) < period:
        rsi[:] = 50.0
        return rsi
    
    # Calculate initial average gain and loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(prices)):
        # Smoothed moving average
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Fill initial values
    for i in range(1, min(period, len(prices))):
        rsi[i] = 50.0
    
    return rsi

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index - optimized version
    """
    if isinstance(data, pd.Series) and len(data) > period:
        # Use numba for better performance
        prices = data.values
        rsi_values = calculate_rsi_numba(prices, period)
        return pd.Series(rsi_values, index=data.index)
    else:
        # Fallback to pandas for small datasets
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