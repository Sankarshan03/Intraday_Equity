import os
import pandas as pd
import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from YAML file
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def save_results(trade_log, trades, equity_curve, output_path):
    """
    Save backtest results to CSV files, creating directory if needed
    """
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Convert to DataFrames
    trade_log_df = pd.DataFrame(trade_log)
    trades_df = pd.DataFrame(trades)
    equity_curve_df = pd.DataFrame(equity_curve, columns=['Equity'])
    
    # Ensure numeric columns are properly formatted to avoid Arrow serialization issues
    numeric_cols = ['Price', 'Quantity', 'Value', 'PnL']
    for col in numeric_cols:
        if col in trade_log_df.columns:
            trade_log_df[col] = pd.to_numeric(trade_log_df[col], errors='coerce')
        if col in trades_df.columns:
            trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')
    
    # Save to CSV
    trade_log_df.to_csv(f"{output_path}/trade_log.csv", index=False)
    trades_df.to_csv(f"{output_path}/trades.csv", index=False)
    equity_curve_df.to_csv(f"{output_path}/equity_curve.csv", index=False)
    
    print(f"Results saved to {output_path}")

def format_currency(amount):
    """
    Format number as currency
    """
    return f"₹{amount:,.2f}"

def format_percentage(amount):
    """
    Format number as percentage
    """
    return f"{amount:.2f}%"