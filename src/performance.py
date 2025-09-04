import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_performance_metrics(trades_df, equity_curve):
    """
    Calculate key performance metrics for trading strategy
    Returns: Total Return, Max Drawdown, Win Rate, Sharpe Ratio
    """
    if trades_df.empty or len(equity_curve) <= 1:
        return {
            'total_return': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'sharpe_ratio': 0
        }
    
    # Basic trade statistics
    total_trades = len(trades_df)
    winning_trades = trades_df[trades_df['PnL'] > 0]
    
    num_winning = len(winning_trades)
    win_rate = (num_winning / total_trades * 100) if total_trades > 0 else 0
    
    # Return calculations
    initial_capital = equity_curve[0]
    final_capital = equity_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital * 100
    
    # Drawdown analysis
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Sharpe Ratio calculation using trade returns
    if not trades_df.empty and 'Return_Pct' in trades_df.columns:
        # Use trade return percentages for Sharpe ratio
        trade_returns = trades_df['Return_Pct'] / 100  # Convert percentage to decimal
        
        if len(trade_returns) > 1:
            mean_return = trade_returns.mean()
            std_return = trade_returns.std()
            
            # For intraday trading, assume risk-free rate = 0
            # Sharpe ratio = (Mean Return - Risk Free Rate) / Standard Deviation
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0
    else:
        sharpe_ratio = 0
    
    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio
    }

def generate_performance_report(metrics, trade_report):
    """
    Generate performance report with simplified metrics
    """
    report = {
        'Total Return (%)': metrics['total_return'],
        'Max Drawdown (%)': metrics['max_drawdown'],
        'Win Rate (%)': metrics['win_rate'],
        'Sharpe Ratio': metrics['sharpe_ratio'],
        'Number of Trades': len(trade_report)
    }
    
    return pd.DataFrame.from_dict(report, orient='index', columns=['Value'])

def get_selected_stocks_info(backtest_engine):
    """
    Get information about stocks selected for trading based on turnover criteria
    """
    if not hasattr(backtest_engine, 'selected_stocks') or not backtest_engine.selected_stocks:
        return pd.DataFrame()
    
    # Get the original data to calculate turnover
    if backtest_engine.original_data is None:
        return pd.DataFrame()
    
    data = backtest_engine.original_data
    
    # Calculate turnover for selected stocks during selection period (9:15-9:25)
    date = data.index[0].date()
    selection_start = pd.to_datetime(f"{date} 09:15")
    selection_end = pd.to_datetime(f"{date} 09:25")
    
    selection_data = data[(data.index >= selection_start) & (data.index <= selection_end)]
    
    if selection_data.empty:
        return pd.DataFrame()
    
    # Calculate turnover for each selected stock
    turnover_data = []
    for symbol in backtest_engine.selected_stocks:
        stock_data = selection_data[selection_data['Symbol'] == symbol]
        if not stock_data.empty:
            turnover = (stock_data['Volume'] * stock_data['Close']).sum()
            avg_price = stock_data['Close'].mean()
            total_volume = stock_data['Volume'].sum()
            
            turnover_data.append({
                'Symbol': symbol,
                'Turnover': turnover,
                'Avg_Price': avg_price,
                'Total_Volume': total_volume,
                'Selection_Period': '9:15-9:25 AM'
            })
    
    # Create DataFrame and sort by turnover (highest first)
    turnover_df = pd.DataFrame(turnover_data)
    if not turnover_df.empty:
        turnover_df = turnover_df.sort_values('Turnover', ascending=False)
    
    return turnover_df