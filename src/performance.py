import pandas as pd
import numpy as np

def calculate_performance_metrics(trades_df, equity_curve):
    """
    Calculate performance metrics
    """
    if trades_df.empty or len(equity_curve) <= 1:
        return {
            'total_return': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
    
    # Calculate basic metrics
    total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
    
    winning_trades = trades_df[trades_df['PnL'] > 0]
    losing_trades = trades_df[trades_df['PnL'] <= 0]
    
    win_rate = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    avg_profit = winning_trades['PnL'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['PnL'].mean() if len(losing_trades) > 0 else 0
    
    total_profit = winning_trades['PnL'].sum()
    total_loss = abs(losing_trades['PnL'].sum())
    profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')
    
    # Calculate maximum drawdown
    equity_series = pd.Series(equity_curve)
    rolling_max = equity_series.expanding().max()
    drawdowns = (equity_series - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0 for intraday)
    daily_returns = equity_series.pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 and daily_returns.std() != 0 else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def generate_performance_report(metrics, trade_report):
    """
    Generate performance report
    """
    report = {
        'Total Return (%)': metrics['total_return'],
        'Number of Trades': len(trade_report),
        'Win Rate (%)': metrics['win_rate'],
        'Average Profit': metrics['avg_profit'],
        'Average Loss': metrics['avg_loss'],
        'Profit Factor': metrics['profit_factor'],
        'Max Drawdown (%)': metrics['max_drawdown'],
        'Sharpe Ratio': metrics['sharpe_ratio']
    }
    
    return pd.DataFrame.from_dict(report, orient='index', columns=['Value'])