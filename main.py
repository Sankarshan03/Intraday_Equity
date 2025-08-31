import pandas as pd
import numpy as np
from src.data_loader import prepare_data_for_backtesting
from src.backtesting import BacktestEngine
from src.performance import calculate_performance_metrics, generate_performance_report
from src.utils import load_config, save_results

def main():
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Prepare data
    print("Loading and preprocessing data...")
    data_folder = "stock_data_aug_2025"  # Update this path if needed
    
    try:
        df, file_name = prepare_data_for_backtesting(
            data_folder, 
            market_start=config['data']['market_start'],
            market_end=config['data']['market_end']
        )
        
        print(f"Data loaded from: {file_name}")
        print(f"Number of stocks selected: {df['Symbol'].nunique()}")
        print(f"Data timeframe: {df.index.min()} to {df.index.max()}")
        
        # Initialize backtest engine
        backtester = BacktestEngine(config)
        
        # Run backtest
        print("Running backtest...")
        trade_log, trades, equity_curve = backtester.run_backtest(df)
        
        # Convert to DataFrames
        trade_log_df = pd.DataFrame(trade_log)
        trades_df = pd.DataFrame(trades)
        
        # Calculate performance metrics
        print("Calculating performance metrics...")
        metrics = calculate_performance_metrics(trades_df, equity_curve)
        performance_report = generate_performance_report(metrics, trades_df)
        
        # Display results
        print("\n" + "="*50)
        print("STRATEGY PERFORMANCE REPORT")
        print("="*50)
        print(performance_report)
        
        print(f"\nInitial Capital: ₹{config['strategy']['base_capital']:,.2f}")
        print(f"Final Capital: ₹{equity_curve[-1]:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        
        # Save results
        save_results(trade_log, trades, equity_curve, 'results')
        
        # Display first few trades
        if not trades_df.empty:
            print("\nFirst 5 Trades:")
            print(trades_df.head().to_string(index=False))
        else:
            print("\nNo trades were executed during the backtest period.")
            
    except Exception as e:
        print(f"Error during backtesting: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()