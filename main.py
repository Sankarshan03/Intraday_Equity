import pandas as pd
import numpy as np
import time
import warnings
from src.data_loader import prepare_data_for_backtesting
from src.backtesting import BacktestEngine
from src.performance import calculate_performance_metrics, generate_performance_report
from src.utils import load_config, save_results

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def performance_timer(operation_name, start_time):
    """Calculate and display performance timing"""
    elapsed = time.time() - start_time
    print(f"⚡ {operation_name}: {elapsed:.3f}s")
    return elapsed

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    original_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    new_memory = df.memory_usage(deep=True).sum() / 1024**2
    memory_saved = ((original_memory - new_memory) / original_memory * 100)
    
    if memory_saved > 0:
        print(f"💾 Memory optimized: {memory_saved:.1f}% reduction ({original_memory:.1f}MB → {new_memory:.1f}MB)")
    
    return df

def main():
    print("🚀 Starting Optimized Intraday EMA-RSI Strategy Backtester")
    print("="*60)
    
    # Performance tracking
    performance_times = {}
    total_start_time = time.time()
    
    # Load configuration
    config_start_time = time.time()
    config = load_config('config/config.yaml')
    performance_times['config_loading'] = performance_timer("Config Loading", config_start_time)
    
    # Prepare data with performance monitoring
    print("📊 Loading and preprocessing data...")
    data_folder = "stock_data_aug_2025"  # Update this path if needed
    
    try:
        data_start_time = time.time()
        df, file_name = prepare_data_for_backtesting(
            data_folder, 
            market_start=config['data']['market_start'],
            market_end=config['data']['market_end']
        )
        performance_times['data_loading'] = performance_timer("Data Loading", data_start_time)
        
        # Optimize memory usage
        memory_start_time = time.time()
        df = optimize_dataframe_memory(df)
        performance_times['memory_optimization'] = performance_timer("Memory Optimization", memory_start_time)
        
        print(f"✅ Data loaded from: {file_name}")
        print(f"📈 Number of stocks selected: {df['Symbol'].nunique()}")
        print(f"📅 Data timeframe: {df.index.min()} to {df.index.max()}")
        print(f"📊 Total data points: {len(df):,} rows")
        
        # Initialize backtest engine with performance monitoring
        engine_start_time = time.time()
        print("🔧 Initializing optimized backtest engine...")
        backtester = BacktestEngine(config)
        backtester.selected_stocks = df['Symbol'].unique().tolist()
        performance_times['engine_initialization'] = performance_timer("Engine Initialization", engine_start_time)
        
        # Run backtest with performance monitoring
        backtest_start_time = time.time()
        print("🎯 Running optimized backtest with enhanced entry/exit logic...")
        print("   - Using collective 30-day EMA50 calculation")
        print("   - Implementing pending entry logic for precise entries")
        print("   - Applying Numba-optimized indicator calculations")
        
        trade_log, trades, equity_curve = backtester.run_backtest(df)
        performance_times['backtesting'] = performance_timer("Backtesting Execution", backtest_start_time)
        
        # Convert to DataFrames with performance monitoring
        conversion_start_time = time.time()
        trade_log_df = pd.DataFrame(trade_log)
        trades_df = pd.DataFrame(trades)
        performance_times['data_conversion'] = performance_timer("Data Conversion", conversion_start_time)
        
        # Calculate performance metrics with performance monitoring
        metrics_start_time = time.time()
        print("📊 Calculating performance metrics...")
        metrics = calculate_performance_metrics(trades_df, equity_curve)
        performance_report = generate_performance_report(metrics, trades_df)
        performance_times['metrics_calculation'] = performance_timer("Metrics Calculation", metrics_start_time)
        
        # Calculate total execution time
        total_time = time.time() - total_start_time
        performance_times['total_execution'] = total_time
        
        # Display results with performance summary
        print("\n" + "="*60)
        print("🎯 OPTIMIZED STRATEGY PERFORMANCE REPORT")
        print("="*60)
        print(performance_report)
        
        # Performance summary
        print("\n" + "⚡" + "="*58)
        print("PERFORMANCE OPTIMIZATION SUMMARY")
        print("="*60)
        print(f"✅ Total execution time: {total_time:.3f}s")
        print(f"🚀 Backtesting speed: {len(trades_df) / performance_times.get('backtesting', 1):.1f} trades/sec")
        print(f"📊 Data processing: {len(df) / performance_times.get('data_loading', 1):.0f} rows/sec")
        
        print("\n🔧 Optimizations Applied:")
        print("   ✅ Numba JIT compilation for indicators")
        print("   ✅ Memory-optimized data structures")
        print("   ✅ Enhanced entry/exit logic with pending orders")
        print("   ✅ Collective EMA50 calculation using 30-day data")
        print("   ✅ Vectorized operations where possible")
        
        # Detailed performance breakdown
        print("\n📈 Performance Breakdown:")
        for operation, time_taken in performance_times.items():
            percentage = (time_taken / total_time) * 100
            print(f"   {operation.replace('_', ' ').title()}: {time_taken:.3f}s ({percentage:.1f}%)")
        
        print("\n" + "💰" + "="*58)
        print("FINANCIAL SUMMARY")
        print("="*60)
        print(f"💵 Initial Capital: ₹{config['strategy']['base_capital']:,.2f}")
        print(f"💰 Final Capital: ₹{equity_curve[-1]:,.2f}")
        return_color = "🟢" if metrics['total_return'] >= 0 else "🔴"
        print(f"{return_color} Total Return: {metrics['total_return']:.2f}%")
        print(f"📊 Total Trades: {len(trades_df)}")
        if len(trades_df) > 0:
            win_rate = (trades_df['PnL'] > 0).mean() * 100
            print(f"🎯 Win Rate: {win_rate:.1f}%")
        
        # Save results with performance monitoring
        save_start_time = time.time()
        print("\n💾 Saving results...")
        save_results(trade_log, trades, equity_curve, 'results')
        performance_times['results_saving'] = performance_timer("Results Saving", save_start_time)
        
        # Display trade summary
        if not trades_df.empty:
            print("\n" + "📋" + "="*58)
            print("TRADE SUMMARY")
            print("="*60)
            print(f"📈 Long trades: {len(trades_df[trades_df['Side'] == 'Long'])}")
            print(f"📉 Short trades: {len(trades_df[trades_df['Side'] == 'Short'])}")
            print(f"💰 Profitable trades: {len(trades_df[trades_df['PnL'] > 0])}")
            print(f"💸 Loss trades: {len(trades_df[trades_df['PnL'] < 0])}")
            
            if len(trades_df) > 0:
                avg_profit = trades_df[trades_df['PnL'] > 0]['PnL'].mean()
                avg_loss = trades_df[trades_df['PnL'] < 0]['PnL'].mean()
                print(f"📊 Average profit: ₹{avg_profit:.2f}" if not pd.isna(avg_profit) else "📊 Average profit: N/A")
                print(f"📊 Average loss: ₹{avg_loss:.2f}" if not pd.isna(avg_loss) else "📊 Average loss: N/A")
            
            print("\n📋 First 5 Trades:")
            print(trades_df.head().to_string(index=False))
        else:
            print("\n⚠️  No trades were executed during the backtest period.")
            print("   This could be due to:")
            print("   - Strict entry conditions not being met")
            print("   - Insufficient data or market conditions")
            print("   - Configuration parameters too restrictive")
            
    except Exception as e:
        print(f"\n❌ Error during backtesting: {str(e)}")
        print("\n🔍 Detailed error information:")
        import traceback
        traceback.print_exc()
        
        # Show partial performance data if available
        if 'performance_times' in locals() and performance_times:
            print("\n⚡ Partial performance data:")
            for operation, time_taken in performance_times.items():
                print(f"   {operation.replace('_', ' ').title()}: {time_taken:.3f}s")

if __name__ == "__main__":
    print("🎯 Optimized Intraday EMA-RSI Strategy Backtester")
    print("🚀 Enhanced with Numba JIT compilation and performance monitoring")
    print("📊 Featuring collective EMA50 calculation and pending entry logic")
    print()
    main()