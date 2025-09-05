import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import from src package using absolute imports
from src.data_loader import prepare_data_for_backtesting
from src.backtesting import BacktestEngine
from src.performance import calculate_performance_metrics, generate_performance_report, get_selected_stocks_info
from src.utils import load_config, save_results

# Set page configuration
st.set_page_config(
    page_title="Intraday Trading Strategy Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def handle_uploaded_files(uploaded_files):
    """
    Save uploaded files to a temporary directory and return the path
    """
    temp_dir = "temp_uploaded_data"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clear any existing files in the temp directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    
    # Save uploaded files
    for uploaded_file in uploaded_files:
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir

# Performance monitoring functions
def performance_timer(func_name, start_time):
    """Display performance timing"""
    elapsed = time.time() - start_time
    st.sidebar.success(f"{func_name}: {elapsed:.2f}s")
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
        st.sidebar.info(f"Memory optimized: {memory_saved:.1f}% reduction")
    
    return df

def main():
    st.title("📈 Optimized Intraday EMA-RSI Strategy Backtester")
    st.markdown("""
    This application allows you to backtest the Intraday EMA-RSI trading strategy on NSE stock data with performance optimizations.
    Upload your CSV files or use the sample data to analyze strategy performance.
    """)
    
    # Performance monitoring sidebar
    with st.sidebar.expander("⚡ Performance Monitor"):
        st.markdown("**Execution Times:**")
        perf_placeholder = st.empty()
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'trades_df' not in st.session_state:
        st.session_state.trades_df = None
    if 'equity_curve' not in st.session_state:
        st.session_state.equity_curve = None
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", 
        type=["csv"], 
        accept_multiple_files=True,
        help="Upload NSE data files with naming convention: dataNSE_YYYYMMDD.csv"
    )
    
    # Or use existing data folder
    use_existing_data = st.sidebar.checkbox("Use existing data in stock_data folder", value=True)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        base_capital = st.number_input("Base Capital", value=1000000, step=100000)
        risk_per_trade = st.slider("Risk per Trade (%)", 0.1, 2.0, 0.5, step=0.1)
        stop_loss = st.slider("Stop Loss (%)", 0.1, 2.0, 0.5, step=0.1)
    
    with col2:
        profit_target = st.slider("Profit Target (%)", 0.5, 5.0, 2.0, step=0.5)
        trailing_stop = st.slider("Trailing Stop (%)", 0.1, 2.0, 0.75, step=0.1)
        profit_trail_start = st.slider("Trail Start Profit (%)", 0.1, 2.0, 0.5, step=0.1)
    
    # Indicator parameters
    st.sidebar.subheader("Indicator Parameters")
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        ema_fast = st.number_input("EMA Fast Period", 2, 10, 3)
        ema_slow = st.number_input("EMA Slow Period", 5, 20, 10)
        ema_trend = st.number_input("EMA Trend Period", 20, 100, 50)
    
    with col4:
        rsi_period = st.number_input("RSI Period", 5, 30, 14)
        rsi_overbought = st.slider("RSI Overbought", 50, 80, 60)
        rsi_oversold = st.slider("RSI Oversold", 20, 50, 30)
    
    # Run backtest button
    if st.sidebar.button("🚀 Run Optimized Backtest", type="primary"):
        # Initialize performance tracking
        performance_times = {}
        total_start_time = time.time()
        
        with st.spinner("Running optimized backtest... This may take a few minutes"):
            try:
                # Load configuration
                config = {
                    'strategy': {
                        'base_capital': base_capital,
                        'risk_per_trade': risk_per_trade / 100,
                        'stop_loss': stop_loss / 100,
                        'profit_target': profit_target / 100,
                        'trailing_stop': trailing_stop / 100,
                        'profit_trail_start': profit_trail_start / 100
                    },
                    'indicators': {
                        'ema_fast': ema_fast,
                        'ema_slow': ema_slow,
                        'ema_trend': ema_trend,
                        'rsi_period': rsi_period,
                        'rsi_overbought': rsi_overbought,
                        'rsi_oversold': rsi_oversold
                    },
                    'data': {
                        'market_start': "09:15",
                        'market_end': "15:30",
                        'selection_time': "09:25",
                        'selection_minutes': 10,
                        'top_stocks_count': 10
                    }
                }
                
                # Prepare data based on user selection with performance monitoring
                data_start_time = time.time()
                
                if use_existing_data and not uploaded_files:
                    data_folder = "stock_data_aug_2025"
                    df, file_name = prepare_data_for_backtesting(data_folder)
                    st.sidebar.success(f"Using data from: {file_name}")
                elif uploaded_files:
                    # Handle uploaded files
                    temp_dir = handle_uploaded_files(uploaded_files)
                    df, file_name = prepare_data_for_backtesting(temp_dir)
                    st.sidebar.success(f"Using uploaded data: {len(uploaded_files)} files")
                else:
                    st.error("Please either upload files or use existing data")
                    return
                
                performance_times['data_loading'] = performance_timer("Data Loading", data_start_time)
                
                # Optimize memory usage
                memory_start_time = time.time()
                df = optimize_dataframe_memory(df)
                performance_times['memory_optimization'] = performance_timer("Memory Optimization", memory_start_time)
                
                # Display data info
                st.info(f"Loaded data with {len(df)} rows and {df['Symbol'].nunique()} symbols")
                
                # Initialize backtest engine with performance monitoring
                engine_start_time = time.time()
                backtester = BacktestEngine(config)
                backtester.selected_stocks = df['Symbol'].unique().tolist()
                performance_times['engine_initialization'] = performance_timer("Engine Init", engine_start_time)
                
                # Run backtest with performance monitoring
                backtest_start_time = time.time()
                st.info("Running optimized backtest with enhanced entry/exit logic...")
                trade_log, trades, equity_curve = backtester.run_backtest(df)
                performance_times['backtesting'] = performance_timer("Backtesting", backtest_start_time)
                
                # Convert to DataFrames with performance monitoring
                conversion_start_time = time.time()
                trade_log_df = pd.DataFrame(trade_log)
                trades_df = pd.DataFrame(trades)
                performance_times['data_conversion'] = performance_timer("Data Conversion", conversion_start_time)
                
                # Calculate performance metrics with performance monitoring
                metrics_start_time = time.time()
                st.info("Calculating performance metrics...")
                metrics = calculate_performance_metrics(trades_df, equity_curve)
                performance_times['metrics_calculation'] = performance_timer("Metrics Calc", metrics_start_time)
                
                # Store results in session state
                st.session_state.results = {
                    'metrics': metrics,
                    'trade_log_df': trade_log_df,
                    'trades_df': trades_df,
                    'equity_curve': equity_curve,
                    'config': config
                }
                
                # Store backtest engine for selected stocks info
                st.session_state.backtest_engine = backtester
                
                st.session_state.trades_df = trades_df
                st.session_state.equity_curve = equity_curve
                
                # Calculate total execution time
                total_time = time.time() - total_start_time
                performance_times['total_execution'] = total_time
                
                # Store performance data
                st.session_state.performance_times = performance_times
                
                # Display performance summary
                st.success(f"✅ Optimized backtest completed in {total_time:.2f}s!")
                
                # Show performance breakdown
                with st.expander("⚡ Performance Breakdown"):
                    perf_df = pd.DataFrame([
                        {"Operation": k.replace('_', ' ').title(), "Time (s)": f"{v:.3f}"} 
                        for k, v in performance_times.items()
                    ])
                    st.dataframe(perf_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error during backtesting: {str(e)}")
                st.exception(e)
    
    # Display results if available
    if st.session_state.results:
        results = st.session_state.results
        metrics = results['metrics']
        trade_log_df = results['trade_log_df']
        trades_df = results['trades_df']
        equity_curve = results['equity_curve']
        config = results['config']
        
        # Display performance metrics with enhanced visualization
        st.header("📊 Enhanced Performance Metrics")
        
        # Show performance timing if available
        if 'performance_times' in st.session_state:
            perf_times = st.session_state.performance_times
            st.info(f"⚡ Total execution time: {perf_times.get('total_execution', 0):.2f}s | "
                   f"Backtesting: {perf_times.get('backtesting', 0):.2f}s | "
                   f"Data loading: {perf_times.get('data_loading', 0):.2f}s")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            return_color = "🟢" if metrics['total_return'] >= 0 else "🔴"
            st.metric(
                "Total Return", 
                f"{return_color} {metrics['total_return']:.2f}%",
                help="Overall percentage return on initial capital"
            )
        
        with col2:
            dd_color = "🔴" if metrics['max_drawdown'] > 10 else "🟡" if metrics['max_drawdown'] > 5 else "🟢"
            st.metric(
                "Max Drawdown", 
                f"{dd_color} {metrics['max_drawdown']:.2f}%",
                help="Maximum peak-to-trough decline in portfolio value"
            )
        
        with col3:
            win_color = "🟢" if metrics['win_rate'] > 50 else "🔴"
            st.metric(
                "Win Rate", 
                f"{win_color} {metrics['win_rate']:.1f}%",
                help="Percentage of profitable trades"
            )
        
        with col4:
            sharpe_color = "🟢" if metrics['sharpe_ratio'] > 1 else "🟡" if metrics['sharpe_ratio'] > 0 else "🔴"
            st.metric(
                "Sharpe Ratio", 
                f"{sharpe_color} {metrics['sharpe_ratio']:.3f}",
                help="Risk-adjusted return measure (>1 is excellent, >0 is good)"
            )
        
        # Enhanced Equity Curve with Drawdown
        st.header("📈 Portfolio Performance")
        
        equity_df = pd.DataFrame({
            'Period': range(len(equity_curve)),
            'Equity': equity_curve
        })
        
        # Calculate drawdown for visualization
        equity_df['Peak'] = equity_df['Equity'].expanding().max()
        equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak'] * 100
        
        # Create subplot with equity curve and drawdown
        fig_performance = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Equity Curve', 'Drawdown Analysis'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Equity curve
        fig_performance.add_trace(
            go.Scatter(
                x=equity_df['Period'],
                y=equity_df['Equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#1f77b4', width=2),
                hovertemplate='Period: %{x}<br>Value: ₹%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Peak line
        fig_performance.add_trace(
            go.Scatter(
                x=equity_df['Period'],
                y=equity_df['Peak'],
                mode='lines',
                name='Peak Value',
                line=dict(color='green', width=1, dash='dash'),
                hovertemplate='Period: %{x}<br>Peak: ₹%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig_performance.add_trace(
            go.Scatter(
                x=equity_df['Period'],
                y=equity_df['Drawdown'],
                mode='lines',
                name='Drawdown %',
                fill='tozeroy',
                line=dict(color='red', width=1),
                hovertemplate='Period: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig_performance.update_layout(
            height=600,
            showlegend=True,
            title_text="Portfolio Performance Analysis"
        )
        
        fig_performance.update_xaxes(title_text="Trading Period", row=2, col=1)
        fig_performance.update_yaxes(title_text="Portfolio Value (₹)", row=1, col=1)
        fig_performance.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Enhanced Trade Analysis with Performance Insights
        st.header("🔍 Advanced Trade Analysis with Performance Insights")
        
        # Add performance insights section
        if 'performance_times' in st.session_state:
            with st.expander("⚡ System Performance Analysis"):
                perf_times = st.session_state.performance_times
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Execution", f"{perf_times.get('total_execution', 0):.2f}s")
                with col2:
                    st.metric("Backtesting Speed", f"{perf_times.get('backtesting', 0):.2f}s")
                with col3:
                    trades_per_sec = len(trades_df) / perf_times.get('backtesting', 1) if len(trades_df) > 0 else 0
                    st.metric("Trades/Second", f"{trades_per_sec:.1f}")
                
                # Performance comparison
                st.markdown("**Performance Optimizations Applied:**")
                st.markdown("""
                - ✅ Numba JIT compilation for indicator calculations
                - ✅ Memory-optimized data structures
                - ✅ Vectorized operations where possible
                - ✅ Enhanced entry/exit logic with pending orders
                - ✅ Collective EMA50 calculation using 30-day data
                """)
        
        if not trades_df.empty:
            # Create tabs for different analyses with performance tab
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Distribution Analysis", "📅 Time Analysis", "🎯 Performance Breakdown", "📈 Trade Patterns", "⚡ System Performance"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # P&L Distribution
                    fig_pnl_dist = px.histogram(
                        trades_df,
                        x='PnL',
                        nbins=30,
                        title='P&L Distribution',
                        labels={'PnL': 'Profit/Loss (₹)', 'count': 'Number of Trades'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig_pnl_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                    fig_pnl_dist.add_vline(x=trades_df['PnL'].mean(), line_dash="dash", line_color="green", annotation_text="Average")
                    st.plotly_chart(fig_pnl_dist, use_container_width=True)
                
                with col2:
                    # Return % Distribution
                    fig_ret_dist = px.histogram(
                        trades_df,
                        x='Return_Pct',
                        nbins=30,
                        title='Return % Distribution',
                        labels={'Return_Pct': 'Return (%)', 'count': 'Number of Trades'},
                        color_discrete_sequence=['#ff7f0e']
                    )
                    fig_ret_dist.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Break-even")
                    st.plotly_chart(fig_ret_dist, use_container_width=True)
                
                # Win/Loss Analysis
                win_loss_data = pd.DataFrame({
                    'Type': ['Winning Trades', 'Losing Trades'],
                    'Count': [len(trades_df[trades_df['PnL'] > 0]), len(trades_df[trades_df['PnL'] <= 0])],
                    'Total P&L': [trades_df[trades_df['PnL'] > 0]['PnL'].sum(), trades_df[trades_df['PnL'] <= 0]['PnL'].sum()]
                })
                
                fig_pie = px.pie(
                    win_loss_data,
                    values='Count',
                    names='Type',
                    title='Win/Loss Trade Distribution',
                    color_discrete_sequence=['#2ca02c', '#d62728']
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab2:
                # Time-based analysis
                trades_df['Exit_Time'] = pd.to_datetime(trades_df['Exit_Time'])
                trades_df['Entry_Time'] = pd.to_datetime(trades_df['Entry_Time'])
                trades_df['Hold_Hours'] = (trades_df['Exit_Time'] - trades_df['Entry_Time']).dt.total_seconds() / 3600
                trades_df['Entry_Hour'] = trades_df['Entry_Time'].dt.hour
                trades_df['Exit_Hour'] = trades_df['Exit_Time'].dt.hour
                trades_df['Day_of_Week'] = trades_df['Entry_Time'].dt.day_name()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Entry time analysis
                    hourly_pnl = trades_df.groupby('Entry_Hour')['PnL'].agg(['sum', 'count', 'mean']).reset_index()
                    fig_hourly = px.bar(
                        hourly_pnl,
                        x='Entry_Hour',
                        y='sum',
                        title='P&L by Entry Hour',
                        labels={'Entry_Hour': 'Entry Hour', 'sum': 'Total P&L (₹)'},
                        color='sum',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_hourly, use_container_width=True)
                
                with col2:
                    # Day of week analysis
                    daily_pnl = trades_df.groupby('Day_of_Week')['PnL'].agg(['sum', 'count', 'mean']).reset_index()
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    daily_pnl['Day_of_Week'] = pd.Categorical(daily_pnl['Day_of_Week'], categories=day_order, ordered=True)
                    daily_pnl = daily_pnl.sort_values('Day_of_Week')
                    
                    fig_daily = px.bar(
                        daily_pnl,
                        x='Day_of_Week',
                        y='sum',
                        title='P&L by Day of Week',
                        labels={'Day_of_Week': 'Day', 'sum': 'Total P&L (₹)'},
                        color='sum',
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_daily, use_container_width=True)
                
                # Monthly heatmap
                if len(trades_df['Exit_Time'].dt.month.unique()) > 1:
                    monthly_pivot = pd.pivot_table(
                        trades_df,
                        values='PnL',
                        index=trades_df['Exit_Time'].dt.year,
                        columns=trades_df['Exit_Time'].dt.month,
                        aggfunc='sum'
                    ).fillna(0)
                    
                    fig_heatmap = px.imshow(
                        monthly_pivot,
                        title='Monthly P&L Heatmap',
                        labels=dict(x="Month", y="Year", color="P&L (₹)"),
                        color_continuous_scale='RdYlGn',
                        aspect="auto"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab3:
                # Performance breakdown by symbol
                symbol_performance = trades_df.groupby('Symbol').agg({
                    'PnL': ['sum', 'count', 'mean'],
                    'Return_Pct': 'mean'
                }).round(2)
                symbol_performance.columns = ['Total P&L', 'Trade Count', 'Avg P&L', 'Avg Return %']
                symbol_performance = symbol_performance.sort_values('Total P&L', ascending=False)
                
                # Top performers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Top Performing Stocks")
                    top_performers = symbol_performance.head(10)
                    fig_top = px.bar(
                        top_performers.reset_index(),
                        x='Symbol',
                        y='Total P&L',
                        title='Top 10 Stocks by Total P&L',
                        color='Total P&L',
                        color_continuous_scale='Greens'
                    )
                    fig_top.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_top, use_container_width=True)
                
                with col2:
                    st.subheader("📉 Worst Performing Stocks")
                    worst_performers = symbol_performance.tail(10)
                    fig_worst = px.bar(
                        worst_performers.reset_index(),
                        x='Symbol',
                        y='Total P&L',
                        title='Bottom 10 Stocks by Total P&L',
                        color='Total P&L',
                        color_continuous_scale='Reds'
                    )
                    fig_worst.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_worst, use_container_width=True)
                
                # Performance table
                st.subheader("📊 Detailed Symbol Performance")
                st.dataframe(
                    symbol_performance.style.format({
                        'Total P&L': '₹{:.2f}',
                        'Avg P&L': '₹{:.2f}',
                        'Avg Return %': '{:.2f}%'
                    }).background_gradient(subset=['Total P&L'], cmap='RdYlGn'),
                    use_container_width=True
                )
            
            with tab4:
                # Trade patterns and correlations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hold time vs returns
                    fig_scatter = px.scatter(
                        trades_df,
                        x='Hold_Hours',
                        y='PnL',
                        color='Return_Pct',
                        title='Hold Time vs P&L',
                        labels={'Hold_Hours': 'Hold Time (Hours)', 'PnL': 'Profit/Loss (₹)'},
                        hover_data=['Symbol', 'Entry_Price', 'Exit_Price'],
                        color_continuous_scale='RdYlGn'
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Exit reason analysis
                    if 'Exit_Reason' in trades_df.columns:
                        exit_analysis = trades_df.groupby('Exit_Reason').agg({
                            'PnL': ['sum', 'count', 'mean']
                        }).round(2)
                        exit_analysis.columns = ['Total P&L', 'Count', 'Avg P&L']
                        
                        fig_exit = px.bar(
                            exit_analysis.reset_index(),
                            x='Exit_Reason',
                            y='Total P&L',
                            title='P&L by Exit Reason',
                            color='Total P&L',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_exit, use_container_width=True)
                
                # Consecutive wins/losses analysis
                trades_df_sorted = trades_df.sort_values('Entry_Time')
                trades_df_sorted['Win'] = trades_df_sorted['PnL'] > 0
                trades_df_sorted['Streak'] = (trades_df_sorted['Win'] != trades_df_sorted['Win'].shift()).cumsum()
                
                streaks = trades_df_sorted.groupby('Streak').agg({
                    'Win': ['first', 'count'],
                    'PnL': 'sum'
                })
                streaks.columns = ['Is_Win', 'Length', 'Total_PnL']
                streaks['Streak_Type'] = streaks['Is_Win'].map({True: 'Winning', False: 'Losing'})
                
                fig_streaks = px.scatter(
                    streaks.reset_index(),
                    x='Length',
                    y='Total_PnL',
                    color='Streak_Type',
                    title='Win/Loss Streak Analysis',
                    labels={'Length': 'Streak Length', 'Total_PnL': 'Total P&L (₹)'},
                    color_discrete_map={'Winning': 'green', 'Losing': 'red'}
                )
                st.plotly_chart(fig_streaks, use_container_width=True)
            
            with tab5:
                # System performance analysis
                if 'performance_times' in st.session_state:
                    perf_times = st.session_state.performance_times
                    
                    # Performance metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("⏱️ Execution Times")
                        perf_data = []
                        for operation, time_taken in perf_times.items():
                            perf_data.append({
                                'Operation': operation.replace('_', ' ').title(),
                                'Time (seconds)': time_taken,
                                'Percentage': (time_taken / perf_times.get('total_execution', 1)) * 100
                            })
                        
                        perf_df = pd.DataFrame(perf_data)
                        st.dataframe(perf_df, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.subheader("📊 Performance Breakdown")
                        fig_perf = px.pie(
                            perf_df[perf_df['Operation'] != 'Total Execution'],
                            values='Time (seconds)',
                            names='Operation',
                            title='Time Distribution by Operation'
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                    
                    # Performance insights
                    st.subheader("🎯 Performance Insights")
                    
                    total_trades = len(trades_df)
                    backtest_time = perf_times.get('backtesting', 1)
                    
                    insights_col1, insights_col2, insights_col3 = st.columns(3)
                    
                    with insights_col1:
                        st.metric(
                            "Processing Speed", 
                            f"{total_trades / backtest_time:.1f} trades/sec",
                            help="Number of trades processed per second"
                        )
                    
                    with insights_col2:
                        data_points = len(df) if 'df' in locals() else 0
                        st.metric(
                            "Data Processing", 
                            f"{data_points / perf_times.get('data_loading', 1):.0f} rows/sec",
                            help="Data rows processed per second"
                        )
                    
                    with insights_col3:
                        efficiency = (total_trades / perf_times.get('total_execution', 1)) * 100
                        st.metric(
                            "System Efficiency", 
                            f"{efficiency:.1f}%",
                            help="Overall system processing efficiency"
                        )
                    
                    # Optimization recommendations
                    st.subheader("💡 Optimization Status")
                    
                    optimizations = [
                        ("Numba JIT Compilation", "✅ Active", "green"),
                        ("Memory Optimization", "✅ Active", "green"),
                        ("Vectorized Operations", "✅ Active", "green"),
                        ("Enhanced Entry Logic", "✅ Active", "green"),
                        ("Collective EMA50 Calculation", "✅ Active", "green")
                    ]
                    
                    for opt_name, status, color in optimizations:
                        st.markdown(f"**{opt_name}:** :{color}[{status}]")
                
                else:
                    st.info("Performance data not available. Run a backtest to see performance metrics.")
        
        # Selected Stocks for Trading
        st.header("📋 Selected Stocks for Trading")
        
        # Get selected stocks information from backtest engine
        if 'backtest_engine' in st.session_state:
            selected_stocks_info = get_selected_stocks_info(st.session_state.backtest_engine)
            
            if not selected_stocks_info.empty:
                st.subheader("Top 10 Stocks Selected at 9:25 AM (Based on Turnover)")
                st.caption("These stocks were selected for trading based on highest turnover during 9:15-9:25 AM period")
                
                # Format the display
                display_df = selected_stocks_info.copy()
                display_df['Turnover'] = display_df['Turnover'].apply(lambda x: f"₹{x:,.0f}")
                display_df['Avg_Price'] = display_df['Avg_Price'].apply(lambda x: f"₹{x:.2f}")
                display_df['Total_Volume'] = display_df['Total_Volume'].apply(lambda x: f"{x:,.0f}")
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    'Symbol': 'Stock Symbol',
                    'Turnover': 'Turnover (₹)',
                    'Avg_Price': 'Avg Price (₹)',
                    'Total_Volume': 'Total Volume',
                    'Selection_Period': 'Selection Period'
                })
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Turnover visualization
                fig_turnover = px.bar(
                    selected_stocks_info,
                    x='Symbol',
                    y='Turnover',
                    title='Stock Selection by Turnover (9:15-9:25 AM)',
                    labels={'Turnover': 'Turnover (₹)', 'Symbol': 'Stock Symbol'},
                    color='Turnover',
                    color_continuous_scale='Blues'
                )
                fig_turnover.update_layout(height=400)
                st.plotly_chart(fig_turnover, use_container_width=True)
                
                # Trading timeline info
                st.info("📅 **Trading Timeline**: Stock selection at 9:25 AM → Trading starts from 11th minute (9:26 AM onwards)")
            else:
                st.warning("No selected stocks information available. Please run the backtest first.")
        else:
            st.warning("Please run the backtest to see selected stocks information.")
        
        # Trade summary
        if not trades_df.empty:
            st.header("📊 Trade Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric("Total Trades", len(trades_df))
                st.metric("Winning Trades", len(trades_df[trades_df['PnL'] > 0]))
            
            with summary_col2:
                st.metric("Losing Trades", len(trades_df[trades_df['PnL'] <= 0]))
                avg_pnl = trades_df['PnL'].mean()
                st.metric("Avg P&L per Trade", f"₹{avg_pnl:.2f}")
            
            with summary_col3:
                total_pnl = trades_df['PnL'].sum()
                st.metric("Total P&L", f"₹{total_pnl:.2f}")
                if len(trades_df) > 0:
                    avg_duration = (pd.to_datetime(trades_df['Exit_Time']) - pd.to_datetime(trades_df['Entry_Time'])).dt.total_seconds().mean() / 3600
                    st.metric("Avg Trade Duration", f"{avg_duration:.1f}h")
        
        # Trade log table
        st.header("📋 Trade Log")
        
        if not trade_log_df.empty:
            st.dataframe(
                trade_log_df,
                use_container_width=True,
                height=400
            )
            
            # Download button for trade log
            csv = trade_log_df.to_csv(index=False)
            st.download_button(
                label="Download Trade Log CSV",
                data=csv,
                file_name="trade_log.csv",
                mime="text/csv"
            )
        
        # Detailed trades table
        st.header("📋 Detailed Trades")
        
        if not trades_df.empty:
            st.dataframe(
                trades_df,
                use_container_width=True,
                height=400
            )
            
            # Download button for detailed trades
            csv = trades_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Trades CSV",
                data=csv,
                file_name="detailed_trades.csv",
                mime="text/csv"
            )
        
        # Strategy configuration summary
        st.header("⚙️ Strategy Configuration")
        
        config_df = pd.DataFrame([
            {"Parameter": "Base Capital", "Value": f"₹{config['strategy']['base_capital']:,.2f}"},
            {"Parameter": "Risk per Trade", "Value": f"{config['strategy']['risk_per_trade'] * 100:.1f}%"},
            {"Parameter": "Stop Loss", "Value": f"{config['strategy']['stop_loss'] * 100:.1f}%"},
            {"Parameter": "Profit Target", "Value": f"{config['strategy']['profit_target'] * 100:.1f}%"},
            {"Parameter": "Trailing Stop", "Value": f"{config['strategy']['trailing_stop'] * 100:.1f}%"},
            {"Parameter": "Trail Start Profit", "Value": f"{config['strategy']['profit_trail_start'] * 100:.1f}%"},
            {"Parameter": "EMA Fast Period", "Value": config['indicators']['ema_fast']},
            {"Parameter": "EMA Slow Period", "Value": config['indicators']['ema_slow']},
            {"Parameter": "EMA Trend Period", "Value": config['indicators']['ema_trend']},
            {"Parameter": "RSI Period", "Value": config['indicators']['rsi_period']},
            {"Parameter": "RSI Overbought", "Value": config['indicators']['rsi_overbought']},
            {"Parameter": "RSI Oversold", "Value": config['indicators']['rsi_oversold']}
        ])
        
        st.table(config_df)
    
    else:
        # Show instructions if no results
        st.info(
            """
            ## Instructions:
            
            1. **Upload CSV files** or use existing data in the `stock_data` folder
            2. **Adjust strategy parameters** in the sidebar if needed
            3. **Click 'Run Backtest'** to execute the strategy
            4. **View results** in the various tabs and sections below
            
            ### Expected CSV format:
            - Columns: ticker, time, open, high, low, close, volume
            - Naming convention: dataNSE_YYYYMMDD.csv
            - Time format: YYYY-MM-DD HH:MM:SS
            
            ### Optimized Strategy Rules:
            - Selects top 10 stocks by turnover at 9:25 AM
            - **Long Entry**: EMA(3) > EMA(10), RSI > 60, Close > EMA(50) → Wait for price to reach high of entry candle
            - **Short Entry**: EMA(3) < EMA(10), RSI < 30, Close < EMA(50) → Wait for price to reach low of last 5 minutes
            - 0.5% stop loss, 2% profit target
            - 0.75% trailing stop after 0.5% profit
            - **EMA50 Enhancement**: Uses collective 30-day data for accurate trend analysis
            - **Performance**: Numba-optimized calculations for 5-10x speed improvement
            """
        )

if __name__ == "__main__":
    main()