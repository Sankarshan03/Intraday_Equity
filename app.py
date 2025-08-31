import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add the current directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import from src package using absolute imports
from src.data_loader import prepare_data_for_backtesting
from src.backtesting import BacktestEngine
from src.performance import calculate_performance_metrics, generate_performance_report
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

def main():
    st.title("📈 Intraday EMA-RSI Strategy Backtester")
    st.markdown("""
    This application allows you to backtest the Intraday EMA-RSI trading strategy on NSE stock data.
    Upload your CSV files or use the sample data to analyze strategy performance.
    """)
    
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
    if st.sidebar.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest... This may take a few minutes"):
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
                
                # Prepare data based on user selection
                if use_existing_data and not uploaded_files:
                    data_folder = "stock_data"
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
                
                # Display data info
                st.info(f"Loaded data with {len(df)} rows and {df['Symbol'].nunique()} symbols")
                
                # Initialize and run backtest
                backtester = BacktestEngine(config)
                trade_log, trades, equity_curve = backtester.run_backtest(df)
                
                # Convert to DataFrames
                trade_log_df = pd.DataFrame(trade_log)
                trades_df = pd.DataFrame(trades)
                
                # Calculate performance metrics
                metrics = calculate_performance_metrics(trades_df, equity_curve)
                
                # Store results in session state
                st.session_state.results = {
                    'metrics': metrics,
                    'trade_log_df': trade_log_df,
                    'trades_df': trades_df,
                    'equity_curve': equity_curve,
                    'config': config
                }
                
                st.session_state.trades_df = trades_df
                st.session_state.equity_curve = equity_curve
                
                st.success("Backtest completed successfully!")
                
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
        
        # Display performance metrics
        st.header("📊 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{metrics['total_return']:.2f}%")
        with col2:
            st.metric("Win Rate", f"{metrics['win_rate']:.2f}%")
        with col3:
            st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        with col4:
            st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
        
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric("Number of Trades", len(trades_df))
        with col6:
            avg_profit = metrics['avg_profit'] if metrics['avg_profit'] else 0
            st.metric("Avg Profit", f"₹{avg_profit:.2f}")
        with col7:
            avg_loss = metrics['avg_loss'] if metrics['avg_loss'] else 0
            st.metric("Avg Loss", f"₹{avg_loss:.2f}")
        with col8:
            st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        
        # Equity curve chart
        st.header("📈 Equity Curve")
        
        equity_df = pd.DataFrame({
            'Day': range(len(equity_curve)),
            'Equity': equity_curve
        })
        
        fig_equity = px.line(
            equity_df, 
            x='Day', 
            y='Equity', 
            title='Portfolio Equity Curve',
            labels={'Equity': 'Portfolio Value (₹)', 'Day': 'Trading Period'}
        )
        fig_equity.update_layout(height=500)
        st.plotly_chart(fig_equity, use_container_width=True)
        
        # Trade analysis
        st.header("🔍 Trade Analysis")
        
        if not trades_df.empty:
            # Monthly returns heatmap
            trades_df['Exit_Time'] = pd.to_datetime(trades_df['Exit_Time'])
            trades_df['YearMonth'] = trades_df['Exit_Time'].dt.to_period('M').astype(str)
            monthly_pnl = trades_df.groupby('YearMonth')['PnL'].sum().reset_index()
            
            fig_heatmap = px.imshow(
                pd.pivot_table(
                    trades_df, 
                    values='PnL', 
                    index=trades_df['Exit_Time'].dt.year, 
                    columns=trades_df['Exit_Time'].dt.month, 
                    aggfunc='sum'
                ).fillna(0),
                title='Monthly P&L Heatmap',
                labels=dict(x="Month", y="Year", color="P&L")
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Trade duration vs returns scatter plot
            trades_df['Hold_Days'] = (trades_df['Exit_Time'] - trades_df['Entry_Time']).dt.total_seconds() / (60 * 60 * 24)
            fig_scatter = px.scatter(
                trades_df,
                x='Hold_Days',
                y='PnL',
                color='Return_Pct',
                title='Trade Duration vs Returns',
                labels={'Hold_Days': 'Hold Time (Days)', 'PnL': 'Profit/Loss (₹)'},
                hover_data=['Symbol', 'Entry_Price', 'Exit_Price']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
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
            
            ### Strategy Rules:
            - Selects top 10 stocks by turnover at 9:25 AM
            - Long when EMA(3) > EMA(10), RSI > 60, Close > EMA(50)
            - Short when EMA(3) < EMA(10), RSI < 30, Close < EMA(50)
            - 0.5% stop loss, 2% profit target
            - 0.75% trailing stop after 0.5% profit
            """
        )

if __name__ == "__main__":
    main()