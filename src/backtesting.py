import pandas as pd
import numpy as np
from src.indicators import calculate_ema, calculate_rsi

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.trade_log = []
        self.trades = []
        self.equity_curve = [config['strategy']['base_capital']]
        self.original_data = None
        self.selected_stocks = []
        
    def calculate_indicators(self, data):
        """
        Calculate all required indicators for the strategy with proper alignment
        EMA(3) and EMA(10) on 10-minute timeframe
        EMA(50) on 1-hour timeframe
        RSI(14) on 10-minute timeframe
        """
        # Store original data for reference
        self.original_data = data.copy()
        
        results = []
        
        for symbol, group in data.groupby('Symbol'):
            group = group.copy()
            
            # Create 10-minute OHLC data for EMA3, EMA10, and RSI14
            ohlc_10min = group.resample('10min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Create 1-hour OHLC data for EMA50
            ohlc_1hour = group.resample('1h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Calculate EMAs and RSI on their respective timeframes
            ohlc_10min['EMA3'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_fast'])
            ohlc_10min['EMA10'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_slow'])
            ohlc_10min['RSI14'] = calculate_rsi(ohlc_10min['Close'], self.config['indicators']['rsi_period'])
            
            # Calculate EMA50 on 1-hour data
            ohlc_1hour['EMA50_1H'] = calculate_ema(ohlc_1hour['Close'], self.config['indicators']['ema_trend'])
            
            # Merge EMA50 from 1-hour data to 10-minute data using forward fill
            # This aligns the 1-hour EMA50 with 10-minute candles
            ohlc_10min = ohlc_10min.join(ohlc_1hour[['EMA50_1H']], how='left')
            ohlc_10min['EMA50'] = ohlc_10min['EMA50_1H'].ffill()
            ohlc_10min.drop('EMA50_1H', axis=1, inplace=True)
            
            # Add symbol information
            ohlc_10min['Symbol'] = symbol
            
            results.append(ohlc_10min)
        
        # Combine all symbols
        if results:
            result_df = pd.concat(results)
        else:
            result_df = pd.DataFrame()
        
        return result_df
    
    def get_low_of_last_5_minutes(self, symbol, timestamp):
        """
        Get the low of the last 5 minutes of 1-minute candles for a specific symbol and timestamp
        """
        # Get the 1-minute data for this symbol
        symbol_data = self.original_data[self.original_data['Symbol'] == symbol]
        
        # Filter to the last 5 minutes before the current timestamp
        start_time = timestamp - pd.Timedelta(minutes=5)
        filtered_data = symbol_data[(symbol_data.index >= start_time) & (symbol_data.index <= timestamp)]
        
        if filtered_data.empty:
            return None
        
        # Return the minimum low price
        return filtered_data['Low'].min()
    
    def run_backtest(self, data):
        """
        Run backtest on the provided data
        """
        # Calculate indicators
        data_with_indicators = self.calculate_indicators(data)
        
        if data_with_indicators.empty:
            print("No data available for backtesting after indicator calculation.")
            return self.trade_log, self.trades, self.equity_curve
        
        # Get unique dates
        dates = data_with_indicators.index.normalize().unique()
        
        for date in dates:
            print(f"Processing date: {date}")
            
            # Process each day - filter to start trading from 9:30 AM onwards
            day_data = data_with_indicators[data_with_indicators.index.normalize() == date]
            
            # Filter trading data to start from 9:26 AM (post stock selection at 9:25 AM)
            trading_start_time = pd.Timestamp.combine(date.date(), pd.Timestamp("09:26:00").time())
            trading_data = day_data[day_data.index >= trading_start_time]
            
            if trading_data.empty:
                print(f"No trading data available for {date} after 9:26 AM")
                continue
            
            # Process each stock in the data (already filtered to top stocks)
            for symbol in trading_data['Symbol'].unique():
                stock_data = trading_data[trading_data['Symbol'] == symbol].copy()
                if not stock_data.empty:
                    self.process_stock(stock_data, symbol)
        
        return self.trade_log, self.trades, self.equity_curve
    
    def process_stock(self, stock_data, symbol):
        """
        Process trading signals for a single stock
        """
        # Initialize position variables
        position = None
        entry_price = 0
        entry_time = None
        stop_loss = 0
        profit_target = 0
        trailing_active = False
        max_profit = 0
        quantity = 0
        
        capital = self.equity_curve[-1]
        risk_amount = capital * self.config['strategy']['risk_per_trade']
        
        # Iterate through each 10-minute candle
        for idx, row in stock_data.iterrows():
            current_price = row['Close']
            
            # Check if we're in a position
            if position is None:
                # Check for long entry signal - with some tolerance
                ema_condition = row['EMA3'] > row['EMA10'] * 1.001  # 0.1% buffer
                rsi_condition = row['RSI14'] > self.config['indicators']['rsi_overbought'] - 5  # 5-point buffer
                trend_condition = row['Close'] > row['EMA50'] * 1.002  # 0.2% buffer
                
                if ema_condition and rsi_condition and trend_condition:
                    position = 'Long'
                    entry_price = row['High']  # Buy at high of entry candle
                    entry_time = idx
                    stop_loss = entry_price * (1 - self.config['strategy']['stop_loss'])
                    profit_target = entry_price * (1 + self.config['strategy']['profit_target'])
                    
                    # Calculate quantity based on risk
                    risk_per_share = entry_price - stop_loss
                    if risk_per_share > 0:
                        quantity = int(risk_amount / risk_per_share)
                    else:
                        quantity = 0
                    
                    # Ensure we have at least 1 share
                    quantity = max(1, quantity)
                    
                    # Log the trade entry
                    self.log_trade_entry(symbol, 'BUY', entry_price, quantity, entry_time)
                    print(f"Long entry: {symbol} at {entry_price}, quantity: {quantity}")
                
                # Check for short entry signal - with some tolerance
                ema_condition = row['EMA3'] < row['EMA10'] * 0.999  # 0.1% buffer
                rsi_condition = row['RSI14'] < self.config['indicators']['rsi_oversold'] + 5  # 5-point buffer
                trend_condition = row['Close'] < row['EMA50'] * 0.998  # 0.2% buffer
                
                if ema_condition and rsi_condition and trend_condition:
                    position = 'Short'
                    # Get low of last 5 minutes of 1-minute candles for short entry
                    entry_price = self.get_low_of_last_5_minutes(symbol, idx)
                    
                    if entry_price is None:
                        # Fallback to current low if no 1-minute data available
                        entry_price = row['Low']
                    
                    entry_time = idx
                    stop_loss = entry_price * (1 + self.config['strategy']['stop_loss'])
                    profit_target = entry_price * (1 - self.config['strategy']['profit_target'])
                    
                    # Calculate quantity based on risk
                    risk_per_share = stop_loss - entry_price
                    if risk_per_share > 0:
                        quantity = int(risk_amount / risk_per_share)
                    else:
                        quantity = 0
                    
                    # Ensure we have at least 1 share
                    quantity = max(1, quantity)
                    
                    # Log the trade entry
                    self.log_trade_entry(symbol, 'SELL', entry_price, quantity, entry_time)
                    print(f"Short entry: {symbol} at {entry_price}, quantity: {quantity}")
            
            else:
                # We're in a position, check exit conditions
                if position == 'Long':
                    exit_signal, exit_price, exit_reason = self.check_long_exit(
                        current_price, entry_price, stop_loss, profit_target, trailing_active, max_profit
                    )
                    
                    if exit_signal:
                        pnl = (exit_price - entry_price) * quantity
                        self.log_trade_exit(symbol, 'SELL', exit_price, quantity, idx, pnl, exit_reason)
                        
                        # Update equity curve
                        self.equity_curve.append(self.equity_curve[-1] + pnl)
                        
                        print(f"Long exit: {symbol} at {exit_price}, PnL: {pnl}, Reason: {exit_reason}")
                        
                        # Reset position
                        position = None
                        trailing_active = False
                    
                    # Update trailing stop if in profit
                    elif current_price > entry_price * (1 + self.config['strategy']['profit_trail_start']):
                        trailing_active = True
                        if current_price > max_profit:
                            max_profit = current_price
                            stop_loss = max_profit * (1 - self.config['strategy']['trailing_stop'])
                
                elif position == 'Short':
                    exit_signal, exit_price, exit_reason = self.check_short_exit(
                        current_price, entry_price, stop_loss, profit_target, trailing_active, max_profit
                    )
                    
                    if exit_signal:
                        pnl = (entry_price - exit_price) * quantity
                        self.log_trade_exit(symbol, 'BUY', exit_price, quantity, idx, pnl, exit_reason)
                        
                        # Update equity curve
                        self.equity_curve.append(self.equity_curve[-1] + pnl)
                        
                        print(f"Short exit: {symbol} at {exit_price}, PnL: {pnl}, Reason: {exit_reason}")
                        
                        # Reset position
                        position = None
                        trailing_active = False
                    
                    # Update trailing stop if in profit
                    elif current_price < entry_price * (1 - self.config['strategy']['profit_trail_start']):
                        trailing_active = True
                        if current_price < max_profit:
                            max_profit = current_price
                            stop_loss = max_profit * (1 + self.config['strategy']['trailing_stop'])
    
    def check_long_exit(self, current_price, entry_price, stop_loss, profit_target, trailing_active, max_profit):
        """
        Check exit conditions for long position
        """
        if current_price <= stop_loss:
            return True, stop_loss, 'Stop Loss'
        elif current_price >= profit_target and not trailing_active:
            return True, profit_target, 'Profit Target'
        elif trailing_active and current_price <= stop_loss:
            return True, stop_loss, 'Trailing Stop'
        
        return False, 0, ''
    
    def check_short_exit(self, current_price, entry_price, stop_loss, profit_target, trailing_active, max_profit):
        """
        Check exit conditions for short position
        """
        if current_price >= stop_loss:
            return True, stop_loss, 'Stop Loss'
        elif current_price <= profit_target and not trailing_active:
            return True, profit_target, 'Profit Target'
        elif trailing_active and current_price >= stop_loss:
            return True, stop_loss, 'Trailing Stop'
        
        return False, 0, ''
    
    def log_trade_entry(self, symbol, action, price, quantity, timestamp):
        """
        Log trade entry
        """
        trade = {
            'Date': timestamp.date(),
            'Time': timestamp.time(),
            'Symbol': symbol,
            'Action': action,
            'Price': price,
            'Quantity': quantity,
            'Value': price * quantity
        }
        
        self.trade_log.append(trade)
    
    def log_trade_exit(self, symbol, action, price, quantity, timestamp, pnl, reason):
        """
        Log trade exit
        """
        trade = {
            'Date': timestamp.date(),
            'Time': timestamp.time(),
            'Symbol': symbol,
            'Action': action,
            'Price': price,
            'Quantity': quantity,
            'Value': price * quantity,
            'PnL': pnl,
            'Exit_Reason': reason
        }
        
        self.trade_log.append(trade)
        
        # Also add to trades summary
        self.trades.append({
            'Symbol': symbol,
            'Entry_Action': 'BUY' if action == 'SELL' else 'SELL',
            'Exit_Action': action,
            'Entry_Price': self.trade_log[-2]['Price'],  # Previous trade log entry
            'Exit_Price': price,
            'Quantity': quantity,
            'PnL': pnl,
            'Return_Pct': (price - self.trade_log[-2]['Price']) / self.trade_log[-2]['Price'] * 100,
            'Exit_Reason': reason,
            'Entry_Time': pd.Timestamp.combine(self.trade_log[-2]['Date'], self.trade_log[-2]['Time']),
            'Exit_Time': timestamp
        })