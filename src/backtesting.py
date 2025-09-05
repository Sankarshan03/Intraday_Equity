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
        self.collective_ema50_data = None
        
    def load_collective_data_for_ema50(self, data_folder="stock_data_aug_2025"):
        """
        Load all 30 days of data from stock_data folder for EMA50 calculation
        """
        from src.data_loader import load_data_from_folder, preprocess_data, filter_market_hours
        
        try:
            # Load complete 30 days dataset
            full_df = load_data_from_folder(data_folder)
            full_df = preprocess_data(full_df)
            full_df = filter_market_hours(full_df)
            
            # Create 1-hour OHLC data for all symbols across all 30 days
            collective_1hour_data = []
            
            for symbol, group in full_df.groupby('Symbol'):
                group = group.copy().sort_index()
                
                # Create 1-hour OHLC data
                ohlc_1hour = group.resample('1h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                ohlc_1hour['Symbol'] = symbol
                collective_1hour_data.append(ohlc_1hour)
            
            if collective_1hour_data:
                all_1hour_data = pd.concat(collective_1hour_data)
                
                # Calculate EMA50 for each symbol using complete 30 days data
                ema50_results = {}
                for symbol, group in all_1hour_data.groupby('Symbol'):
                    group = group.sort_index()
                    group['EMA50_1H'] = calculate_ema(group['Close'], self.config['indicators']['ema_trend'])
                    ema50_results[symbol] = group[['EMA50_1H']]
                
                self.collective_ema50_data = ema50_results
                print(f"Loaded EMA50 data for {len(ema50_results)} symbols using 30 days collective data")
            
        except Exception as e:
            print(f"Warning: Could not load collective data for EMA50: {e}")
            self.collective_ema50_data = None
    
    def calculate_indicators(self, data):
        """
        Calculate all required indicators for the strategy with proper alignment
        EMA(3) and EMA(10) on 10-minute timeframe
        EMA(50) on 1-hour timeframe using 30 days of collective data from stock_data folder
        RSI(14) on 10-minute timeframe
        """
        # Store original data for reference
        self.original_data = data.copy()
        
        # Load collective 30 days data for EMA50 if not already loaded
        if self.collective_ema50_data is None:
            self.load_collective_data_for_ema50()
        
        results = []
        
        # Process each symbol for 10-minute indicators
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
            
            # Calculate EMAs and RSI on 10-minute timeframe
            ohlc_10min['EMA3'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_fast'])
            ohlc_10min['EMA10'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_slow'])
            ohlc_10min['RSI14'] = calculate_rsi(ohlc_10min['Close'], self.config['indicators']['rsi_period'])
            
            # Merge EMA50 from collective 30 days data to 10-minute data using forward fill
            if self.collective_ema50_data and symbol in self.collective_ema50_data:
                ohlc_10min = ohlc_10min.join(self.collective_ema50_data[symbol], how='left')
                ohlc_10min['EMA50'] = ohlc_10min['EMA50_1H'].ffill()
                ohlc_10min.drop('EMA50_1H', axis=1, inplace=True)
            else:
                # Fallback if no EMA50 data available
                ohlc_10min['EMA50'] = np.nan
                print(f"Warning: No collective EMA50 data available for {symbol}")
            
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
        pending_long_entry = False
        pending_long_target_price = 0
        pending_long_candle_idx = None
        pending_short_entry = False
        pending_short_target_price = 0
        pending_short_candle_idx = None
        
        capital = self.equity_curve[-1]
        risk_amount = capital * self.config['strategy']['risk_per_trade']
        
        # Iterate through each 10-minute candle
        for idx, row in stock_data.iterrows():
            current_price = row['Close']
            
            # Check if we're in a position
            if position is None:
                # Check if we have a pending long entry and price has reached target
                if pending_long_entry:
                    # Check if current candle's high reaches the target price
                    if row['High'] >= pending_long_target_price:
                        position = 'Long'
                        entry_price = pending_long_target_price  # Enter at the target price
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
                        print(f"Long entry executed: {symbol} at {entry_price}, quantity: {quantity}")
                        
                        # Reset pending entry
                        pending_long_entry = False
                        pending_long_target_price = 0
                        pending_long_candle_idx = None
                        # Also reset any pending short entry
                        pending_short_entry = False
                        pending_short_target_price = 0
                        pending_short_candle_idx = None
                
                # Check for new long entry signal - with some tolerance
                ema_condition = row['EMA3'] > row['EMA10'] * 1.001  # 0.1% buffer
                rsi_condition = row['RSI14'] > self.config['indicators']['rsi_overbought'] - 5  # 5-point buffer
                trend_condition = row['Close'] > row['EMA50'] * 1.002  # 0.2% buffer
                
                if ema_condition and rsi_condition and trend_condition and not pending_long_entry:
                    # Set up pending long entry - wait for price to reach high of entry candle
                    pending_long_entry = True
                    pending_long_target_price = row['High']  # Target price is high of current candle
                    pending_long_candle_idx = idx
                    print(f"Long entry signal detected for {symbol}. Waiting for price to reach {pending_long_target_price}")
                
                # Check if we have a pending short entry and price has reached target
                if pending_short_entry:
                    # Check if current candle's low reaches the target price
                    if row['Low'] <= pending_short_target_price:
                        position = 'Short'
                        entry_price = pending_short_target_price  # Enter at the target price
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
                        print(f"Short entry executed: {symbol} at {entry_price}, quantity: {quantity}")
                        
                        # Reset pending entry
                        pending_short_entry = False
                        pending_short_target_price = 0
                        pending_short_candle_idx = None
                
                # Check for new short entry signal - with some tolerance
                ema_condition = row['EMA3'] < row['EMA10'] * 0.999  # 0.1% buffer
                rsi_condition = row['RSI14'] < self.config['indicators']['rsi_oversold'] + 5  # 5-point buffer
                trend_condition = row['Close'] < row['EMA50'] * 0.998  # 0.2% buffer
                
                if ema_condition and rsi_condition and trend_condition and not pending_short_entry:
                    # Set up pending short entry - wait for price to reach low of last 5 minutes
                    target_price = self.get_low_of_last_5_minutes(symbol, idx)
                    
                    if target_price is None:
                        # Fallback to current low if no 1-minute data available
                        target_price = row['Low']
                    
                    pending_short_entry = True
                    pending_short_target_price = target_price
                    pending_short_candle_idx = idx
                    print(f"Short entry signal detected for {symbol}. Waiting for price to reach {pending_short_target_price}")
            
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
                        # Also reset any pending entries
                        pending_long_entry = False
                        pending_long_target_price = 0
                        pending_long_candle_idx = None
                        pending_short_entry = False
                        pending_short_target_price = 0
                        pending_short_candle_idx = None
                    
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
                        # Also reset any pending entries
                        pending_long_entry = False
                        pending_long_target_price = 0
                        pending_long_candle_idx = None
                        pending_short_entry = False
                        pending_short_target_price = 0
                        pending_short_candle_idx = None
                    
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