import pandas as pd
import numpy as np
# FIXED: Use absolute import instead of relative import
from src.indicators import calculate_ema, calculate_rsi

class BacktestEngine:
    def __init__(self, config):
        self.config = config
        self.trade_log = []
        self.trades = []
        # FIXED: Access base_capital from the strategy section of config
        self.equity_curve = [config['strategy']['base_capital']]
        
    def calculate_indicators(self, data):
        """
        Calculate all required indicators for the strategy
        """
        results = []
        
        for symbol, group in data.groupby('Symbol'):
            group = group.copy()
            
            # Calculate 10-minute OHLC data - FIXED: Changed '10T' to '10min'
            ohlc_10min = group.resample('10min').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Calculate 1-hour OHLC data for EMA50 - FIXED: Changed '1H' to '1h'
            ohlc_1hr = group.resample('1h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
            # Calculate EMAs and RSI on 10-minute data
            ohlc_10min['EMA3'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_fast'])
            ohlc_10min['EMA10'] = calculate_ema(ohlc_10min['Close'], self.config['indicators']['ema_slow'])
            ohlc_10min['RSI14'] = calculate_rsi(ohlc_10min['Close'], self.config['indicators']['rsi_period'])
            
            # Calculate EMA50 on 1-hour data
            ohlc_1hr['EMA50'] = calculate_ema(ohlc_1hr['Close'], self.config['indicators']['ema_trend'])
            
            # Merge EMA50 from 1-hour to 10-minute data
            ohlc_10min = ohlc_10min.merge(
                ohlc_1hr['EMA50'], 
                how='left', 
                left_index=True, 
                right_index=True,
                suffixes=('', '_1hr')
            )
            ohlc_10min['EMA50'] = ohlc_10min['EMA50'].ffill()
            
            # Add symbol information
            ohlc_10min['Symbol'] = symbol
            
            results.append(ohlc_10min)
        
        # Combine all symbols
        if results:
            result_df = pd.concat(results)
        else:
            result_df = pd.DataFrame()
        
        return result_df
    
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
            
            # Process each day
            day_data = data_with_indicators[data_with_indicators.index.normalize() == date]
            
            # Process each stock in the data (already filtered to top stocks)
            for symbol in day_data['Symbol'].unique():
                stock_data = day_data[day_data['Symbol'] == symbol].copy()
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
        
        # FIXED: Access base_capital from the strategy section of config
        capital = self.equity_curve[-1]
        risk_amount = capital * self.config['strategy']['risk_per_trade']
        
        # Iterate through each candle
        for idx, row in stock_data.iterrows():
            current_price = row['Close']
            
            # Check if we're in a position
            if position is None:
                # Check for long entry signal
                if (row['EMA3'] > row['EMA10'] and 
                    row['RSI14'] > self.config['indicators']['rsi_overbought'] and 
                    row['Close'] > row['EMA50']):
                    
                    position = 'Long'
                    entry_price = row['High']  # Buy at high of entry candle
                    entry_time = idx
                    # FIXED: Access stop_loss from the strategy section of config
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
                
                # Check for short entry signal
                elif (row['EMA3'] < row['EMA10'] and 
                      row['RSI14'] < self.config['indicators']['rsi_oversold'] and 
                      row['Close'] < row['EMA50']):
                    
                    position = 'Short'
                    # Get low of last 5 minutes for short entry
                    prev_data = stock_data.loc[:idx].tail(5)
                    if not prev_data.empty:
                        entry_price = prev_data['Low'].min()
                    else:
                        entry_price = row['Low']  # Fallback to current low
                    
                    entry_time = idx
                    # FIXED: Access stop_loss from the strategy section of config
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
                    # FIXED: Access profit_trail_start from the strategy section of config
                    elif current_price > entry_price * (1 + self.config['strategy']['profit_trail_start']):
                        trailing_active = True
                        if current_price > max_profit:
                            max_profit = current_price
                            # FIXED: Access trailing_stop from the strategy section of config
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
                    # FIXED: Access profit_trail_start from the strategy section of config
                    elif current_price < entry_price * (1 - self.config['strategy']['profit_trail_start']):
                        trailing_active = True
                        if current_price < max_profit:
                            max_profit = current_price
                            # FIXED: Access trailing_stop from the strategy section of config
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