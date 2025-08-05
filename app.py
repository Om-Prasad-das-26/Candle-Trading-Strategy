import os
from flask import Flask, render_template, request, jsonify, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import csv
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class AdvancedEMAStrategy:
    def __init__(self, data, symbol, ema_period=5, risk_reward_ratio=3.0, currency='USD'):
        self.data = data.copy()
        self.symbol = symbol
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.currency = currency
        self.signals = pd.DataFrame()
        self.trades = []
        self.future_predictions = []
        
    def get_currency_symbol(self):
        """Get currency symbol based on selection"""
        currency_symbols = {
            'USD': '$',
            'INR': '₹',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'C$',
            'AUD': 'A$'
        }
        return currency_symbols.get(self.currency, '$')
    
    def detect_market_type(self):
        """Detect if it's NSE, crypto, or US market"""
        symbol_upper = self.symbol.upper()
        if '.NS' in symbol_upper or '.BO' in symbol_upper:
            return 'NSE'
        elif '-USD' in symbol_upper or 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            return 'CRYPTO'
        else:
            return 'US'
    
    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI for trend analysis"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, data):
        """Calculate MACD for trend confirmation"""
        ema12 = self.calculate_ema(data, 12)
        ema26 = self.calculate_ema(data, 26)
        macd = ema12 - ema26
        signal = self.calculate_ema(macd, 9)
        return macd, signal
    
    def analyze_trend(self):
        """Advanced trend analysis"""
        if len(self.data) < 50:
            return "INSUFFICIENT_DATA"
        
        # EMA trend analysis
        ema_5 = self.data['EMA_5'].iloc[-1]
        ema_20 = self.data['EMA_20'].iloc[-1]
        ema_50 = self.calculate_ema(self.data['Close'], 50).iloc[-1]
        
        current_price = self.data['Close'].iloc[-1]
        
        # Short-term trend (5 vs 20 EMA)
        if ema_5 > ema_20 > ema_50:
            short_trend = "STRONG_BULLISH"
        elif ema_5 > ema_20:
            short_trend = "BULLISH"
        elif ema_5 < ema_20 < ema_50:
            short_trend = "STRONG_BEARISH"
        else:
            short_trend = "BEARISH"
        
        # Price vs EMA analysis
        if current_price > ema_5 > ema_20:
            price_trend = "ABOVE_ALL_EMAS"
        elif current_price < ema_5 < ema_20:
            price_trend = "BELOW_ALL_EMAS"
        else:
            price_trend = "MIXED"
        
        # EMA slope analysis
        ema_5_slope = (self.data['EMA_5'].iloc[-1] - self.data['EMA_5'].iloc[-10]) / 10
        ema_20_slope = (self.data['EMA_20'].iloc[-1] - self.data['EMA_20'].iloc[-10]) / 10
        
        # Volume trend (if available)
        volume_trend = "NORMAL"
        if 'Volume' in self.data.columns:
            recent_volume = self.data['Volume'].tail(5).mean()
            historical_volume = self.data['Volume'].tail(20).mean()
            if recent_volume > historical_volume * 1.2:
                volume_trend = "HIGH"
            elif recent_volume < historical_volume * 0.8:
                volume_trend = "LOW"
        
        return {
            'short_trend': short_trend,
            'price_trend': price_trend,
            'ema_5_slope': ema_5_slope,
            'ema_20_slope': ema_20_slope,
            'volume_trend': volume_trend,
            'current_price': current_price,
            'ema_5': ema_5,
            'ema_20': ema_20,
            'ema_50': ema_50
        }
    
    def prepare_data(self):
        """Prepare data with all technical indicators"""
        self.data['EMA_5'] = self.calculate_ema(self.data['Close'], self.ema_period)
        self.data['EMA_20'] = self.calculate_ema(self.data['Close'], 20)
        self.data['EMA_50'] = self.calculate_ema(self.data['Close'], 50)
        self.data['RSI'] = self.calculate_rsi(self.data['Close'])
        
        # MACD
        macd, macd_signal = self.calculate_macd(self.data['Close'])
        self.data['MACD'] = macd
        self.data['MACD_Signal'] = macd_signal
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        self.data['BB_Middle'] = self.data['Close'].rolling(window=bb_period).mean()
        bb_std_dev = self.data['Close'].rolling(window=bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std_dev * bb_std)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std_dev * bb_std)
        
        return self.data
    
    def identify_alerts(self):
        """Enhanced alert candle identification"""
        # Long Alert Candles: Close completely below EMA
        self.data['Long_Alert'] = (
            (self.data['Close'] < self.data['EMA_5']) & 
            (self.data['High'] < self.data['EMA_5']) &
            (self.data['RSI'] < 40)  # Additional RSI filter
        )
        
        # Short Alert Candles: Close completely above EMA  
        self.data['Short_Alert'] = (
            (self.data['Close'] > self.data['EMA_5']) & 
            (self.data['Low'] > self.data['EMA_5']) &
            (self.data['RSI'] > 60)  # Additional RSI filter
        )
        return self.data
    
    def generate_advanced_predictions(self):
        """Generate intelligent future predictions with reasoning"""
        if len(self.data) < 50:
            return []
        
        trend_analysis = self.analyze_trend()
        current_price = trend_analysis['current_price']
        ema_5 = trend_analysis['ema_5']
        ema_20 = trend_analysis['ema_20']
        ema_50 = trend_analysis['ema_50']
        
        currency_symbol = self.get_currency_symbol()
        predictions = []
        
        # Recent price volatility
        price_volatility = self.data['Close'].tail(20).std()
        avg_daily_range = (self.data['High'] - self.data['Low']).tail(20).mean()
        
        # Support and Resistance levels
        recent_highs = self.data['High'].tail(50).nlargest(5).mean()
        recent_lows = self.data['Low'].tail(50).nsmallest(5).mean()
        
        # Prediction 1: EMA-based predictions
        if trend_analysis['short_trend'] in ['BULLISH', 'STRONG_BULLISH']:
            if current_price < ema_5:
                # Potential long setup
                entry_level = ema_5 * 1.002  # Slightly above EMA
                stop_level = recent_lows * 0.998
                target_level = entry_level + (entry_level - stop_level) * self.risk_reward_ratio
                
                confidence = "HIGH" if trend_analysis['volume_trend'] == "HIGH" else "MEDIUM"
                
                predictions.append({
                    'type': 'LONG SETUP',
                    'entry_level': round(entry_level, 2),
                    'stop_loss': round(stop_level, 2),
                    'target': round(target_level, 2),
                    'confidence': confidence,
                    'reasoning': f"Bullish trend with price near EMA-{self.ema_period}. Entry above {currency_symbol}{entry_level:.2f} with EMA support. Stop below recent lows at {currency_symbol}{stop_level:.2f}.",
                    'time_frame': '3-7 days',
                    'risk_reward': f"1:{self.risk_reward_ratio}",
                    'additional_notes': f"Watch for volume confirmation. RSI: {self.data['RSI'].iloc[-1]:.1f}"
                })
        
        elif trend_analysis['short_trend'] in ['BEARISH', 'STRONG_BEARISH']:
            if current_price > ema_5:
                # Potential short setup
                entry_level = ema_5 * 0.998  # Slightly below EMA
                stop_level = recent_highs * 1.002
                target_level = entry_level - (stop_level - entry_level) * self.risk_reward_ratio
                
                confidence = "HIGH" if trend_analysis['volume_trend'] == "HIGH" else "MEDIUM"
                
                predictions.append({
                    'type': 'SHORT SETUP',
                    'entry_level': round(entry_level, 2),
                    'stop_loss': round(stop_level, 2),
                    'target': round(target_level, 2),
                    'confidence': confidence,
                    'reasoning': f"Bearish trend with price near EMA-{self.ema_period}. Entry below {currency_symbol}{entry_level:.2f} with EMA resistance. Stop above recent highs at {currency_symbol}{stop_level:.2f}.",
                    'time_frame': '3-7 days',
                    'risk_reward': f"1:{self.risk_reward_ratio}",
                    'additional_notes': f"Watch for volume confirmation. RSI: {self.data['RSI'].iloc[-1]:.1f}"
                })
        
        # Prediction 2: Breakout predictions
        if abs(current_price - ema_20) / current_price < 0.02:  # Price near EMA-20
            # Consolidation breakout setup
            upper_breakout = max(recent_highs, ema_20 * 1.03)
            lower_breakout = min(recent_lows, ema_20 * 0.97)
            
            predictions.append({
                'type': 'BREAKOUT SETUP',
                'entry_level': f"Above {currency_symbol}{upper_breakout:.2f} OR Below {currency_symbol}{lower_breakout:.2f}",
                'stop_loss': f"Opposite side of breakout level",
                'target': f"Based on breakout direction",
                'confidence': 'MEDIUM',
                'reasoning': f"Price consolidating near EMA-20 ({currency_symbol}{ema_20:.2f}). Breakout likely above {currency_symbol}{upper_breakout:.2f} or below {currency_symbol}{lower_breakout:.2f}.",
                'time_frame': '1-5 days',
                'risk_reward': f"1:{self.risk_reward_ratio}",
                'additional_notes': f"Wait for volume surge on breakout. Current consolidation range: {abs(upper_breakout - lower_breakout):.2f}"
            })
        
        # Prediction 3: Mean reversion setup
        rsi_current = self.data['RSI'].iloc[-1]
        if rsi_current > 75:
            # Overbought - potential short
            predictions.append({
                'type': 'MEAN REVERSION SHORT',
                'entry_level': f"Current levels around {currency_symbol}{current_price:.2f}",
                'stop_loss': f"{currency_symbol}{current_price * 1.03:.2f}",
                'target': f"{currency_symbol}{ema_20:.2f} (EMA-20)",
                'confidence': 'MEDIUM',
                'reasoning': f"RSI extremely overbought at {rsi_current:.1f}. Price likely to revert to EMA-20 mean.",
                'time_frame': '2-7 days',
                'risk_reward': '1:2',
                'additional_notes': f"High-risk setup. Consider partial position sizing."
            })
        elif rsi_current < 25:
            # Oversold - potential long
            predictions.append({
                'type': 'MEAN REVERSION LONG',
                'entry_level': f"Current levels around {currency_symbol}{current_price:.2f}",
                'stop_loss': f"{currency_symbol}{current_price * 0.97:.2f}",
                'target': f"{currency_symbol}{ema_20:.2f} (EMA-20)",
                'confidence': 'MEDIUM',
                'reasoning': f"RSI extremely oversold at {rsi_current:.1f}. Price likely to revert to EMA-20 mean.",
                'time_frame': '2-7 days',
                'risk_reward': '1:2',
                'additional_notes': f"High-risk setup. Consider partial position sizing."
            })
        
        self.future_predictions = predictions
        return predictions
    
    def generate_signals(self):
        """Generate detailed trading signals with entry/exit points"""
        signals = []
        
        for i in range(10, len(self.data)):  # Start from index 10 for better analysis
            current = self.data.iloc[i]
            current_date = current.name
            
            # Look for Alert Candles in recent history (last 5 candles)
            recent_data = self.data.iloc[max(0, i-5):i]
            
            # Check for long signals
            long_alerts = recent_data[recent_data['Long_Alert']]
            if not long_alerts.empty:
                latest_long_alert = long_alerts.iloc[-1]
                alert_date = latest_long_alert.name
                
                # Entry condition: Current high breaks Alert Candle high
                if current['High'] > latest_long_alert['High']:
                    entry_price = latest_long_alert['High']
                    stop_loss = latest_long_alert['Low']
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.risk_reward_ratio)
                    
                    signals.append({
                        'Signal_Date': current_date,
                        'Alert_Candle_Date': alert_date,
                        'Type': 'LONG',
                        'Entry_Price': round(entry_price, 2),
                        'Stop_Loss': round(stop_loss, 2),
                        'Take_Profit': round(take_profit, 2),
                        'Risk_Amount': round(risk, 2),
                        'Reward_Potential': round(risk * self.risk_reward_ratio, 2),
                        'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                        'Alert_Candle_High': round(latest_long_alert['High'], 2),
                        'Alert_Candle_Low': round(latest_long_alert['Low'], 2),
                        'Alert_Candle_Close': round(latest_long_alert['Close'], 2),
                        'Current_Price': round(current['Close'], 2),
                        'RSI_At_Signal': round(current['RSI'], 1),
                        'MACD_At_Signal': round(current['MACD'], 3)
                    })
            
            # Check for short signals
            short_alerts = recent_data[recent_data['Short_Alert']]
            if not short_alerts.empty:
                latest_short_alert = short_alerts.iloc[-1]
                alert_date = latest_short_alert.name
                
                # Entry condition: Current low breaks Alert Candle low
                if current['Low'] < latest_short_alert['Low']:
                    entry_price = latest_short_alert['Low']
                    stop_loss = latest_short_alert['High']
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.risk_reward_ratio)
                    
                    signals.append({
                        'Signal_Date': current_date,
                        'Alert_Candle_Date': alert_date,
                        'Type': 'SHORT',
                        'Entry_Price': round(entry_price, 2),
                        'Stop_Loss': round(stop_loss, 2),
                        'Take_Profit': round(take_profit, 2),
                        'Risk_Amount': round(risk, 2),
                        'Reward_Potential': round(risk * self.risk_reward_ratio, 2),
                        'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                        'Alert_Candle_High': round(latest_short_alert['High'], 2),
                        'Alert_Candle_Low': round(latest_short_alert['Low'], 2),
                        'Alert_Candle_Close': round(latest_short_alert['Close'], 2),
                        'Current_Price': round(current['Close'], 2),
                        'RSI_At_Signal': round(current['RSI'], 1),
                        'MACD_At_Signal': round(current['MACD'], 3)
                    })
        
        self.signals = pd.DataFrame(signals)
        return self.signals
    
    def backtest_signals(self):
        """Backtest the generated signals"""
        if self.signals.empty:
            return pd.DataFrame()
        
        trades = []
        
        for idx, signal in self.signals.iterrows():
            entry_date = signal['Signal_Date']
            entry_price = signal['Entry_Price']
            stop_loss = signal['Stop_Loss']
            take_profit = signal['Take_Profit']
            trade_type = signal['Type']
            
            # Look for exit in future data
            future_data = self.data[self.data.index > entry_date]
            exit_price = None
            exit_date = None
            exit_reason = None
            
            for future_idx, future_candle in future_data.iterrows():
                if trade_type == 'LONG':
                    if future_candle['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_date = future_idx
                        exit_reason = 'Stop Loss Hit'
                        break
                    elif future_candle['High'] >= take_profit:
                        exit_price = take_profit
                        exit_date = future_idx
                        exit_reason = 'Take Profit Hit'
                        break
                else:  # SHORT
                    if future_candle['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_date = future_idx
                        exit_reason = 'Stop Loss Hit'
                        break
                    elif future_candle['Low'] <= take_profit:
                        exit_price = take_profit
                        exit_date = future_idx
                        exit_reason = 'Take Profit Hit'
                        break
            
            # If no exit found, mark as open
            if exit_price is None:
                exit_price = self.data['Close'].iloc[-1]
                exit_date = self.data.index[-1]
                exit_reason = 'Position Still Open'
            
            # Calculate P&L
            if trade_type == 'LONG':
                pnl = exit_price - entry_price
                pnl_percent = (pnl / entry_price) * 100
            else:
                pnl = entry_price - exit_price
                pnl_percent = (pnl / entry_price) * 100
            
            # Calculate holding period
            holding_days = (exit_date - entry_date).days if exit_date != entry_date else 0
            
            trades.append({
                'Entry_Date': entry_date.strftime('%Y-%m-%d'),
                'Exit_Date': exit_date.strftime('%Y-%m-%d'),
                'Type': trade_type,
                'Entry_Price': round(entry_price, 2),
                'Exit_Price': round(exit_price, 2),
                'Stop_Loss': round(stop_loss, 2),
                'Take_Profit': round(take_profit, 2),
                'PnL': round(pnl, 2),
                'PnL_Percent': round(pnl_percent, 2),
                'Exit_Reason': exit_reason,
                'Holding_Days': holding_days,
                'Risk_Amount': signal['Risk_Amount'],
                'Reward_Potential': signal['Reward_Potential'],
                'RSI_At_Entry': signal['RSI_At_Signal'],
                'MACD_At_Entry': signal['MACD_At_Signal']
            })
        
        self.trades = pd.DataFrame(trades)
        return self.trades
    
    def analyze(self):
        """Run complete analysis"""
        self.prepare_data()
        self.identify_alerts()
        signals = self.generate_signals()
        trades = self.backtest_signals()
        future_predictions = self.generate_advanced_predictions()
        trend_analysis = self.analyze_trend()
        
        # Calculate performance metrics
        total_signals = len(signals)
        total_trades = len(trades)
        
        if total_trades > 0:
            winning_trades = len(trades[trades['PnL'] > 0])
            losing_trades = len(trades[trades['PnL'] < 0])
            win_rate = winning_trades / total_trades
            avg_pnl = trades['PnL'].mean()
            total_pnl = trades['PnL'].sum()
            avg_holding_days = trades['Holding_Days'].mean()
            max_win = trades['PnL'].max()
            max_loss = trades['PnL'].min()
            
            # Calculate profit factor
            total_wins = trades[trades['PnL'] > 0]['PnL'].sum()
            total_losses = abs(trades[trades['PnL'] < 0]['PnL'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_pnl = total_pnl = avg_holding_days = max_win = max_loss = profit_factor = 0
        
        return {
            'long_alerts': int(self.data['Long_Alert'].sum()),
            'short_alerts': int(self.data['Short_Alert'].sum()),
            'total_candles': len(self.data),
            'total_signals': total_signals,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate * 100, 1),
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_holding_days': round(avg_holding_days, 1),
            'max_win': round(max_win, 2),
            'max_loss': round(max_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'future_predictions': future_predictions,
            'trend_analysis': trend_analysis,
            'success': True
        }

def fetch_extended_data(symbol, period='1y'):
    """Enhanced data fetching with better error handling for extended periods"""
    try:
        ticker = yf.Ticker(symbol)
        
        # For maximum data, try different approaches
        if period == 'max':
            # Try to get maximum available data
            data = ticker.history(period='max', auto_adjust=True, prepost=True)
            if data.empty:
                # Fallback to 20 years
                data = ticker.history(period='20y', auto_adjust=True)
        elif period == '10y':
            # Try 10 years, fallback to 5y if needed
            try:
                data = ticker.history(period='10y', auto_adjust=True)
                if len(data) < 100:  # If very little data, try different approach
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=10*365)
                    data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            except:
                data = ticker.history(period='5y', auto_adjust=True)
        else:
            data = ticker.history(period=period, auto_adjust=True)
        
        if data.empty:
            return None
            
        # Clean the data
        data = data.dropna()
        
        # Ensure we have minimum required data
        if len(data) < 50:
            print(f"Warning: Only {len(data)} data points available for {symbol}")
        
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        # Try fallback periods
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='2y', auto_adjust=True)
            if not data.empty:
                return data.dropna()
        except:
            pass
        return None

def create_professional_chart(strategy, symbol):
    """Create professional-grade chart with all indicators"""
    try:
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
        ax1 = fig.add_subplot(gs[0])  # Price chart
        ax2 = fig.add_subplot(gs[1])  # Volume
        ax3 = fig.add_subplot(gs[2])  # RSI
        ax4 = fig.add_subplot(gs[3])  # MACD
        
        currency_symbol = strategy.get_currency_symbol()
        
        # Main price chart
        ax1.plot(strategy.data.index, strategy.data['Close'], label='Close', linewidth=1.5, color='#2E86AB')
        ax1.plot(strategy.data.index, strategy.data['EMA_5'], label=f'EMA {strategy.ema_period}', color='#A23B72', linewidth=2)
        ax1.plot(strategy.data.index, strategy.data['EMA_20'], label='EMA 20', color='#F18F01', linewidth=1.5)
        ax1.plot(strategy.data.index, strategy.data['EMA_50'], label='EMA 50', color='#C73E1D', linewidth=1, alpha=0.7)
        
        # Bollinger Bands
        ax1.fill_between(strategy.data.index, strategy.data['BB_Upper'], strategy.data['BB_Lower'], 
                        alpha=0.1, color='gray', label='Bollinger Bands')
        ax1.plot(strategy.data.index, strategy.data['BB_Upper'], color='gray', linestyle='--', alpha=0.5)
        ax1.plot(strategy.data.index, strategy.data['BB_Lower'], color='gray', linestyle='--', alpha=0.5)
        
        # Alert Candles
        long_alerts = strategy.data[strategy.data['Long_Alert']]
        short_alerts = strategy.data[strategy.data['Short_Alert']]
        
        if not long_alerts.empty:
            ax1.scatter(long_alerts.index, long_alerts['Low'], color='#90EE90', marker='^', 
                       s=60, label='Long Alert', alpha=0.8, edgecolor='darkgreen', linewidth=1)
        if not short_alerts.empty:
            ax1.scatter(short_alerts.index, short_alerts['High'], color='#FFB6C1', marker='v', 
                       s=60, label='Short Alert', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Entry Points
        if not strategy.signals.empty:
            long_entries = strategy.signals[strategy.signals['Type'] == 'LONG']
            short_entries = strategy.signals[strategy.signals['Type'] == 'SHORT']
            
            if not long_entries.empty:
                ax1.scatter(long_entries['Signal_Date'], long_entries['Entry_Price'], 
                           color='darkgreen', marker='o', s=100, label='Long Entry', 
                           edgecolor='white', linewidth=2, zorder=5)
            if not short_entries.empty:
                ax1.scatter(short_entries['Signal_Date'], short_entries['Entry_Price'], 
                           color='darkred', marker='o', s=100, label='Short Entry', 
                           edgecolor='white', linewidth=2, zorder=5)
        
        ax1.set_title(f'{symbol} - Advanced EMA Alert Candle Strategy', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Volume chart
        colors = ['red' if close < open_ else 'green' for close, open_ in 
                 zip(strategy.data['Close'], strategy.data['Open'])]
        ax2.bar(strategy.data.index, strategy.data['Volume'], alpha=0.7, color=colors, width=1)
        ax2.set_ylabel('Volume', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # RSI chart
        ax3.plot(strategy.data.index, strategy.data['RSI'], color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax3.fill_between(strategy.data.index, 30, 70, alpha=0.1, color='yellow')
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # MACD chart
        ax4.plot(strategy.data.index, strategy.data['MACD'], color='blue', linewidth=1.5, label='MACD')
        ax4.plot(strategy.data.index, strategy.data['MACD_Signal'], color='red', linewidth=1.5, label='Signal')
        macd_histogram = strategy.data['MACD'] - strategy.data['MACD_Signal']
        ax4.bar(strategy.data.index, macd_histogram, alpha=0.3, color='gray', width=1, label='Histogram')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax4.set_ylabel('MACD', fontsize=10)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Complete Technical Analysis - {len(strategy.data)} data points', fontsize=14, y=0.98)
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return chart_url
    except Exception as e:
        print(f"Chart error: {e}")
        return None

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Professional EMA Alert Candle Strategy</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; padding: 60px 0; 
        }
        .card { 
            box-shadow: 0 8px 16px rgba(0,0,0,0.1); 
            border: none; margin-bottom: 20px; 
            transition: transform 0.2s ease;
        }
        .card:hover { transform: translateY(-2px); }
        .metric-card { 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
            border-left: 4px solid #007bff; 
            transition: all 0.3s ease;
        }
        .metric-card:hover { border-left-width: 6px; }
        .btn-analyze { 
            background: linear-gradient(45deg, #007bff, #0056b3); 
            border: none; 
            box-shadow: 0 4px 8px rgba(0,123,255,0.3);
        }
        .download-section { 
            background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%); 
            border-radius: 15px; padding: 25px; margin: 20px 0;
            border: 1px solid #dee2e6;
        }
        .prediction-card { 
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
            border-radius: 10px; padding: 20px; margin: 10px 0;
            border-left: 5px solid #ffc107;
        }
        .trend-analysis { 
            background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%); 
            border-radius: 10px; padding: 20px;
            border-left: 5px solid #17a2b8;
        }
        .confidence-high { border-left-color: #28a745 !important; }
        .confidence-medium { border-left-color: #ffc107 !important; }
        .confidence-low { border-left-color: #dc3545 !important; }
        .currency-badge { 
            background: linear-gradient(45deg, #28a745, #20c997);
            border: none;
        }
    </style>
</head>
<body>
    <div class="hero text-center">
        <div class="container">
            <h1 class="display-4 mb-3">
                <i class="fas fa-chart-line me-3"></i>Professional EMA Alert Strategy
            </h1>
            <p class="lead">Advanced 10-Year Analysis • Intelligent Predictions • Multi-Currency Support</p>
        </div>
    </div>
    
    <div class="container my-5">
        <div class="row">
            <div class="col-md-4">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0"><i class="fas fa-cog me-2"></i>Strategy Configuration</h5>
                    </div>
                    <div class="card-body">
                        <form id="analysisForm">
                            <div class="mb-3">
                                <label class="form-label">Stock Symbol</label>
                                <input type="text" class="form-control" id="symbol" value="AAPL" placeholder="AAPL, RELIANCE.NS, BTC-USD">
                                <small class="form-text text-muted">
                                    <strong>US:</strong> AAPL, MSFT, GOOGL<br>
                                    <strong>NSE:</strong> RELIANCE.NS, TCS.NS<br>
                                    <strong>Crypto:</strong> BTC-USD, ETH-USD
                                </small>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Time Period</label>
                                <select class="form-select" id="period">
                                    <option value="1mo">1 Month</option>
                                    <option value="3mo">3 Months</option>
                                    <option value="6mo">6 Months</option>
                                    <option value="1y" selected>1 Year</option>
                                    <option value="2y">2 Years</option>
                                    <option value="5y">5 Years</option>
                                    <option value="10y">🔥 10 Years</option>
                                    <option value="max">🚀 Maximum Available</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Currency Display</label>
                                <select class="form-select" id="currency">
                                    <option value="USD">🇺🇸 US Dollar ($)</option>
                                    <option value="INR">🇮🇳 Indian Rupee (₹)</option>
                                    <option value="EUR">🇪🇺 Euro (€)</option>
                                    <option value="GBP">🇬🇧 British Pound (£)</option>
                                    <option value="JPY">🇯🇵 Japanese Yen (¥)</option>
                                    <option value="CAD">🇨🇦 Canadian Dollar (C$)</option>
                                    <option value="AUD">🇦🇺 Australian Dollar (A$)</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">EMA Period</label>
                                <select class="form-select" id="ema_period">
                                    <option value="5" selected>5 Period (Fast)</option>
                                    <option value="8">8 Period</option>
                                    <option value="13">13 Period</option>
                                    <option value="21">21 Period (Slow)</option>
                                </select>
                            </div>
                            
                            <div class="mb-3">
                                <label class="form-label">Risk:Reward Ratio</label>
                                <select class="form-select" id="risk_reward">
                                    <option value="1.5">1:1.5 (Conservative)</option>
                                    <option value="2">1:2 (Balanced)</option>
                                    <option value="3" selected>1:3 (Aggressive)</option>
                                    <option value="4">1:4 (Very Aggressive)</option>
                                    <option value="5">1:5 (Extreme)</option>
                                </select>
                            </div>
                            
                            <button type="submit" class="btn btn-analyze btn-primary w-100 py-3 mb-3">
                                <i class="fas fa-rocket me-2"></i>Analyze Strategy
                            </button>
                            
                            <div class="alert alert-info">
                                <small><i class="fas fa-info-circle me-1"></i>
                                <strong>New:</strong> Extended 10-year analysis with intelligent future predictions!</small>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
            
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-star me-2"></i>Professional Features</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6 class="text-primary"><i class="fas fa-chart-area me-2"></i>Advanced Analysis</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-check text-success me-2"></i>Up to 10 years historical data</li>
                                    <li><i class="fas fa-check text-success me-2"></i>RSI + MACD + Bollinger Bands</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Multi-timeframe EMA analysis</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Volume confirmation</li>
                                </ul>
                                
                                <h6 class="text-primary mt-3"><i class="fas fa-brain me-2"></i>AI Predictions</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-robot text-warning me-2"></i>Intelligent trend analysis</li>
                                    <li><i class="fas fa-target text-info me-2"></i>Future entry/exit levels</li>
                                    <li><i class="fas fa-percentage text-success me-2"></i>Confidence scoring</li>
                                    <li><i class="fas fa-clock text-primary me-2"></i>Time frame predictions</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6 class="text-primary"><i class="fas fa-globe me-2"></i>Multi-Market Support</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-flag-usa text-primary me-2"></i>US Stocks (NYSE, NASDAQ)</li>
                                    <li><i class="fas fa-rupee-sign text-warning me-2"></i>NSE India (.NS, .BO)</li>
                                    <li><i class="fas fa-bitcoin text-danger me-2"></i>Cryptocurrencies</li>
                                    <li><i class="fas fa-coins text-success me-2"></i>7 Currency displays</li>
                                </ul>
                                
                                <h6 class="text-primary mt-3"><i class="fas fa-download me-2"></i>Data Export</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-file-csv text-success me-2"></i>Complete trade history</li>
                                    <li><i class="fas fa-chart-bar text-info me-2"></i>Performance metrics</li>
                                    <li><i class="fas fa-crystal-ball text-warning me-2"></i>Future predictions</li>
                                    <li><i class="fas fa-cog text-secondary me-2"></i>Custom analysis</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="loading" class="text-center my-5" style="display:none;">
            <div class="spinner-border text-primary" style="width: 4rem; height: 4rem;"></div>
            <p class="mt-3 h5">Analyzing strategy... Loading extended data may take up to 30 seconds.</p>
            <div class="progress mt-3" style="height: 6px;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
            </div>
        </div>
        
        <div id="results" class="mt-5" style="display:none;">
            <!-- Trend Analysis Section -->
            <div id="trendSection" class="trend-analysis mb-4">
                <h5><i class="fas fa-trending-up me-2"></i>Current Market Trend Analysis</h5>
                <div id="trendAnalysis"></div>
            </div>
            
            <!-- Performance Summary -->
            <div class="card">
                <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
                    <h5 class="mb-0"><i class="fas fa-trophy me-2"></i>Performance Dashboard</h5>
                    <div id="statusBadges"></div>
                </div>
                <div class="card-body">
                    <div class="row" id="metricsRow"></div>
                </div>
            </div>
            
            <!-- Future Predictions Section -->
            <div id="predictionsSection" class="mb-4" style="display:none;">
                <h5 class="mb-3"><i class="fas fa-crystal-ball me-2"></i>Intelligent Future Predictions</h5>
                <div id="futurePredictions"></div>
            </div>
            
            <!-- Download Section -->
            <div class="download-section">
                <h5><i class="fas fa-cloud-download-alt me-2"></i>Export Trading Data</h5>
                <p class="text-muted mb-3">Download complete analysis including historical trades, predictions, and performance metrics</p>
                <div class="row">
                    <div class="col-md-3">
                        <button class="btn btn-outline-primary w-100 mb-2" onclick="downloadSignals()">
                            <i class="fas fa-signal me-2"></i>Trading Signals
                        </button>
                    </div>
                    <div class="col-md-3">
                        <button class="btn btn-outline-success w-100 mb-2" onclick="downloadTrades()">
                            <i class="fas fa-exchange-alt me-2"></i>Trade Results
                        </button>
                    </div>
                    <div class="col-md-3">
                        <button class="btn btn-outline-warning w-100 mb-2" onclick="downloadPredictions()">
                            <i class="fas fa-crystal-ball me-2"></i>Predictions
                        </button>
                    </div>
                    <div class="col-md-3">
                        <button class="btn btn-outline-info w-100 mb-2" onclick="downloadAll()">
                            <i class="fas fa-file-export me-2"></i>Complete Report
                        </button>
                    </div>
                </div>
            </div>
            
            <!-- Chart Section -->
            <div class="card mt-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="fas fa-chart-area me-2"></i>Professional Technical Analysis</h5>
                </div>
                <div class="card-body text-center">
                    <img id="strategyChart" class="img-fluid rounded shadow" alt="Strategy Chart" style="max-height: 800px;">
                </div>
            </div>
        </div>
        
        <div id="error" class="alert alert-danger mt-4" style="display:none;"></div>
        <div id="connectionStatus" class="mt-3"></div>
    </div>
    
    <footer class="bg-dark text-white text-center py-4 mt-5">
        <div class="container">
            <p class="mb-1">Professional EMA Alert Candle Strategy v2.0</p>
            <small class="text-muted">Educational Purpose Only • Not Financial Advice</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentAnalysisData = null;
        let currentSymbol = null;
        let currentCurrency = null;
        
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const symbol = document.getElementById('symbol').value.toUpperCase().trim();
            const period = document.getElementById('period').value;
            const currency = document.getElementById('currency').value;
            const ema_period = document.getElementById('ema_period').value;
            const risk_reward = document.getElementById('risk_reward').value;
            
            if (!symbol) {
                showError('Please enter a valid stock symbol');
                return;
            }
            
            currentSymbol = symbol;
            currentCurrency = currency;
            
            showLoading(true);
            hideResults();
            hideError();
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: symbol, 
                        period: period,
                        currency: currency,
                        ema_period: parseInt(ema_period),
                        risk_reward: parseFloat(risk_reward)
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentAnalysisData = data;
                    displayResults(data, symbol, currency);
                } else {
                    showError(data.error || 'Analysis failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        });
        
        function displayResults(data, symbol, currency) {
            const currencySymbols = {
                'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£', 
                'JPY': '¥', 'CAD': 'C$', 'AUD': 'A$'
            };
            const currencySymbol = currencySymbols[currency] || '$';
            
            // Display trend analysis
            if (data.trend_analysis) {
                const trend = data.trend_analysis;
                const trendHtml = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6><i class="fas fa-chart-line me-2"></i>Trend Direction</h6>
                            <p><strong>Short-term:</strong> <span class="badge bg-${getTrendColor(trend.short_trend)}">${trend.short_trend}</span></p>
                            <p><strong>Price vs EMAs:</strong> <span class="badge bg-${getTrendColor(trend.price_trend)}">${trend.price_trend}</span></p>
                            <p><strong>Volume:</strong> <span class="badge bg-${getVolumeColor(trend.volume_trend)}">${trend.volume_trend}</span></p>
                        </div>
                        <div class="col-md-6">
                            <h6><i class="fas fa-layer-group me-2"></i>EMA Levels</h6>
                            <p><strong>Current Price:</strong> ${currencySymbol}${trend.current_price.toFixed(2)}</p>
                            <p><strong>EMA-5:</strong> ${currencySymbol}${trend.ema_5.toFixed(2)}</p>
                            <p><strong>EMA-20:</strong> ${currencySymbol}${trend.ema_20.toFixed(2)}</p>
                            <p><strong>EMA-50:</strong> ${currencySymbol}${trend.ema_50.toFixed(2)}</p>
                        </div>
                    </div>
                `;
                document.getElementById('trendAnalysis').innerHTML = trendHtml;
            }
            
            // Status badges
            document.getElementById('statusBadges').innerHTML = `
                <span class="badge bg-light text-dark me-1">${data.total_candles} candles</span>
                <span class="badge bg-primary me-1">${data.total_signals} signals</span>
                <span class="badge bg-success me-1">${data.long_alerts} long alerts</span>
                <span class="badge bg-danger me-1">${data.short_alerts} short alerts</span>
                <span class="badge currency-badge text-white">${currency}</span>
            `;
            
            // Performance metrics
            const metricsRow = document.getElementById('metricsRow');
            metricsRow.innerHTML = `
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total Signals</h6>
                        <h3 class="text-primary mb-0">${data.total_signals}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Win Rate</h6>
                        <h3 class="text-success mb-0">${data.win_rate}%</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total P&L</h6>
                        <h3 class="${data.total_pnl >= 0 ? 'text-success' : 'text-danger'} mb-0">${currencySymbol}${data.total_pnl}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Profit Factor</h6>
                        <h3 class="text-info mb-0">${data.profit_factor}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Max Win</h6>
                        <h3 class="text-success mb-0">${currencySymbol}${data.max_win}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Max Loss</h6>
                        <h3 class="text-danger mb-0">${currencySymbol}${data.max_loss}</h3>
                    </div>
                </div>
            `;
            
            // Future predictions
            if (data.future_predictions && data.future_predictions.length > 0) {
                let predictionsHtml = '';
                data.future_predictions.forEach((prediction, index) => {
                    const confidenceClass = `confidence-${prediction.confidence.toLowerCase()}`;
                    predictionsHtml += `
                        <div class="prediction-card ${confidenceClass} mb-3">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <h6 class="mb-0">
                                    <i class="fas fa-bullseye me-2"></i>${prediction.type}
                                    <span class="badge bg-${getConfidenceBadgeColor(prediction.confidence)} ms-2">${prediction.confidence}</span>
                                </h6>
                                <small class="text-muted">${prediction.time_frame}</small>
                            </div>
                            <div class="row">
                                <div class="col-md-8">
                                    <p class="mb-2"><strong>Entry:</strong> ${prediction.entry_level}</p>
                                    <p class="mb-2"><strong>Stop Loss:</strong> ${prediction.stop_loss}</p>
                                    <p class="mb-2"><strong>Target:</strong> ${prediction.target}</p>
                                    <p class="mb-1"><strong>Reasoning:</strong></p>
                                    <p class="small text-muted">${prediction.reasoning}</p>
                                </div>
                                <div class="col-md-4">
                                    <p class="mb-1"><strong>Risk:Reward:</strong> ${prediction.risk_reward}</p>
                                    <p class="small text-info">${prediction.additional_notes}</p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('futurePredictions').innerHTML = predictionsHtml;
                document.getElementById('predictionsSection').style.display = 'block';
            }
            
            // Chart
            if (data.chart) {
                document.getElementById('strategyChart').src = 'data:image/png;base64,' + data.chart;
            }
            
            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }
        
        function getTrendColor(trend) {
            if (trend.includes('BULLISH') || trend.includes('ABOVE')) return 'success';
            if (trend.includes('BEARISH') || trend.includes('BELOW')) return 'danger';
            return 'warning';
        }
        
        function getVolumeColor(volume) {
            if (volume === 'HIGH') return 'success';
            if (volume === 'LOW') return 'danger';
            return 'secondary';
        }
        
        function getConfidenceBadgeColor(confidence) {
            switch(confidence) {
                case 'HIGH': return 'success';
                case 'MEDIUM': return 'warning';
                case 'LOW': return 'danger';
                default: return 'secondary';
            }
        }
        
        // Download functions
        async function downloadSignals() {
            await downloadData('/download/signals', 'signals.csv');
        }
        
        async function downloadTrades() {
            await downloadData('/download/trades', 'trades.csv');
        }
        
        async function downloadPredictions() {
            await downloadData('/download/predictions', 'predictions.csv');
        }
        
        async function downloadAll() {
            await downloadData('/download/all', 'complete_analysis.csv');
        }
        
        async function downloadData(endpoint, filename) {
            if (!currentAnalysisData) {
                alert('Please run analysis first');
                return;
            }
            
            try {
                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        symbol: currentSymbol,
                        currency: currentCurrency
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Download failed');
                }
                
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${currentSymbol}_${filename}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                alert('Download failed: ' + error.message);
            }
        }
        
        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }
        
        function hideResults() {
            document.getElementById('results').style.display = 'none';
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = '<i class="fas fa-exclamation-triangle me-2"></i>' + message;
            errorDiv.style.display = 'block';
            errorDiv.scrollIntoView({ behavior: 'smooth' });
        }
        
        function hideError() {
            document.getElementById('error').style.display = 'none';
        }
    </script>
</body>
</html>"""

# Store analysis data globally for downloads
analysis_cache = {}

@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis_cache
    
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper().strip()
        period = data.get('period', '1y')
        currency = data.get('currency', 'USD')
        ema_period = int(data.get('ema_period', 5))
        risk_reward = float(data.get('risk_reward', 3.0))
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Invalid symbol'})
        
        # Auto-detect currency based on symbol
        if '.NS' in symbol or '.BO' in symbol:
            currency = 'INR'  # NSE stocks default to INR
        elif 'BTC' in symbol or 'ETH' in symbol or '-USD' in symbol:
            currency = 'USD'  # Crypto default to USD
        
        print(f"Fetching {period} data for {symbol}...")
        market_data = fetch_extended_data(symbol, period)
        
        if market_data is None:
            return jsonify({'success': False, 'error': f'Could not fetch data for {symbol}. Please verify the symbol format and try again.'})
        
        print(f"Retrieved {len(market_data)} data points for {symbol}")
        
        strategy = AdvancedEMAStrategy(
            market_data, 
            symbol=symbol,
            ema_period=ema_period, 
            risk_reward_ratio=risk_reward,
            currency=currency
        )
        
        results = strategy.analyze()
        
        # Store data for downloads
        analysis_cache[symbol] = {
            'strategy': strategy,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'predictions': strategy.future_predictions,
            'market_data': market_data,
            'currency': currency,
            'timestamp': datetime.now()
        }
        
        chart_data = create_professional_chart(strategy, symbol)
        results['chart'] = chart_data
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})

@app.route('/download/signals', methods=['POST'])
def download_signals():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        
        if symbol not in analysis_cache:
            return jsonify({'error': 'No analysis data found. Please run analysis first.'}), 400
        
        signals_df = analysis_cache[symbol]['signals']
        currency = analysis_cache[symbol]['currency']
        
        if signals_df.empty:
            return jsonify({'error': 'No signals found for this analysis.'}), 400
        
        # Create CSV with currency info
        output = StringIO()
        output.write(f"# Trading Signals for {symbol} (Currency: {currency})\n")
        output.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        signals_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_signals_{currency}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/trades', methods=['POST'])
def download_trades():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        
        if symbol not in analysis_cache:
            return jsonify({'error': 'No analysis data found. Please run analysis first.'}), 400
        
        trades_df = analysis_cache[symbol]['trades']
        currency = analysis_cache[symbol]['currency']
        
        if trades_df.empty:
            return jsonify({'error': 'No trades found for this analysis.'}), 400
        
        # Create CSV with currency info
        output = StringIO()
        output.write(f"# Trade Results for {symbol} (Currency: {currency})\n")
        output.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        trades_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_trades_{currency}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/predictions', methods=['POST'])
def download_predictions():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        
        if symbol not in analysis_cache:
            return jsonify({'error': 'No analysis data found. Please run analysis first.'}), 400
        
        predictions = analysis_cache[symbol]['predictions']
        currency = analysis_cache[symbol]['currency']
        
        if not predictions:
            return jsonify({'error': 'No predictions found for this analysis.'}), 400
        
        # Convert predictions to DataFrame
        predictions_df = pd.DataFrame(predictions)
        
        # Create CSV
        output = StringIO()
        output.write(f"# Future Predictions for {symbol} (Currency: {currency})\n")
        output.write(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        predictions_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_predictions_{currency}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/all', methods=['POST'])
def download_all():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        
        if symbol not in analysis_cache:
            return jsonify({'error': 'No analysis data found. Please run analysis first.'}), 400
        
        cache_data = analysis_cache[symbol]
        currency = cache_data['currency']
        
        # Create comprehensive report
        output = StringIO()
        
        # Header
        output.write(f"PROFESSIONAL EMA ALERT CANDLE STRATEGY - COMPLETE ANALYSIS REPORT\n")
        output.write(f"=" * 80 + "\n")
        output.write(f"Symbol: {symbol}\n")
        output.write(f"Currency: {currency}\n")
        output.write(f"Analysis Date: {cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(f"Total Data Points: {len(cache_data['market_data'])}\n")
        output.write(f"Data Range: {cache_data['market_data'].index[0].strftime('%Y-%m-%d')} to {cache_data['market_data'].index[-1].strftime('%Y-%m-%d')}\n")
        output.write("\n")
        
        # Trading Signals
        output.write("SECTION 1: TRADING SIGNALS\n")
        output.write("-" * 40 + "\n")
        if not cache_data['signals'].empty:
            cache_data['signals'].to_csv(output, index=False)
        else:
            output.write("No signals generated\n")
        
        output.write("\n\nSECTION 2: TRADE RESULTS\n")
        output.write("-" * 40 + "\n")
        if not cache_data['trades'].empty:
            cache_data['trades'].to_csv(output, index=False)
        else:
            output.write("No trades executed\n")
        
        output.write("\n\nSECTION 3: FUTURE PREDICTIONS\n")
        output.write("-" * 40 + "\n")
        if cache_data['predictions']:
            predictions_df = pd.DataFrame(cache_data['predictions'])
            predictions_df.to_csv(output, index=False)
        else:
            output.write("No predictions available\n")
        
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_complete_analysis_{currency}.csv'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Professional EMA Alert Candle Strategy API v2.0 is running',
        'features': [
            '10-year historical data support',
            'Multi-currency display',
            'Intelligent future predictions',
            'Advanced trend analysis',
            'Professional charting'
        ]
    })

@app.route('/test')
def test():
    try:
        # Test with different data periods
        test_symbol = 'AAPL'
        test_data_1y = fetch_extended_data(test_symbol, '1y')
        test_data_5y = fetch_extended_data(test_symbol, '5y')
        test_data_10y = fetch_extended_data(test_symbol, '10y')
        
        return jsonify({
            'status': 'ok',
            'test_results': {
                '1_year_data_points': len(test_data_1y) if test_data_1y is not None else 0,
                '5_year_data_points': len(test_data_5y) if test_data_5y is not None else 0,
                '10_year_data_points': len(test_data_10y) if test_data_10y is not None else 0,
                'latest_price': f"${test_data_1y['Close'].iloc[-1]:.2f}" if test_data_1y is not None else "N/A"
            },
            'message': 'All systems operational - Extended data fetching working properly'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
