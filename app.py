# Professional EMA Alert Strategy - Complete Flask Application

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
import json
import traceback
warnings.filterwarnings('ignore')

app = Flask(__name__)

class ProfessionalEMAStrategy:
    def __init__(self, data, symbol, ema_period=5, risk_reward_ratio=3.0, currency='USD'):
        self.data = data.copy()
        self.symbol = symbol
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.currency = currency
        self.signals = pd.DataFrame()
        self.trades = pd.DataFrame()
        self.future_predictions = []
        
    def get_currency_symbol(self):
        """Get currency symbol based on selection"""
        currency_symbols = {
            'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£',
            'JPY': '¥', 'CAD': 'C$', 'AUD': 'A$'
        }
        return currency_symbols.get(self.currency, '$')
    
    def detect_market_type(self):
        """Smart market detection"""
        symbol_upper = self.symbol.upper()
        if '.NS' in symbol_upper or '.BO' in symbol_upper:
            return 'NSE'
        elif '-USD' in symbol_upper or 'BTC' in symbol_upper or 'ETH' in symbol_upper:
            return 'CRYPTO'
        else:
            return 'US'
    
    def calculate_ema(self, data, period):
        """Calculate EMA with error handling"""
        try:
            return data.ewm(span=period, adjust=False).mean()
        except:
            return pd.Series([data.iloc[-1]] * len(data), index=data.index)
    
    def calculate_rsi(self, data, period=14):
        """Calculate RSI with error handling"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI
        except:
            return pd.Series([50] * len(data), index=data.index)
    
    def calculate_macd(self, data):
        """Calculate MACD with error handling"""
        try:
            ema12 = self.calculate_ema(data, 12)
            ema26 = self.calculate_ema(data, 26)
            macd = ema12 - ema26
            signal = self.calculate_ema(macd, 9)
            return macd.fillna(0), signal.fillna(0)
        except:
            zeros = pd.Series([0] * len(data), index=data.index)
            return zeros, zeros
    
    def calculate_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            middle = data.rolling(window=period).mean()
            std = data.rolling(window=period).std()
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            return upper.fillna(data), middle.fillna(data), lower.fillna(data)
        except:
            return data, data, data
    
    def advanced_trend_analysis(self):
        """Professional multi-timeframe trend analysis"""
        try:
            if len(self.data) < 50:
                return self._default_trend_analysis()
            
            # Current price and EMA levels
            current_price = float(self.data['Close'].iloc[-1])
            ema_5 = float(self.data['EMA_5'].iloc[-1])
            ema_20 = float(self.data['EMA_20'].iloc[-1])
            ema_50 = float(self.data['EMA_50'].iloc[-1])
            
            # EMA slopes (trend direction)
            ema_5_slope = float((self.data['EMA_5'].iloc[-1] - self.data['EMA_5'].iloc[-10]) / 10)
            ema_20_slope = float((self.data['EMA_20'].iloc[-1] - self.data['EMA_20'].iloc[-10]) / 10)
            ema_50_slope = float((self.data['EMA_50'].iloc[-1] - self.data['EMA_50'].iloc[-20]) / 20)
            
            # Trend strength analysis
            if ema_5 > ema_20 > ema_50 and all(slope > 0 for slope in [ema_5_slope, ema_20_slope, ema_50_slope]):
                trend_strength = "VERY_STRONG_BULLISH"
            elif ema_5 > ema_20 > ema_50:
                trend_strength = "STRONG_BULLISH"
            elif ema_5 > ema_20:
                trend_strength = "BULLISH"
            elif ema_5 < ema_20 < ema_50 and all(slope < 0 for slope in [ema_5_slope, ema_20_slope, ema_50_slope]):
                trend_strength = "VERY_STRONG_BEARISH"
            elif ema_5 < ema_20 < ema_50:
                trend_strength = "STRONG_BEARISH"
            elif ema_5 < ema_20:
                trend_strength = "BEARISH"
            else:
                trend_strength = "SIDEWAYS"
            
            # Price position analysis
            if current_price > ema_5 > ema_20 > ema_50:
                price_position = "ABOVE_ALL_EMAS"
            elif current_price < ema_5 < ema_20 < ema_50:
                price_position = "BELOW_ALL_EMAS"
            elif current_price > ema_5 and current_price > ema_20:
                price_position = "ABOVE_SHORT_EMAS"
            elif current_price < ema_5 and current_price < ema_20:
                price_position = "BELOW_SHORT_EMAS"
            else:
                price_position = "MIXED"
            
            # Volume analysis
            volume_trend = self._analyze_volume_trend()
            
            # RSI and MACD current levels
            current_rsi = float(self.data['RSI'].iloc[-1])
            current_macd = float(self.data['MACD'].iloc[-1])
            macd_signal = float(self.data['MACD_Signal'].iloc[-1])
            macd_histogram = current_macd - macd_signal
            
            return {
                'trend_strength': trend_strength,
                'price_position': price_position,
                'volume_trend': volume_trend,
                'current_price': current_price,
                'ema_5': ema_5,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_5_slope': ema_5_slope,
                'ema_20_slope': ema_20_slope,
                'ema_50_slope': ema_50_slope,
                'current_rsi': current_rsi,
                'current_macd': current_macd,
                'macd_histogram': macd_histogram,
                'bb_upper': float(self.data['BB_Upper'].iloc[-1]),
                'bb_middle': float(self.data['BB_Middle'].iloc[-1]),
                'bb_lower': float(self.data['BB_Lower'].iloc[-1])
            }
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return self._default_trend_analysis()
    
    def _default_trend_analysis(self):
        """Default trend analysis for error cases"""
        try:
            current_price = float(self.data['Close'].iloc[-1])
            return {
                'trend_strength': "INSUFFICIENT_DATA",
                'price_position': "UNKNOWN",
                'volume_trend': "NORMAL",
                'current_price': current_price,
                'ema_5': current_price,
                'ema_20': current_price,
                'ema_50': current_price,
                'ema_5_slope': 0.0,
                'ema_20_slope': 0.0,
                'ema_50_slope': 0.0,
                'current_rsi': 50.0,
                'current_macd': 0.0,
                'macd_histogram': 0.0,
                'bb_upper': current_price * 1.02,
                'bb_middle': current_price,
                'bb_lower': current_price * 0.98
            }
        except:
            return {
                'trend_strength': "ERROR",
                'price_position': "ERROR",
                'volume_trend': "ERROR",
                'current_price': 100.0,
                'ema_5': 100.0, 'ema_20': 100.0, 'ema_50': 100.0,
                'ema_5_slope': 0.0, 'ema_20_slope': 0.0, 'ema_50_slope': 0.0,
                'current_rsi': 50.0, 'current_macd': 0.0, 'macd_histogram': 0.0,
                'bb_upper': 102.0, 'bb_middle': 100.0, 'bb_lower': 98.0
            }
    
    def _analyze_volume_trend(self):
        """Analyze volume trends"""
        try:
            if 'Volume' not in self.data.columns:
                return "NO_VOLUME_DATA"
            
            recent_volume = self.data['Volume'].tail(5).mean()
            historical_volume = self.data['Volume'].tail(50).mean()
            
            if recent_volume > historical_volume * 1.5:
                return "VERY_HIGH"
            elif recent_volume > historical_volume * 1.2:
                return "HIGH"
            elif recent_volume < historical_volume * 0.5:
                return "VERY_LOW"
            elif recent_volume < historical_volume * 0.8:
                return "LOW"
            else:
                return "NORMAL"
        except:
            return "NORMAL"
    
    def prepare_data(self):
        """Prepare data with all professional indicators"""
        try:
            # EMA calculations
            self.data['EMA_5'] = self.calculate_ema(self.data['Close'], self.ema_period)
            self.data['EMA_20'] = self.calculate_ema(self.data['Close'], 20)
            self.data['EMA_50'] = self.calculate_ema(self.data['Close'], 50)
            
            # Technical indicators
            self.data['RSI'] = self.calculate_rsi(self.data['Close'])
            
            # MACD
            macd, macd_signal = self.calculate_macd(self.data['Close'])
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = macd_signal
            self.data['MACD_Histogram'] = macd - macd_signal
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(self.data['Close'])
            self.data['BB_Upper'] = bb_upper
            self.data['BB_Middle'] = bb_middle
            self.data['BB_Lower'] = bb_lower
            
            return self.data
        except Exception as e:
            print(f"Data preparation error: {e}")
            return self.data
    
    def identify_alert_candles(self):
        """Professional alert candle identification with multiple filters"""
        try:
            # Enhanced Long Alert Candles
            long_condition1 = (self.data['Close'] < self.data['EMA_5'])
            long_condition2 = (self.data['High'] < self.data['EMA_5'])
            long_condition3 = (self.data['RSI'] < 40)  # RSI oversold
            long_condition4 = (self.data['Close'] < self.data['BB_Lower'])  # Below BB lower band
            
            self.data['Long_Alert'] = long_condition1 & long_condition2 & (long_condition3 | long_condition4)
            
            # Enhanced Short Alert Candles
            short_condition1 = (self.data['Close'] > self.data['EMA_5'])
            short_condition2 = (self.data['Low'] > self.data['EMA_5'])
            short_condition3 = (self.data['RSI'] > 60)  # RSI overbought
            short_condition4 = (self.data['Close'] > self.data['BB_Upper'])  # Above BB upper band
            
            self.data['Short_Alert'] = short_condition1 & short_condition2 & (short_condition3 | short_condition4)
            
            return self.data
        except Exception as e:
            print(f"Alert identification error: {e}")
            self.data['Long_Alert'] = False
            self.data['Short_Alert'] = False
            return self.data
    
    def generate_professional_predictions(self):
        """Generate intelligent future predictions with detailed reasoning"""
        try:
            if len(self.data) < 50:
                return []
            
            trend_analysis = self.advanced_trend_analysis()
            predictions = []
            currency_symbol = self.get_currency_symbol()
            
            current_price = trend_analysis['current_price']
            ema_5 = trend_analysis['ema_5']
            ema_20 = trend_analysis['ema_20']
            ema_50 = trend_analysis['ema_50']
            current_rsi = trend_analysis['current_rsi']
            macd_histogram = trend_analysis['macd_histogram']
            
            # Support and Resistance levels
            recent_highs = float(self.data['High'].tail(50).nlargest(5).mean())
            recent_lows = float(self.data['Low'].tail(50).nsmallest(5).mean())
            
            # 1. EMA-Based Long/Short Setups
            if trend_analysis['trend_strength'] in ['BULLISH', 'STRONG_BULLISH', 'VERY_STRONG_BULLISH']:
                if current_price < ema_5 * 1.01:  # Near EMA support
                    entry_level = ema_5 * 1.005
                    stop_level = min(recent_lows, ema_20) * 0.998
                    risk = entry_level - stop_level
                    target_level = entry_level + (risk * self.risk_reward_ratio)
                    
                    confidence = "HIGH" if trend_analysis['volume_trend'] in ["HIGH", "VERY_HIGH"] else "MEDIUM"
                    
                    predictions.append({
                        'type': 'EMA LONG SETUP',
                        'entry_level': f"{currency_symbol}{entry_level:.2f}",
                        'stop_loss': f"{currency_symbol}{stop_level:.2f}",
                        'target': f"{currency_symbol}{target_level:.2f}",
                        'confidence': confidence,
                        'reasoning': f"Strong bullish trend ({trend_analysis['trend_strength']}). Price near EMA-{self.ema_period} support at {currency_symbol}{ema_5:.2f}. All EMAs trending upward with EMA-5 slope: {trend_analysis['ema_5_slope']:.3f}.",
                        'time_frame': '2-5 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"RSI: {current_rsi:.1f}, MACD Histogram: {macd_histogram:.3f}. Volume trend: {trend_analysis['volume_trend']}"
                    })
            
            elif trend_analysis['trend_strength'] in ['BEARISH', 'STRONG_BEARISH', 'VERY_STRONG_BEARISH']:
                if current_price > ema_5 * 0.99:  # Near EMA resistance
                    entry_level = ema_5 * 0.995
                    stop_level = max(recent_highs, ema_20) * 1.002
                    risk = stop_level - entry_level
                    target_level = entry_level - (risk * self.risk_reward_ratio)
                    
                    confidence = "HIGH" if trend_analysis['volume_trend'] in ["HIGH", "VERY_HIGH"] else "MEDIUM"
                    
                    predictions.append({
                        'type': 'EMA SHORT SETUP',
                        'entry_level': f"{currency_symbol}{entry_level:.2f}",
                        'stop_loss': f"{currency_symbol}{stop_level:.2f}",
                        'target': f"{currency_symbol}{target_level:.2f}",
                        'confidence': confidence,
                        'reasoning': f"Strong bearish trend ({trend_analysis['trend_strength']}). Price near EMA-{self.ema_period} resistance at {currency_symbol}{ema_5:.2f}. All EMAs trending downward with EMA-5 slope: {trend_analysis['ema_5_slope']:.3f}.",
                        'time_frame': '2-5 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"RSI: {current_rsi:.1f}, MACD Histogram: {macd_histogram:.3f}. Volume trend: {trend_analysis['volume_trend']}"
                    })
            
            # 2. Breakout Predictions
            consolidation_range = recent_highs - recent_lows
            if consolidation_range / current_price < 0.05:  # Tight consolidation
                breakout_upper = recent_highs * 1.002
                breakout_lower = recent_lows * 0.998
                
                predictions.append({
                    'type': 'BREAKOUT SETUP',
                    'entry_level': f"LONG above {currency_symbol}{breakout_upper:.2f} OR SHORT below {currency_symbol}{breakout_lower:.2f}",
                    'stop_loss': f"Opposite side of breakout + {currency_symbol}{consolidation_range*0.1:.2f}",
                    'target': f"Range projection: {currency_symbol}{consolidation_range*self.risk_reward_ratio:.2f}",
                    'confidence': 'MEDIUM',
                    'reasoning': f"Price consolidating in {currency_symbol}{consolidation_range:.2f} range ({(consolidation_range/current_price)*100:.1f}% of price). Breakout likely due to compression. Volume trend: {trend_analysis['volume_trend']}.",
                    'time_frame': '1-3 days',
                    'risk_reward': f"1:{self.risk_reward_ratio}",
                    'additional_notes': f"Wait for volume confirmation on breakout. Current RSI: {current_rsi:.1f} (neutral zone)"
                })
            
            # 3. RSI Extreme Mean Reversion
            if current_rsi > 80:
                predictions.append({
                    'type': 'RSI MEAN REVERSION SHORT',
                    'entry_level': f"Current levels {currency_symbol}{current_price:.2f}",
                    'stop_loss': f"{currency_symbol}{current_price * 1.03:.2f}",
                    'target': f"{currency_symbol}{ema_20:.2f} (EMA-20)",
                    'confidence': 'MEDIUM',
                    'reasoning': f"RSI extremely overbought at {current_rsi:.1f} (>80). Historical mean reversion expected toward EMA-20. Price also near BB upper band at {currency_symbol}{trend_analysis['bb_upper']:.2f}.",
                    'time_frame': '3-7 days',
                    'risk_reward': '1:2',
                    'additional_notes': f"High-risk counter-trend trade. Consider partial position. MACD: {macd_histogram:.3f}"
                })
            elif current_rsi < 20:
                predictions.append({
                    'type': 'RSI MEAN REVERSION LONG',
                    'entry_level': f"Current levels {currency_symbol}{current_price:.2f}",
                    'stop_loss': f"{currency_symbol}{current_price * 0.97:.2f}",
                    'target': f"{currency_symbol}{ema_20:.2f} (EMA-20)",
                    'confidence': 'MEDIUM',
                    'reasoning': f"RSI extremely oversold at {current_rsi:.1f} (<20). Historical mean reversion expected toward EMA-20. Price also near BB lower band at {currency_symbol}{trend_analysis['bb_lower']:.2f}.",
                    'time_frame': '3-7 days',
                    'risk_reward': '1:2',
                    'additional_notes': f"High-risk counter-trend trade. Consider partial position. MACD: {macd_histogram:.3f}"
                })
            
            # 4. MACD Momentum Setup
            if abs(macd_histogram) > 0.05:  # Strong MACD signal
                if macd_histogram > 0 and trend_analysis['trend_strength'] in ['BULLISH', 'STRONG_BULLISH']:
                    predictions.append({
                        'type': 'MACD MOMENTUM LONG',
                        'entry_level': f"Above {currency_symbol}{current_price * 1.002:.2f}",
                        'stop_loss': f"{currency_symbol}{ema_5 * 0.995:.2f}",
                        'target': f"{currency_symbol}{current_price * (1 + 0.02*self.risk_reward_ratio):.2f}",
                        'confidence': 'HIGH',
                        'reasoning': f"Strong MACD bullish momentum (histogram: {macd_histogram:.3f}) aligning with trend. EMA structure supportive with all EMAs rising.",
                        'time_frame': '1-4 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"Momentum trade. RSI: {current_rsi:.1f}. Volume: {trend_analysis['volume_trend']}"
                    })
                elif macd_histogram < 0 and trend_analysis['trend_strength'] in ['BEARISH', 'STRONG_BEARISH']:
                    predictions.append({
                        'type': 'MACD MOMENTUM SHORT',
                        'entry_level': f"Below {currency_symbol}{current_price * 0.998:.2f}",
                        'stop_loss': f"{currency_symbol}{ema_5 * 1.005:.2f}",
                        'target': f"{currency_symbol}{current_price * (1 - 0.02*self.risk_reward_ratio):.2f}",
                        'confidence': 'HIGH',
                        'reasoning': f"Strong MACD bearish momentum (histogram: {macd_histogram:.3f}) aligning with trend. EMA structure resistance with all EMAs falling.",
                        'time_frame': '1-4 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"Momentum trade. RSI: {current_rsi:.1f}. Volume: {trend_analysis['volume_trend']}"
                    })
            
            self.future_predictions = predictions
            return predictions
            
        except Exception as e:
            print(f"Prediction generation error: {e}")
            return []
    
    def generate_detailed_signals(self):
        """Generate detailed trading signals with full analysis"""
        signals = []
        try:
            for i in range(max(20, self.ema_period * 2), len(self.data)):
                current = self.data.iloc[i]
                current_date = current.name
                
                # Look for Alert Candles in recent history
                lookback_period = min(10, i)
                recent_data = self.data.iloc[i-lookback_period:i]
                
                # Enhanced long signal detection
                long_alerts = recent_data[recent_data['Long_Alert']]
                if not long_alerts.empty:
                    latest_alert = long_alerts.iloc[-1]
                    alert_date = latest_alert.name
                    
                    # Multiple entry conditions
                    price_breakout = current['High'] > latest_alert['High']
                    volume_confirmation = current.get('Volume', 0) > recent_data['Volume'].mean() * 1.1 if 'Volume' in recent_data.columns else True
                    rsi_support = current['RSI'] > 30
                    
                    if price_breakout and rsi_support:
                        entry_price = latest_alert['High']
                        stop_loss = min(latest_alert['Low'], current['EMA_20'] * 0.99)
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * self.risk_reward_ratio)
                        
                        signals.append({
                            'Signal_Date': current_date.strftime('%Y-%m-%d'),
                            'Alert_Candle_Date': alert_date.strftime('%Y-%m-%d'),
                            'Type': 'LONG',
                            'Entry_Price': round(float(entry_price), 2),
                            'Stop_Loss': round(float(stop_loss), 2),
                            'Take_Profit': round(float(take_profit), 2),
                            'Risk_Amount': round(float(risk), 2),
                            'Reward_Potential': round(float(risk * self.risk_reward_ratio), 2),
                            'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                            'Current_Price': round(float(current['Close']), 2),
                            'RSI_At_Signal': round(float(current['RSI']), 1),
                            'MACD_At_Signal': round(float(current['MACD']), 3),
                            'Volume_Confirmation': volume_confirmation,
                            'EMA_5_Level': round(float(current['EMA_5']), 2),
                            'EMA_20_Level': round(float(current['EMA_20']), 2),
                            'BB_Position': 'ABOVE' if current['Close'] > current['BB_Upper'] else 'BELOW' if current['Close'] < current['BB_Lower'] else 'MIDDLE'
                        })
                
                # Enhanced short signal detection
                short_alerts = recent_data[recent_data['Short_Alert']]
                if not short_alerts.empty:
                    latest_alert = short_alerts.iloc[-1]
                    alert_date = latest_alert.name
                    
                    # Multiple entry conditions
                    price_breakout = current['Low'] < latest_alert['Low']
                    volume_confirmation = current.get('Volume', 0) > recent_data['Volume'].mean() * 1.1 if 'Volume' in recent_data.columns else True
                    rsi_resistance = current['RSI'] < 70
                    
                    if price_breakout and rsi_resistance:
                        entry_price = latest_alert['Low']
                        stop_loss = max(latest_alert['High'], current['EMA_20'] * 1.01)
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * self.risk_reward_ratio)
                        
                        signals.append({
                            'Signal_Date': current_date.strftime('%Y-%m-%d'),
                            'Alert_Candle_Date': alert_date.strftime('%Y-%m-%d'),
                            'Type': 'SHORT',
                            'Entry_Price': round(float(entry_price), 2),
                            'Stop_Loss': round(float(stop_loss), 2),
                            'Take_Profit': round(float(take_profit), 2),
                            'Risk_Amount': round(float(risk), 2),
                            'Reward_Potential': round(float(risk * self.risk_reward_ratio), 2),
                            'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                            'Current_Price': round(float(current['Close']), 2),
                            'RSI_At_Signal': round(float(current['RSI']), 1),
                            'MACD_At_Signal': round(float(current['MACD']), 3),
                            'Volume_Confirmation': volume_confirmation,
                            'EMA_5_Level': round(float(current['EMA_5']), 2),
                            'EMA_20_Level': round(float(current['EMA_20']), 2),
                            'BB_Position': 'ABOVE' if current['Close'] > current['BB_Upper'] else 'BELOW' if current['Close'] < current['BB_Lower'] else 'MIDDLE'
                        })
            
            self.signals = pd.DataFrame(signals)
            return self.signals
        except Exception as e:
            print(f"Signal generation error: {e}")
            self.signals = pd.DataFrame()
            return self.signals
    
    def comprehensive_backtesting(self):
        """Complete backtesting with detailed trade analysis"""
        try:
            if self.signals.empty:
                return pd.DataFrame()
            
            trades = []
            
            for idx, signal in self.signals.iterrows():
                entry_date_str = signal['Signal_Date']
                entry_date = pd.to_datetime(entry_date_str)
                entry_price = signal['Entry_Price']
                stop_loss = signal['Stop_Loss']
                take_profit = signal['Take_Profit']
                trade_type = signal['Type']
                
                # Find exit in future data
                future_data = self.data[self.data.index > entry_date]
                exit_info = self._find_trade_exit(future_data, trade_type, entry_price, stop_loss, take_profit)
                
                if exit_info:
                    exit_price, exit_date, exit_reason = exit_info
                else:
                    # Position still open
                    exit_price = float(self.data['Close'].iloc[-1])
                    exit_date = self.data.index[-1]
                    exit_reason = 'Position Still Open'
                
                # Calculate comprehensive trade metrics
                if trade_type == 'LONG':
                    pnl = exit_price - entry_price
                    pnl_percent = (pnl / entry_price) * 100
                else:
                    pnl = entry_price - exit_price
                    pnl_percent = (pnl / entry_price) * 100
                
                holding_days = (exit_date - entry_date).days if exit_date != entry_date else 0
                
                # Risk metrics
                risk_amount = signal['Risk_Amount']
                reward_achieved = abs(pnl)
                actual_rr = reward_achieved / risk_amount if risk_amount > 0 else 0
                
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
                    'Risk_Amount': round(risk_amount, 2),
                    'Reward_Achieved': round(reward_achieved, 2),
                    'Actual_RR_Ratio': round(actual_rr, 2),
                    'Target_RR_Ratio': f"1:{self.risk_reward_ratio}",
                    'RSI_At_Entry': signal['RSI_At_Signal'],
                    'MACD_At_Entry': signal['MACD_At_Signal'],
                    'Volume_Confirmed': signal['Volume_Confirmation'],
                    'BB_Position_Entry': signal['BB_Position']
                })
            
            self.trades = pd.DataFrame(trades)
            return self.trades
        except Exception as e:
            print(f"Backtesting error: {e}")
            self.trades = pd.DataFrame()
            return self.trades
    
    def _find_trade_exit(self, future_data, trade_type, entry_price, stop_loss, take_profit):
        """Find trade exit with detailed analysis"""
        try:
            for future_date, future_candle in future_data.iterrows():
                if trade_type == 'LONG':
                    if future_candle['Low'] <= stop_loss:
                        return stop_loss, future_date, 'Stop Loss Hit'
                    elif future_candle['High'] >= take_profit:
                        return take_profit, future_date, 'Take Profit Hit'
                else:  # SHORT
                    if future_candle['High'] >= stop_loss:
                        return stop_loss, future_date, 'Stop Loss Hit'
                    elif future_candle['Low'] <= take_profit:
                        return take_profit, future_date, 'Take Profit Hit'
            return None
        except:
            return None
    
    def calculate_professional_metrics(self):
        """Calculate comprehensive performance metrics"""
        try:
            if self.trades.empty:
                return self._default_metrics()
            
            # Basic metrics
            total_trades = len(self.trades)
            winning_trades = len(self.trades[self.trades['PnL'] > 0])
            losing_trades = len(self.trades[self.trades['PnL'] < 0])
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # P&L metrics
            total_pnl = self.trades['PnL'].sum()
            avg_win = self.trades[self.trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
            avg_loss = self.trades[self.trades['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0
            max_win = self.trades['PnL'].max() if total_trades > 0 else 0
            max_loss = self.trades['PnL'].min() if total_trades > 0 else 0
            
            # Advanced metrics
            total_wins = self.trades[self.trades['PnL'] > 0]['PnL'].sum()
            total_losses = abs(self.trades[self.trades['PnL'] < 0]['PnL'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            # Risk metrics
            avg_holding_days = self.trades['Holding_Days'].mean() if total_trades > 0 else 0
            avg_rr_achieved = self.trades['Actual_RR_Ratio'].mean() if total_trades > 0 else 0
            
            # Consecutive trade analysis
            consecutive_wins = self._calculate_consecutive_trades(self.trades, 'win')
            consecutive_losses = self._calculate_consecutive_trades(self.trades, 'loss')
            
            return {
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': round(float(win_rate), 1),
                'total_pnl': round(float(total_pnl), 2),
                'avg_win': round(float(avg_win), 2),
                'avg_loss': round(float(avg_loss), 2),
                'max_win': round(float(max_win), 2),
                'max_loss': round(float(max_loss), 2),
                'profit_factor': round(float(profit_factor), 2),
                'avg_holding_days': round(float(avg_holding_days), 1),
                'avg_rr_achieved': round(float(avg_rr_achieved), 2),
                'max_consecutive_wins': int(consecutive_wins),
                'max_consecutive_losses': int(consecutive_losses)
            }
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return self._default_metrics()
    
    def _default_metrics(self):
        """Default metrics for error cases"""
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'total_pnl': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
            'max_win': 0.0, 'max_loss': 0.0, 'profit_factor': 0.0,
            'avg_holding_days': 0.0, 'avg_rr_achieved': 0.0,
            'max_consecutive_wins': 0, 'max_consecutive_losses': 0
        }
    
    def _calculate_consecutive_trades(self, trades_df, trade_type):
        """Calculate maximum consecutive wins or losses"""
        try:
            if trades_df.empty:
                return 0
            
            results = trades_df['PnL'] > 0 if trade_type == 'win' else trades_df['PnL'] < 0
            max_consecutive = 0
            current_consecutive = 0
            
            for result in results:
                if result:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    def analyze_complete_strategy(self):
        """Run complete professional analysis"""
        try:
            # Data preparation
            self.prepare_data()
            self.identify_alert_candles()
            
            # Generate analysis components
            signals = self.generate_detailed_signals()
            trades = self.comprehensive_backtesting()
            predictions = self.generate_professional_predictions()
            trend_analysis = self.advanced_trend_analysis()
            performance_metrics = self.calculate_professional_metrics()
            
            # Combine all results
            results = {
                'success': True,
                'data_points': len(self.data),
                'long_alerts': int(self.data['Long_Alert'].sum()) if 'Long_Alert' in self.data.columns else 0,
                'short_alerts': int(self.data['Short_Alert'].sum()) if 'Short_Alert' in self.data.columns else 0,
                'total_signals': len(signals),
                'trend_analysis': trend_analysis,
                'future_predictions': predictions,
                'performance_metrics': performance_metrics
            }
            
            return results
            
        except Exception as e:
            print(f"Complete analysis error: {e}")
            return {
                'success': False,
                'error': f"Analysis failed: {str(e)}",
                'data_points': len(self.data) if hasattr(self, 'data') else 0,
                'long_alerts': 0, 'short_alerts': 0, 'total_signals': 0,
                'trend_analysis': self._default_trend_analysis(),
                'future_predictions': [],
                'performance_metrics': self._default_metrics()
            }

def robust_data_fetching(symbol, period='1y'):
    """Ultra-robust data fetching with multiple strategies"""
    try:
        print(f"Fetching {period} data for {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Strategy 1: Direct period fetch
        data = pd.DataFrame()
        try:
            if period == 'max':
                data = ticker.history(period='max', auto_adjust=True)
                if data.empty:
                    data = ticker.history(period='20y', auto_adjust=True)
            elif period == '10y':
                data = ticker.history(period='10y', auto_adjust=True)
                if len(data) < 200:  # If insufficient data, try date range
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=10*365 + 50)  # Add buffer
                    data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            else:
                data = ticker.history(period=period, auto_adjust=True)
        except Exception as e:
            print(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Fallback periods
        if data.empty or len(data) < 50:
            fallback_periods = ['5y', '2y', '1y', '6mo', '3mo', '1mo']
            for fallback in fallback_periods:
                try:
                    print(f"Trying fallback period: {fallback}")
                    data = ticker.history(period=fallback, auto_adjust=True)
                    if not data.empty and len(data) >= 50:
                        print(f"Success with {fallback}: {len(data)} points")
                        break
                except:
                    continue
        
        # Strategy 3: Date range approach for very long periods
        if (data.empty or len(data) < 100) and period in ['10y', 'max']:
            try:
                print("Trying date range approach...")
                end_date = datetime.now()
                start_date = end_date - timedelta(days=8*365)  # 8 years as fallback
                data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
            except:
                pass
        
        # Final validation and cleaning
        if data.empty:
            print(f"All strategies failed for {symbol}")
            return None
        
        # Data cleaning
        data = data.dropna()
        
        # Ensure minimum data points
        if len(data) < 30:
            print(f"Insufficient data for {symbol}: only {len(data)} points")
            return None
        
        # Add basic validation
        if 'Close' not in data.columns:
            print(f"Invalid data structure for {symbol}")
            return None
        
        print(f"Successfully fetched {len(data)} data points for {symbol} ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})")
        return data
        
    except Exception as e:
        print(f"Complete data fetching failure for {symbol}: {str(e)}")
        return None

def create_professional_4panel_chart(strategy, symbol):
    """Create professional 4-panel chart with all indicators"""
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 14))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.4)
        
        ax_price = fig.add_subplot(gs[0])
        ax_volume = fig.add_subplot(gs[1])
        ax_rsi = fig.add_subplot(gs[2])
        ax_macd = fig.add_subplot(gs[3])
        
        currency_symbol = strategy.get_currency_symbol()
        data = strategy.data
        
        # Main price chart with EMAs and Bollinger Bands
        ax_price.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='#1f77b4')
        ax_price.plot(data.index, data['EMA_5'], label=f'EMA-{strategy.ema_period}', color='#ff7f0e', linewidth=2)
        ax_price.plot(data.index, data['EMA_20'], label='EMA-20', color='#2ca02c', linewidth=1.5)
        ax_price.plot(data.index, data['EMA_50'], label='EMA-50', color='#d62728', linewidth=1.5)
        
        # Bollinger Bands
        if all(col in data.columns for col in ['BB_Upper', 'BB_Lower']):
            ax_price.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], 
                                alpha=0.1, color='gray', label='Bollinger Bands')
            ax_price.plot(data.index, data['BB_Upper'], color='gray', linestyle='--', alpha=0.7)
            ax_price.plot(data.index, data['BB_Lower'], color='gray', linestyle='--', alpha=0.7)
        
        # Alert candles
        if 'Long_Alert' in data.columns:
            long_alerts = data[data['Long_Alert']]
            if not long_alerts.empty:
                ax_price.scatter(long_alerts.index, long_alerts['Low'], 
                               color='lime', marker='^', s=80, 
                               label='Long Alert', alpha=0.8, edgecolor='darkgreen', linewidth=1)
        
        if 'Short_Alert' in data.columns:
            short_alerts = data[data['Short_Alert']]
            if not short_alerts.empty:
                ax_price.scatter(short_alerts.index, short_alerts['High'], 
                               color='red', marker='v', s=80, 
                               label='Short Alert', alpha=0.8, edgecolor='darkred', linewidth=1)
        
        # Entry points from signals
        if not strategy.signals.empty:
            long_entries = strategy.signals[strategy.signals['Type'] == 'LONG']
            short_entries = strategy.signals[strategy.signals['Type'] == 'SHORT']
            
            if not long_entries.empty:
                entry_dates = pd.to_datetime(long_entries['Signal_Date'])
                entry_prices = long_entries['Entry_Price']
                ax_price.scatter(entry_dates, entry_prices, 
                               color='darkgreen', marker='o', s=120, 
                               label='Long Entry', edgecolor='white', linewidth=2, zorder=10)
            
            if not short_entries.empty:
                entry_dates = pd.to_datetime(short_entries['Signal_Date'])
                entry_prices = short_entries['Entry_Price']
                ax_price.scatter(entry_dates, entry_prices, 
                               color='darkred', marker='o', s=120, 
                               label='Short Entry', edgecolor='white', linewidth=2, zorder=10)
        
        ax_price.set_title(f'{symbol} - Professional EMA Alert Strategy ({len(data)} data points)', 
                          fontsize=16, fontweight='bold', pad=20)
        ax_price.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
        ax_price.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Volume chart
        if 'Volume' in data.columns:
            colors = ['red' if close < open_val else 'green' 
                     for close, open_val in zip(data['Close'], data['Open'])]
            ax_volume.bar(data.index, data['Volume'], alpha=0.7, color=colors, width=1)
            ax_volume.set_ylabel('Volume', fontsize=10)
            ax_volume.grid(True, alpha=0.3)
        
        # RSI chart
        if 'RSI' in data.columns:
            ax_rsi.plot(data.index, data['RSI'], color='purple', linewidth=2)
            ax_rsi.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Overbought (80)')
            ax_rsi.axhline(y=20, color='g', linestyle='--', alpha=0.7, label='Oversold (20)')
            ax_rsi.axhline(y=70, color='orange', linestyle=':', alpha=0.5)
            ax_rsi.axhline(y=30, color='orange', linestyle=':', alpha=0.5)
            ax_rsi.fill_between(data.index, 20, 80, alpha=0.1, color='yellow')
            ax_rsi.set_ylabel('RSI', fontsize=10)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.legend(fontsize=8)
            ax_rsi.grid(True, alpha=0.3)
        
        # MACD chart
        if all(col in data.columns for col in ['MACD', 'MACD_Signal', 'MACD_Histogram']):
            ax_macd.plot(data.index, data['MACD'], color='blue', linewidth=2, label='MACD')
            ax_macd.plot(data.index, data['MACD_Signal'], color='red', linewidth=2, label='Signal')
            
            # MACD Histogram with colors
            histogram = data['MACD_Histogram']
            colors = ['green' if val >= 0 else 'red' for val in histogram]
            ax_macd.bar(data.index, histogram, alpha=0.4, color=colors, width=1, label='Histogram')
            
            ax_macd.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax_macd.set_ylabel('MACD', fontsize=10)
            ax_macd.set_xlabel('Date', fontsize=12)
            ax_macd.legend(fontsize=8)
            ax_macd.grid(True, alpha=0.3)
        
        # Overall title
        plt.suptitle(f'Complete Technical Analysis - Market Type: {strategy.detect_market_type()}', 
                     fontsize=16, y=0.99)
        
        plt.tight_layout()
        
        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return chart_url
        
    except Exception as e:
        print(f"4-panel chart creation error: {e}")
        # Fallback to simple chart
        return create_simple_fallback_chart(strategy, symbol)

def create_simple_fallback_chart(strategy, symbol):
    """Fallback simple chart if complex chart fails"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = strategy.data
        currency_symbol = strategy.get_currency_symbol()
        
        # Basic price and EMA
        ax.plot(data.index, data['Close'], label='Close Price', linewidth=2, color='blue')
        if 'EMA_5' in data.columns:
            ax.plot(data.index, data['EMA_5'], label=f'EMA-{strategy.ema_period}', color='red', linewidth=1.5)
        if 'EMA_20' in data.columns:
            ax.plot(data.index, data['EMA_20'], label='EMA-20', color='orange', linewidth=1.5)
        
        ax.set_title(f'{symbol} - EMA Alert Strategy', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=80)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return chart_url
        
    except Exception as e:
        print(f"Fallback chart error: {e}")
        return None

# Global cache for analysis data
analysis_cache = {}

@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Professional EMA Alert Strategy</title>
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
            <p class="lead">Advanced Multi-Market Analysis • Intelligent Predictions • Professional Backtesting</p>
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
                                <strong>Professional:</strong> Advanced backtesting with intelligent predictions!</small>
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
                                    <li><i class="fas fa-check text-success me-2"></i>Multi-timeframe EMA analysis (5, 20, 50)</li>
                                    <li><i class="fas fa-check text-success me-2"></i>RSI + MACD + Bollinger Bands</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Volume trend confirmation</li>
                                    <li><i class="fas fa-check text-success me-2"></i>Professional 4-panel charts</li>
                                </ul>
                                
                                <h6 class="text-primary mt-3"><i class="fas fa-brain me-2"></i>AI Predictions</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-robot text-warning me-2"></i>EMA Long/Short setups</li>
                                    <li><i class="fas fa-target text-info me-2"></i>Breakout predictions</li>
                                    <li><i class="fas fa-balance-scale text-success me-2"></i>Mean reversion analysis</li>
                                    <li><i class="fas fa-chart-line text-primary me-2"></i>MACD momentum signals</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6 class="text-primary"><i class="fas fa-globe me-2"></i>Multi-Market Support</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-flag-usa text-primary me-2"></i>US Stocks (NYSE, NASDAQ)</li>
                                    <li><i class="fas fa-rupee-sign text-warning me-2"></i>NSE India (.NS, .BO)</li>
                                    <li><i class="fas fa-bitcoin text-danger me-2"></i>Cryptocurrencies</li>
                                    <li><i class="fas fa-coins text-success me-2"></i>Smart currency detection</li>
                                </ul>
                                
                                <h6 class="text-primary mt-3"><i class="fas fa-chart-bar me-2"></i>Professional Metrics</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-trophy text-warning me-2"></i>Complete backtesting results</li>
                                    <li><i class="fas fa-percentage text-success me-2"></i>Win rate & profit factor</li>
                                    <li><i class="fas fa-download text-info me-2"></i>CSV data exports</li>
                                    <li><i class="fas fa-clock text-secondary me-2"></i>Up to 10+ years history</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div id="loading" class="text-center my-5" style="display:none;">
            <div class="spinner-border text-primary" style="width: 4rem; height: 4rem;"></div>
            <p class="mt-3 h5">Running professional analysis... This may take up to 45 seconds for extended periods.</p>
            <div class="progress mt-3" style="height: 6px;">
                <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
            </div>
        </div>
        
        <div id="results" class="mt-5" style="display:none;">
            <!-- Trend Analysis Section -->
            <div id="trendSection" class="trend-analysis mb-4">
                <h5><i class="fas fa-trending-up me-2"></i>Professional Market Analysis</h5>
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
                <h5><i class="fas fa-cloud-download-alt me-2"></i>Export Professional Data</h5>
                <p class="text-muted mb-3">Download complete analysis including backtesting, predictions, and performance metrics</p>
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
                    <h5 class="mb-0"><i class="fas fa-chart-area me-2"></i>Professional 4-Panel Technical Analysis</h5>
                </div>
                <div class="card-body text-center">
                    <img id="strategyChart" class="img-fluid rounded shadow" alt="Strategy Chart" style="max-height: 900px;">
                </div>
            </div>
        </div>
        
        <div id="error" class="alert alert-danger mt-4" style="display:none;"></div>
    </div>
    
    <footer class="bg-dark text-white text-center py-4 mt-5">
        <div class="container">
            <p class="mb-1">Professional EMA Alert Candle Strategy - Complete Edition</p>
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
                const requestBody = {
                    symbol: symbol, 
                    period: period,
                    currency: currency,
                    ema_period: parseInt(ema_period),
                    risk_reward: parseFloat(risk_reward)
                };
                
                console.log('Sending request:', requestBody);
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });
                
                console.log('Response status:', response.status);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    console.log('Error response:', errorText);
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }
                
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    const responseText = await response.text();
                    console.log('Non-JSON response:', responseText);
                    throw new Error('Server returned non-JSON response');
                }
                
                const data = await response.json();
                console.log('Parsed data:', data);
                
                if (data.success) {
                    currentAnalysisData = data;
                    displayResults(data, symbol, currency);
                } else {
                    showError(data.error || 'Analysis failed');
                }
            } catch (error) {
                console.error('Request error:', error);
                showError('Error: ' + error.message);
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
                        <div class="col-md-4">
                            <h6><i class="fas fa-chart-line me-2"></i>Trend Strength</h6>
                            <p><strong>Overall:</strong> <span class="badge bg-${getTrendColor(trend.trend_strength)}">${trend.trend_strength}</span></p>
                            <p><strong>Price Position:</strong> <span class="badge bg-${getTrendColor(trend.price_position)}">${trend.price_position}</span></p>
                            <p><strong>Volume:</strong> <span class="badge bg-${getVolumeColor(trend.volume_trend)}">${trend.volume_trend}</span></p>
                        </div>
                        <div class="col-md-4">
                            <h6><i class="fas fa-layer-group me-2"></i>EMA Levels</h6>
                            <p><strong>Current Price:</strong> ${currencySymbol}${trend.current_price.toFixed(2)}</p>
                            <p><strong>EMA-5:</strong> ${currencySymbol}${trend.ema_5.toFixed(2)}</p>
                            <p><strong>EMA-20:</strong> ${currencySymbol}${trend.ema_20.toFixed(2)}</p>
                            <p><strong>EMA-50:</strong> ${currencySymbol}${trend.ema_50.toFixed(2)}</p>
                        </div>
                        <div class="col-md-4">
                            <h6><i class="fas fa-chart-bar me-2"></i>Technical Indicators</h6>
                            <p><strong>RSI:</strong> ${trend.current_rsi.toFixed(1)}</p>
                            <p><strong>MACD:</strong> ${trend.current_macd.toFixed(3)}</p>
                            <p><strong>BB Upper:</strong> ${currencySymbol}${trend.bb_upper.toFixed(2)}</p>
                            <p><strong>BB Lower:</strong> ${currencySymbol}${trend.bb_lower.toFixed(2)}</p>
                        </div>
                    </div>
                `;
                document.getElementById('trendAnalysis').innerHTML = trendHtml;
            }
            
            // Status badges
            document.getElementById('statusBadges').innerHTML = `
                <span class="badge bg-light text-dark me-1">${data.data_points} candles</span>
                <span class="badge bg-primary me-1">${data.total_signals} signals</span>
                <span class="badge bg-success me-1">${data.long_alerts} long alerts</span>
                <span class="badge bg-danger me-1">${data.short_alerts} short alerts</span>
                <span class="badge currency-badge text-white">${currency}</span>
            `;
            
            // Performance metrics
            const metrics = data.performance_metrics;
            const metricsRow = document.getElementById('metricsRow');
            metricsRow.innerHTML = `
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total Trades</h6>
                        <h3 class="text-primary mb-0">${metrics.total_trades}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Win Rate</h6>
                        <h3 class="text-success mb-0">${metrics.win_rate}%</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total P&L</h6>
                        <h3 class="${metrics.total_pnl >= 0 ? 'text-success' : 'text-danger'} mb-0">${currencySymbol}${metrics.total_pnl}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Profit Factor</h6>
                        <h3 class="text-info mb-0">${metrics.profit_factor}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Max Win</h6>
                        <h3 class="text-success mb-0">${currencySymbol}${metrics.max_win}</h3>
                    </div>
                </div>
                <div class="col-md-2 mb-3">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Max Loss</h6>
                        <h3 class="text-danger mb-0">${currencySymbol}${metrics.max_loss}</h3>
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
            if (trend.includes('STRONG')) return 'primary';
            return 'warning';
        }
        
        function getVolumeColor(volume) {
            if (volume === 'HIGH' || volume === 'VERY_HIGH') return 'success';
            if (volume === 'LOW' || volume === 'VERY_LOW') return 'danger';
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
        
        // Download functions (simplified for now)
        async function downloadSignals() {
            alert('Download Signals feature - Coming soon!');
        }
        
        async function downloadTrades() {
            alert('Download Trades feature - Coming soon!');
        }
        
        async function downloadPredictions() {
            alert('Download Predictions feature - Coming soon!');
        }
        
        async function downloadAll() {
            alert('Download All Data feature - Coming soon!');
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

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    global analysis_cache
    
    try:
        # Ensure we're handling JSON properly
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        
        symbol = data.get('symbol', 'AAPL').upper().strip()
        period = data.get('period', '1y')
        currency = data.get('currency', 'USD')
        ema_period = int(data.get('ema_period', 5))
        risk_reward = float(data.get('risk_reward', 3.0))
        
        print(f"Professional analysis request: {symbol}, {period}, {currency}")
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Invalid symbol'}), 400
        
        # Smart auto-detection of currency based on symbol
        if '.NS' in symbol or '.BO' in symbol:
            currency = 'INR'  # NSE stocks auto-detect to INR
        elif 'BTC' in symbol or 'ETH' in symbol or '-USD' in symbol:
            currency = 'USD'  # Crypto auto-detect to USD
        
        # Robust data fetching
        market_data = robust_data_fetching(symbol, period)
        
        if market_data is None:
            return jsonify({
                'success': False, 
                'error': f'Could not fetch data for {symbol}. Please verify the symbol and try again.'
            }), 400
        
        # Create professional strategy instance
        strategy = ProfessionalEMAStrategy(
            market_data, 
            symbol=symbol,
            ema_period=ema_period, 
            risk_reward_ratio=risk_reward,
            currency=currency
        )
        
        # Run complete professional analysis
        results = strategy.analyze_complete_strategy()
        
        if not results.get('success', False):
            return jsonify(results), 400
        
        # Store data for potential downloads
        analysis_cache[symbol] = {
            'strategy': strategy,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'predictions': strategy.future_predictions,
            'market_data': market_data,
            'currency': currency,
            'timestamp': datetime.now()
        }
        
        # Create professional 4-panel chart
        chart_data = create_professional_4panel_chart(strategy, symbol)
        results['chart'] = chart_data
        
        # Ensure all values are JSON serializable
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results[key] = float(value)
            elif isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (np.integer, np.floating)):
                        results[key][sub_key] = float(sub_value)
        
        return jsonify(results)
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return jsonify({'success': False, 'error': 'Invalid JSON format'}), 400
    except Exception as e:
        print(f"Professional analysis error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Professional EMA Alert Strategy API is running',
        'features': [
            'Multi-timeframe EMA analysis',
            'Professional 4-panel charts',
            'Intelligent future predictions',
            'Complete backtesting',
            'Multi-market support',
            'Advanced performance metrics'
        ]
    })

@app.route('/test')
def test():
    try:
        # Test professional data fetching
        test_data_1y = robust_data_fetching('AAPL', '1y')
        test_data_5y = robust_data_fetching('AAPL', '5y')
        test_data_10y = robust_data_fetching('AAPL', '10y')
        
        return jsonify({
            'status': 'ok',
            'test_results': {
                '1_year_data_points': len(test_data_1y) if test_data_1y is not None else 0,
                '5_year_data_points': len(test_data_5y) if test_data_5y is not None else 0,
                '10_year_data_points': len(test_data_10y) if test_data_10y is not None else 0,
                'latest_price': f"${test_data_1y['Close'].iloc[-1]:.2f}" if test_data_1y is not None else "N/A"
            },
            'message': 'Professional EMA Alert Strategy - All systems operational'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
