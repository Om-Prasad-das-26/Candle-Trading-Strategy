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

    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def calculate_rsi(self, data, period=14):
        """Calculate RSI for trend analysis"""
        try:
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        except:
            return pd.Series([50] * len(data), index=data.index)  # Default RSI of 50

    def calculate_macd(self, data):
        """Calculate MACD for trend confirmation"""
        try:
            ema12 = self.calculate_ema(data, 12)
            ema26 = self.calculate_ema(data, 26)
            macd = ema12 - ema26
            signal = self.calculate_ema(macd, 9)
            return macd, signal
        except:
            # Return default values if calculation fails
            zeros = pd.Series([0] * len(data), index=data.index)
            return zeros, zeros

    def analyze_trend(self):
        """Advanced trend analysis with error handling"""
        try:
            if len(self.data) < 50:
                return {
                    'short_trend': "INSUFFICIENT_DATA",
                    'price_trend': "INSUFFICIENT_DATA",
                    'ema_5_slope': 0,
                    'ema_20_slope': 0,
                    'volume_trend': "NORMAL",
                    'current_price': float(self.data['Close'].iloc[-1]),
                    'ema_5': float(self.data['EMA_5'].iloc[-1]),
                    'ema_20': float(self.data['EMA_20'].iloc[-1]),
                    'ema_50': float(self.data['EMA_50'].iloc[-1])
                }

            # EMA trend analysis
            ema_5 = float(self.data['EMA_5'].iloc[-1])
            ema_20 = float(self.data['EMA_20'].iloc[-1])
            ema_50 = float(self.data['EMA_50'].iloc[-1])

            current_price = float(self.data['Close'].iloc[-1])

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
            try:
                ema_5_slope = float((self.data['EMA_5'].iloc[-1] - self.data['EMA_5'].iloc[-10]) / 10)
                ema_20_slope = float((self.data['EMA_20'].iloc[-1] - self.data['EMA_20'].iloc[-10]) / 10)
            except:
                ema_5_slope = ema_20_slope = 0.0

            # Volume trend (if available)
            volume_trend = "NORMAL"
            try:
                if 'Volume' in self.data.columns:
                    recent_volume = self.data['Volume'].tail(5).mean()
                    historical_volume = self.data['Volume'].tail(20).mean()
                    if recent_volume > historical_volume * 1.2:
                        volume_trend = "HIGH"
                    elif recent_volume < historical_volume * 0.8:
                        volume_trend = "LOW"
            except:
                volume_trend = "NORMAL"

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
        except Exception as e:
            print(f"Trend analysis error: {e}")
            return {
                'short_trend': "ERROR",
                'price_trend': "ERROR",
                'ema_5_slope': 0,
                'ema_20_slope': 0,
                'volume_trend': "NORMAL",
                'current_price': float(self.data['Close'].iloc[-1]) if len(self.data) > 0 else 0,
                'ema_5': 0,
                'ema_20': 0,
                'ema_50': 0
            }

    def prepare_data(self):
        """Prepare data with all technical indicators"""
        try:
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
        except Exception as e:
            print(f"Data preparation error: {e}")
            return self.data

    def identify_alerts(self):
        """Enhanced alert candle identification"""
        try:
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
        except Exception as e:
            print(f"Alert identification error: {e}")
            # Create default alert columns
            self.data['Long_Alert'] = False
            self.data['Short_Alert'] = False
            return self.data

    def generate_advanced_predictions(self):
        """Generate intelligent future predictions with reasoning"""
        try:
            if len(self.data) < 50:
                return []

            trend_analysis = self.analyze_trend()
            current_price = trend_analysis['current_price']
            ema_5 = trend_analysis['ema_5']
            ema_20 = trend_analysis['ema_20']

            currency_symbol = self.get_currency_symbol()
            predictions = []

            # Recent price volatility
            price_volatility = self.data['Close'].tail(20).std()

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
                        'entry_level': f"{currency_symbol}{entry_level:.2f}",
                        'stop_loss': f"{currency_symbol}{stop_level:.2f}",
                        'target': f"{currency_symbol}{target_level:.2f}",
                        'confidence': confidence,
                        'reasoning': f"Bullish trend with price near EMA-{self.ema_period}. Entry above {currency_symbol}{entry_level:.2f} with EMA support.",
                        'time_frame': '3-7 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"Watch for volume confirmation. Current RSI: {self.data['RSI'].iloc[-1]:.1f}"
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
                        'entry_level': f"{currency_symbol}{entry_level:.2f}",
                        'stop_loss': f"{currency_symbol}{stop_level:.2f}",
                        'target': f"{currency_symbol}{target_level:.2f}",
                        'confidence': confidence,
                        'reasoning': f"Bearish trend with price near EMA-{self.ema_period}. Entry below {currency_symbol}{entry_level:.2f} with EMA resistance.",
                        'time_frame': '3-7 days',
                        'risk_reward': f"1:{self.risk_reward_ratio}",
                        'additional_notes': f"Watch for volume confirmation. Current RSI: {self.data['RSI'].iloc[-1]:.1f}"
                    })

            # Prediction 2: Mean reversion setup
            try:
                rsi_current = float(self.data['RSI'].iloc[-1])
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
            except:
                pass  # Skip RSI predictions if calculation fails

            self.future_predictions = predictions
            return predictions

        except Exception as e:
            print(f"Prediction generation error: {e}")
            return []

    def generate_signals(self):
        """Generate basic trading signals"""
        signals = []
        try:
            for i in range(10, len(self.data)):
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
                            'Entry_Price': round(float(entry_price), 2),
                            'Stop_Loss': round(float(stop_loss), 2),
                            'Take_Profit': round(float(take_profit), 2),
                            'Risk_Amount': round(float(risk), 2),
                            'Reward_Potential': round(float(risk * self.risk_reward_ratio), 2),
                            'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                            'Current_Price': round(float(current['Close']), 2),
                            'RSI_At_Signal': round(float(current['RSI']), 1) if not pd.isna(current['RSI']) else 50.0
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
                            'Entry_Price': round(float(entry_price), 2),
                            'Stop_Loss': round(float(stop_loss), 2),
                            'Take_Profit': round(float(take_profit), 2),
                            'Risk_Amount': round(float(risk), 2),
                            'Reward_Potential': round(float(risk * self.risk_reward_ratio), 2),
                            'Risk_Reward_Ratio': f"1:{self.risk_reward_ratio}",
                            'Current_Price': round(float(current['Close']), 2),
                            'RSI_At_Signal': round(float(current['RSI']), 1) if not pd.isna(current['RSI']) else 50.0
                        })

            self.signals = pd.DataFrame(signals)
            return self.signals
        except Exception as e:
            print(f"Signal generation error: {e}")
            self.signals = pd.DataFrame()
            return self.signals

    def analyze(self):
        """Run complete analysis with extensive error handling"""
        try:
            self.prepare_data()
            self.identify_alerts()
            signals = self.generate_signals()
            future_predictions = self.generate_advanced_predictions()
            trend_analysis = self.analyze_trend()

            # Calculate performance metrics
            total_signals = len(signals)

            return {
                'long_alerts': int(self.data['Long_Alert'].sum()) if 'Long_Alert' in self.data.columns else 0,
                'short_alerts': int(self.data['Short_Alert'].sum()) if 'Short_Alert' in self.data.columns else 0,
                'total_candles': len(self.data),
                'total_signals': total_signals,
                'total_trades': 0,  # Simplified for now
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'avg_holding_days': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0,
                'profit_factor': 0.0,
                'future_predictions': future_predictions,
                'trend_analysis': trend_analysis,
                'success': True
            }
        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'success': False,
                'error': f"Analysis failed: {str(e)}"
            }

def fetch_extended_data(symbol, period='1y'):
    """Enhanced data fetching with better error handling"""
    try:
        print(f"Fetching data for {symbol} with period {period}")
        ticker = yf.Ticker(symbol)

        # Try different approaches based on period
        if period == 'max':
            try:
                data = ticker.history(period='max')
                if data.empty:
                    data = ticker.history(period='10y')
                if data.empty:
                    data = ticker.history(period='5y')
            except:
                data = ticker.history(period='5y')
        elif period == '10y':
            try:
                data = ticker.history(period='10y')
                if data.empty or len(data) < 100:
                    # Try date-based approach
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=10*365)
                    data = ticker.history(start=start_date, end=end_date)
                if data.empty:
                    data = ticker.history(period='5y')
            except:
                data = ticker.history(period='5y')
        else:
            data = ticker.history(period=period)

        if data.empty:
            print(f"No data received for {symbol}")
            return None

        # Clean the data
        data = data.dropna()

        if len(data) < 10:
            print(f"Insufficient data for {symbol}: only {len(data)} points")
            return None

        print(f"Successfully fetched {len(data)} data points for {symbol}")
        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def create_simple_chart(strategy, symbol):
    """Create a simple chart to avoid complex plotting errors"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        currency_symbol = strategy.get_currency_symbol()

        # Basic price chart
        ax.plot(strategy.data.index, strategy.data['Close'], label='Close Price', linewidth=2, color='blue')

        # Add EMAs if available
        if 'EMA_5' in strategy.data.columns:
            ax.plot(strategy.data.index, strategy.data['EMA_5'], label=f'EMA {strategy.ema_period}', color='red', linewidth=1.5)
        if 'EMA_20' in strategy.data.columns:
            ax.plot(strategy.data.index, strategy.data['EMA_20'], label='EMA 20', color='orange', linewidth=1.5)

        # Mark alert candles if available
        if 'Long_Alert' in strategy.data.columns:
            long_alerts = strategy.data[strategy.data['Long_Alert']]
            if not long_alerts.empty:
                ax.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=50, label='Long Alert')

        if 'Short_Alert' in strategy.data.columns:
            short_alerts = strategy.data[strategy.data['Short_Alert']]
            if not short_alerts.empty:
                ax.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=50, label='Short Alert')

        ax.set_title(f'{symbol} - EMA Alert Candle Strategy ({len(strategy.data)} data points)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Price ({currency_symbol})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=80)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return chart_url
    except Exception as e:
        print(f"Chart creation error: {e}")
        return None

# Store analysis data globally for downloads
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
        .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 0; }
        .card { box-shadow: 0 8px 16px rgba(0,0,0,0.1); border: none; margin-bottom: 20px; }
        .metric-card { background: #f8f9fa; border-left: 4px solid #007bff; }
        .btn-analyze { background: linear-gradient(45deg, #007bff, #0056b3); border: none; }
        .prediction-card { background: #fff3cd; border-radius: 10px; padding: 20px; margin: 10px 0; border-left: 5px solid #ffc107; }
        .trend-analysis { background: #d1ecf1; border-radius: 10px; padding: 20px; border-left: 5px solid #17a2b8; }
    </style>
</head>
<body>
    <div class="hero text-center">
        <div class="container">
            <h1 class="display-4 mb-3"><i class="fas fa-chart-line me-3"></i>EMA Alert Strategy</h1>
            <p class="lead">Professional Trading Analysis with Extended Data Support</p>
        </div>
    </div>

    <div class="container my-5">
        <div class="row">
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="mb-0"><i class="fas fa-cog me-2"></i>Configuration</h5>
                    </div>
                    <div class="card-body">
                        <form id="analysisForm">
                            <div class="mb-3">
                                <label class="form-label">Stock Symbol</label>
                                <input type="text" class="form-control" id="symbol" value="AAPL" placeholder="AAPL, RELIANCE.NS, BTC-USD">
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
                                    <option value="10y">10 Years</option>
                                    <option value="max">Maximum Available</option>
                                </select>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Currency</label>
                                <select class="form-select" id="currency">
                                    <option value="USD">🇺🇸 US Dollar ($)</option>
                                    <option value="INR">🇮🇳 Indian Rupee (₹)</option>
                                    <option value="EUR">🇪🇺 Euro (€)</option>
                                    <option value="GBP">🇬🇧 British Pound (£)</option>
                                </select>
                            </div>

                            <button type="submit" class="btn btn-analyze btn-primary w-100 py-3">
                                <i class="fas fa-rocket me-2"></i>Analyze Strategy
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-info-circle me-2"></i>Features</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-unstyled">
                            <li><i class="fas fa-check text-success me-2"></i>Extended historical data (up to 10+ years)</li>
                            <li><i class="fas fa-check text-success me-2"></i>Multi-currency support</li>
                            <li><i class="fas fa-check text-success me-2"></i>Intelligent future predictions</li>
                            <li><i class="fas fa-check text-success me-2"></i>Professional trend analysis</li>
                            <li><i class="fas fa-check text-success me-2"></i>Multi-market support (US, NSE, Crypto)</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <div id="loading" class="text-center my-5" style="display:none;">
            <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;"></div>
            <p class="mt-3 h5">Analyzing... This may take up to 30 seconds for extended periods.</p>
        </div>

        <div id="results" class="mt-5" style="display:none;">
            <div id="trendSection" class="trend-analysis mb-4">
                <h5><i class="fas fa-trending-up me-2"></i>Market Trend Analysis</h5>
                <div id="trendAnalysis"></div>
            </div>

            <div class="card">
                <div class="card-header bg-success text-white">
                    <h5 class="mb-0"><i class="fas fa-chart-bar me-2"></i>Performance Summary</h5>
                </div>
                <div class="card-body">
                    <div class="row" id="metricsRow"></div>
                </div>
            </div>

            <div id="predictionsSection" class="mb-4" style="display:none;">
                <h5 class="mb-3"><i class="fas fa-crystal-ball me-2"></i>Future Predictions</h5>
                <div id="futurePredictions"></div>
            </div>

            <div class="card mt-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="fas fa-chart-area me-2"></i>Technical Chart</h5>
                </div>
                <div class="card-body text-center">
                    <img id="strategyChart" class="img-fluid rounded" alt="Strategy Chart" style="max-height: 600px;">
                </div>
            </div>
        </div>

        <div id="error" class="alert alert-danger mt-4" style="display:none;"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const symbol = document.getElementById('symbol').value.toUpperCase().trim();
            const period = document.getElementById('period').value;
            const currency = document.getElementById('currency').value;

            if (!symbol) {
                showError('Please enter a valid stock symbol');
                return;
            }

            showLoading(true);
            hideResults();
            hideError();

            try {
                const requestBody = {
                    symbol: symbol, 
                    period: period,
                    currency: currency,
                    ema_period: 5,
                    risk_reward: 3.0
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
                console.log('Response headers:', response.headers);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.log('Error response:', errorText);
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }

                const contentType = response.headers.get('content-type');
                console.log('Content-Type:', contentType);

                if (!contentType || !contentType.includes('application/json')) {
                    const responseText = await response.text();
                    console.log('Non-JSON response:', responseText);
                    throw new Error('Server returned non-JSON response: ' + responseText.substring(0, 200));
                }

                const data = await response.json();
                console.log('Parsed data:', data);

                if (data.success) {
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
            const currencySymbols = {'USD': '$', 'INR': '₹', 'EUR': '€', 'GBP': '£'};
            const currencySymbol = currencySymbols[currency] || '$';

            // Display trend analysis
            if (data.trend_analysis) {
                const trend = data.trend_analysis;
                const trendHtml = `
                    <div class="row">
                        <div class="col-md-6">
                            <h6>Trend Direction</h6>
                            <p><strong>Short-term:</strong> ${trend.short_trend}</p>
                            <p><strong>Price vs EMAs:</strong> ${trend.price_trend}</p>
                            <p><strong>Volume:</strong> ${trend.volume_trend}</p>
                        </div>
                        <div class="col-md-6">
                            <h6>EMA Levels</h6>
                            <p><strong>Current Price:</strong> ${currencySymbol}${trend.current_price.toFixed(2)}</p>
                            <p><strong>EMA-5:</strong> ${currencySymbol}${trend.ema_5.toFixed(2)}</p>
                            <p><strong>EMA-20:</strong> ${currencySymbol}${trend.ema_20.toFixed(2)}</p>
                        </div>
                    </div>
                `;
                document.getElementById('trendAnalysis').innerHTML = trendHtml;
            }

            // Performance metrics
            const metricsRow = document.getElementById('metricsRow');
            metricsRow.innerHTML = `
                <div class="col-md-3">
                    <div class="metric-card p-3 rounded text-center">
                        <h6 class="text-muted mb-1">Total Candles</h6>
                        <h3 class="text-primary mb-0">${data.total_candles}</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card p-3 rounded text-center">
                        <h6 class="text-muted mb-1">Long Alerts</h6>
                        <h3 class="text-success mb-0">${data.long_alerts}</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card p-3 rounded text-center">
                        <h6 class="text-muted mb-1">Short Alerts</h6>
                        <h3 class="text-danger mb-0">${data.short_alerts}</h3>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="metric-card p-3 rounded text-center">
                        <h6 class="text-muted mb-1">Total Signals</h6>
                        <h3 class="text-info mb-0">${data.total_signals}</h3>
                    </div>
                </div>
            `;

            // Future predictions
            if (data.future_predictions && data.future_predictions.length > 0) {
                let predictionsHtml = '';
                data.future_predictions.forEach((prediction) => {
                    predictionsHtml += `
                        <div class="prediction-card">
                            <h6><i class="fas fa-bullseye me-2"></i>${prediction.type} - ${prediction.confidence} Confidence</h6>
                            <p><strong>Entry:</strong> ${prediction.entry_level}</p>
                            <p><strong>Stop Loss:</strong> ${prediction.stop_loss}</p>
                            <p><strong>Target:</strong> ${prediction.target}</p>
                            <p><strong>Reasoning:</strong> ${prediction.reasoning}</p>
                            <small class="text-muted">${prediction.additional_notes}</small>
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

        print(f"Analysis request: {symbol}, {period}, {currency}")

        if not symbol:
            return jsonify({'success': False, 'error': 'Invalid symbol'}), 400

        # Auto-detect currency based on symbol
        if '.NS' in symbol or '.BO' in symbol:
            currency = 'INR'
        elif 'BTC' in symbol or 'ETH' in symbol or '-USD' in symbol:
            currency = 'USD'

        market_data = fetch_extended_data(symbol, period)

        if market_data is None:
            return jsonify({
                'success': False, 
                'error': f'Could not fetch data for {symbol}. Please verify the symbol and try again.'
            }), 400

        strategy = AdvancedEMAStrategy(
            market_data, 
            symbol=symbol,
            ema_period=ema_period, 
            risk_reward_ratio=risk_reward,
            currency=currency
        )

        results = strategy.analyze()

        if not results.get('success', False):
            return jsonify(results), 400

        # Store data for potential downloads
        analysis_cache[symbol] = {
            'strategy': strategy,
            'market_data': market_data,
            'currency': currency,
            'timestamp': datetime.now()
        }

        # Create chart
        chart_data = create_simple_chart(strategy, symbol)
        results['chart'] = chart_data

        # Ensure all values are JSON serializable
        for key, value in results.items():
            if isinstance(value, (np.integer, np.floating)):
                results[key] = float(value)
            elif isinstance(value, np.ndarray):
                results[key] = value.tolist()

        return jsonify(results)

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return jsonify({'success': False, 'error': 'Invalid JSON format'}), 400
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'EMA Alert Strategy API is running'
    })

@app.route('/test')
def test():
    try:
        test_data = fetch_extended_data('AAPL', '5d')
        return jsonify({
            'status': 'ok',
            'data_points': len(test_data) if test_data is not None else 0,
            'message': 'System operational'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
