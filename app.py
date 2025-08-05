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

app = Flask(__name__)

class EnhancedEMAStrategy:
    def __init__(self, data, ema_period=5, risk_reward_ratio=3.0):
        self.data = data.copy()
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.signals = pd.DataFrame()
        self.trades = []

    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def prepare_data(self):
        self.data['EMA_5'] = self.calculate_ema(self.data['Close'], self.ema_period)
        self.data['EMA_20'] = self.calculate_ema(self.data['Close'], 20)
        return self.data

    def identify_alerts(self):
        # Long Alert Candles: Close completely below 5 EMA
        self.data['Long_Alert'] = (
            (self.data['Close'] < self.data['EMA_5']) & 
            (self.data['High'] < self.data['EMA_5'])
        )

        # Short Alert Candles: Close completely above 5 EMA  
        self.data['Short_Alert'] = (
            (self.data['Close'] > self.data['EMA_5']) & 
            (self.data['Low'] > self.data['EMA_5'])
        )
        return self.data

    def generate_signals(self):
        """Generate detailed trading signals with entry/exit points"""
        signals = []

        for i in range(5, len(self.data)):  # Start from index 5 to look back
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
                        'Current_Price': round(current['Close'], 2)
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
                        'Current_Price': round(current['Close'], 2)
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
                'Reward_Potential': signal['Reward_Potential']
            })

        self.trades = pd.DataFrame(trades)
        return self.trades

    def get_future_alerts(self, days_ahead=30):
        """Identify potential future alert candles"""
        if len(self.data) < 50:
            return []

        # Get recent data for trend analysis
        recent_data = self.data.tail(20)
        current_price = self.data['Close'].iloc[-1]
        current_ema = self.data['EMA_5'].iloc[-1]
        ema_trend = self.data['EMA_5'].iloc[-1] - self.data['EMA_5'].iloc[-10]

        future_alerts = []

        # Simple trend-based prediction
        if ema_trend > 0:  # EMA trending up
            # Look for potential short alerts
            potential_short_level = current_price * 1.02  # 2% above current
            future_alerts.append({
                'Type': 'Potential SHORT Alert',
                'Expected_Level': round(potential_short_level, 2),
                'Confidence': 'Medium',
                'Notes': 'If price moves above this level with EMA trending up'
            })
        else:  # EMA trending down
            # Look for potential long alerts
            potential_long_level = current_price * 0.98  # 2% below current
            future_alerts.append({
                'Type': 'Potential LONG Alert',
                'Expected_Level': round(potential_long_level, 2),
                'Confidence': 'Medium',
                'Notes': 'If price moves below this level with EMA trending down'
            })

        return future_alerts

    def analyze(self):
        """Run complete analysis"""
        self.prepare_data()
        self.identify_alerts()
        signals = self.generate_signals()
        trades = self.backtest_signals()
        future_alerts = self.get_future_alerts()

        # Calculate performance metrics
        total_signals = len(signals)
        total_trades = len(trades)

        if total_trades > 0:
            winning_trades = len(trades[trades['PnL'] > 0])
            win_rate = winning_trades / total_trades
            avg_pnl = trades['PnL'].mean()
            total_pnl = trades['PnL'].sum()
            avg_holding_days = trades['Holding_Days'].mean()
        else:
            win_rate = 0
            avg_pnl = 0
            total_pnl = 0
            avg_holding_days = 0

        return {
            'long_alerts': int(self.data['Long_Alert'].sum()),
            'short_alerts': int(self.data['Short_Alert'].sum()),
            'total_candles': len(self.data),
            'total_signals': total_signals,
            'total_trades': total_trades,
            'win_rate': round(win_rate * 100, 1),
            'avg_pnl': round(avg_pnl, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_holding_days': round(avg_holding_days, 1),
            'future_alerts': future_alerts,
            'success': True
        }

def fetch_data(symbol, period='1y'):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if data.empty:
            return None
        return data.dropna()
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_enhanced_chart(strategy, symbol):
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Main price chart
        ax1.plot(strategy.data.index, strategy.data['Close'], label='Close', linewidth=1.5, color='blue')
        ax1.plot(strategy.data.index, strategy.data['EMA_5'], label='EMA 5', color='orange', linewidth=2)
        ax1.plot(strategy.data.index, strategy.data['EMA_20'], label='EMA 20', color='red', linewidth=1)

        # Mark Alert Candles
        long_alerts = strategy.data[strategy.data['Long_Alert']]
        short_alerts = strategy.data[strategy.data['Short_Alert']]

        if not long_alerts.empty:
            ax1.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=40, label='Long Alert', alpha=0.7)
        if not short_alerts.empty:
            ax1.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=40, label='Short Alert', alpha=0.7)

        # Mark Entry Points
        if not strategy.signals.empty:
            long_entries = strategy.signals[strategy.signals['Type'] == 'LONG']
            short_entries = strategy.signals[strategy.signals['Type'] == 'SHORT']

            if not long_entries.empty:
                ax1.scatter(long_entries['Signal_Date'], long_entries['Entry_Price'], 
                           color='darkgreen', marker='o', s=60, label='Long Entry', edgecolor='white', linewidth=1)
            if not short_entries.empty:
                ax1.scatter(short_entries['Signal_Date'], short_entries['Entry_Price'], 
                           color='darkred', marker='o', s=60, label='Short Entry', edgecolor='white', linewidth=1)

        ax1.set_title(f'{symbol} - Enhanced EMA Alert Candle Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # Volume chart
        ax2.bar(strategy.data.index, strategy.data['Volume'], alpha=0.6, color='gray', width=1)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
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
    <title>Enhanced EMA Alert Candle Strategy</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 60px 0; }
        .card { box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none; margin-bottom: 20px; }
        .metric-card { background: #f8f9fa; border-left: 4px solid #007bff; }
        .btn-analyze { background: linear-gradient(45deg, #007bff, #0056b3); border: none; }
        .download-section { background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 20px 0; }
        .future-alerts { background: #e3f2fd; border-radius: 10px; padding: 15px; }
        .trade-summary { background: #fff3cd; border-radius: 10px; padding: 15px; }
    </style>
</head>
<body>
    <div class="hero text-center">
        <div class="container">
            <h1 class="display-4 mb-3">Enhanced EMA Alert Candle Strategy</h1>
            <p class="lead">Advanced trading strategy with detailed analysis, backtesting & data export</p>
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
                                <input type="text" class="form-control" id="symbol" value="AAPL" placeholder="Enter symbol (AAPL, MSFT, etc.)">
                                <small class="form-text text-muted">Popular: AAPL, MSFT, GOOGL, TSLA, SPY, HDB</small>
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
                                <label class="form-label">EMA Period</label>
                                <select class="form-select" id="ema_period">
                                    <option value="5" selected>5 Period</option>
                                    <option value="8">8 Period</option>
                                    <option value="13">13 Period</option>
                                    <option value="21">21 Period</option>
                                </select>
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Risk:Reward Ratio</label>
                                <select class="form-select" id="risk_reward">
                                    <option value="2">1:2</option>
                                    <option value="3" selected>1:3</option>
                                    <option value="4">1:4</option>
                                    <option value="5">1:5</option>
                                </select>
                            </div>
                            <button type="submit" class="btn btn-analyze btn-primary w-100 py-2">
                                <i class="fas fa-chart-line me-2"></i>Analyze Strategy
                            </button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0"><i class="fas fa-info-circle me-2"></i>Enhanced Strategy Features</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <h6 class="text-primary">Alert Candle Detection</h6>
                                <ul class="list-unstyled">
                                    <li><span class="badge bg-success me-2">Long Alert</span>Candle closes below EMA</li>
                                    <li><span class="badge bg-danger me-2">Short Alert</span>Candle closes above EMA</li>
                                </ul>

                                <h6 class="text-primary mt-3">New Features</h6>
                                <ul class="list-unstyled">
                                    <li><i class="fas fa-download text-info me-2"></i>Download trade data</li>
                                    <li><i class="fas fa-crystal-ball text-warning me-2"></i>Future signal predictions</li>
                                    <li><i class="fas fa-chart-bar text-success me-2"></i>Enhanced backtesting</li>
                                </ul>
                            </div>
                            <div class="col-md-6">
                                <h6 class="text-primary">Entry & Exit Rules</h6>
                                <ul class="list-unstyled">
                                    <li><strong>Entry:</strong> Price breaks Alert Candle high/low</li>
                                    <li><strong>Stop Loss:</strong> Opposite side of Alert Candle</li>
                                    <li><strong>Take Profit:</strong> Based on risk:reward ratio</li>
                                </ul>

                                <div class="alert alert-info mt-3">
                                    <small><strong>Extended Analysis:</strong> Now supports up to 10 years of historical data with detailed entry/exit tracking.</small>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div id="loading" class="text-center my-5" style="display:none;">
            <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;"></div>
            <p class="mt-3 h5">Analyzing strategy... This may take a moment for longer periods.</p>
        </div>

        <div id="results" class="mt-5" style="display:none;">
            <!-- Performance Summary -->
            <div class="card">
                <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
                    <h5 class="mb-0"><i class="fas fa-chart-line me-2"></i>Performance Summary</h5>
                    <div id="statusBadges"></div>
                </div>
                <div class="card-body">
                    <div class="row" id="metricsRow"></div>
                </div>
            </div>

            <!-- Download Section -->
            <div class="download-section">
                <h5><i class="fas fa-download me-2"></i>Download Trading Data</h5>
                <div class="row">
                    <div class="col-md-4">
                        <button class="btn btn-outline-primary w-100" onclick="downloadSignals()">
                            <i class="fas fa-signal me-2"></i>Download Signals
                        </button>
                    </div>
                    <div class="col-md-4">
                        <button class="btn btn-outline-success w-100" onclick="downloadTrades()">
                            <i class="fas fa-exchange-alt me-2"></i>Download Trades
                        </button>
                    </div>
                    <div class="col-md-4">
                        <button class="btn btn-outline-info w-100" onclick="downloadAll()">
                            <i class="fas fa-file-export me-2"></i>Download All Data
                        </button>
                    </div>
                </div>
            </div>

            <!-- Future Predictions -->
            <div id="futureSection" class="future-alerts" style="display:none;">
                <h5><i class="fas fa-crystal-ball me-2"></i>Future Signal Predictions</h5>
                <div id="futureAlerts"></div>
            </div>

            <!-- Trade Summary -->
            <div id="tradeSummary" class="trade-summary" style="display:none;">
                <h5><i class="fas fa-list me-2"></i>Recent Trades Summary</h5>
                <div id="recentTrades"></div>
            </div>

            <!-- Chart -->
            <div class="card mt-4">
                <div class="card-header">
                    <h5 class="mb-0"><i class="fas fa-chart-area me-2"></i>Enhanced Strategy Visualization</h5>
                </div>
                <div class="card-body text-center">
                    <img id="strategyChart" class="img-fluid rounded" alt="Strategy Chart" style="max-height: 600px;">
                </div>
            </div>
        </div>

        <div id="error" class="alert alert-danger mt-4" style="display:none;"></div>
        <div id="connectionStatus" class="mt-3"></div>
    </div>

    <footer class="bg-dark text-white text-center py-3 mt-5">
        <p class="mb-0">Enhanced EMA Alert Candle Strategy - Educational Purpose Only</p>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentAnalysisData = null;

        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const symbol = document.getElementById('symbol').value.toUpperCase().trim();
            const period = document.getElementById('period').value;
            const ema_period = document.getElementById('ema_period').value;
            const risk_reward = document.getElementById('risk_reward').value;

            if (!symbol) {
                showError('Please enter a valid stock symbol');
                return;
            }

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
                        ema_period: parseInt(ema_period),
                        risk_reward: parseFloat(risk_reward)
                    })
                });

                const data = await response.json();

                if (data.success) {
                    currentAnalysisData = data;
                    displayResults(data, symbol);
                } else {
                    showError(data.error || 'Analysis failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                showLoading(false);
            }
        });

        function displayResults(data, symbol) {
            // Status badges
            document.getElementById('statusBadges').innerHTML = `
                <span class="badge bg-light text-dark me-1">${data.total_candles} candles</span>
                <span class="badge bg-primary me-1">${data.total_signals} signals</span>
                <span class="badge bg-success me-1">${data.long_alerts} long alerts</span>
                <span class="badge bg-danger me-1">${data.short_alerts} short alerts</span>
            `;

            // Performance metrics
            const metricsRow = document.getElementById('metricsRow');
            metricsRow.innerHTML = `
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total Signals</h6>
                        <h3 class="text-primary mb-0">${data.total_signals}</h3>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total Trades</h6>
                        <h3 class="text-info mb-0">${data.total_trades}</h3>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Win Rate</h6>
                        <h3 class="text-success mb-0">${data.win_rate}%</h3>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Avg P&L</h6>
                        <h3 class="${data.avg_pnl >= 0 ? 'text-success' : 'text-danger'} mb-0">$${data.avg_pnl}</h3>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Total P&L</h6>
                        <h3 class="${data.total_pnl >= 0 ? 'text-success' : 'text-danger'} mb-0">$${data.total_pnl}</h3>
                    </div>
                </div>
                <div class="col-md-2">
                    <div class="metric-card p-3 rounded h-100 text-center">
                        <h6 class="text-muted mb-1">Avg Hold Days</h6>
                        <h3 class="text-warning mb-0">${data.avg_holding_days}</h3>
                    </div>
                </div>
            `;

            // Future alerts
            if (data.future_alerts && data.future_alerts.length > 0) {
                const futureAlertsDiv = document.getElementById('futureAlerts');
                let futureHtml = '';
                data.future_alerts.forEach(alert => {
                    futureHtml += `
                        <div class="alert alert-info mb-2">
                            <strong>${alert.Type}</strong> at $${alert.Expected_Level} 
                            <span class="badge bg-secondary">${alert.Confidence}</span>
                            <br><small>${alert.Notes}</small>
                        </div>
                    `;
                });
                futureAlertsDiv.innerHTML = futureHtml;
                document.getElementById('futureSection').style.display = 'block';
            }

            // Chart
            if (data.chart) {
                document.getElementById('strategyChart').src = 'data:image/png;base64,' + data.chart;
            }

            document.getElementById('results').style.display = 'block';
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        async function downloadSignals() {
            if (!currentAnalysisData) {
                alert('Please run analysis first');
                return;
            }

            try {
                const symbol = document.getElementById('symbol').value.toUpperCase();
                const response = await fetch('/download/signals', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol: symbol})
                });

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${symbol}_signals.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                alert('Download failed: ' + error.message);
            }
        }

        async function downloadTrades() {
            if (!currentAnalysisData) {
                alert('Please run analysis first');
                return;
            }

            try {
                const symbol = document.getElementById('symbol').value.toUpperCase();
                const response = await fetch('/download/trades', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol: symbol})
                });

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${symbol}_trades.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
            } catch (error) {
                alert('Download failed: ' + error.message);
            }
        }

        async function downloadAll() {
            if (!currentAnalysisData) {
                alert('Please run analysis first');
                return;
            }

            try {
                const symbol = document.getElementById('symbol').value.toUpperCase();
                const response = await fetch('/download/all', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({symbol: symbol})
                });

                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${symbol}_complete_analysis.csv`;
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
        ema_period = int(data.get('ema_period', 5))
        risk_reward = float(data.get('risk_reward', 3.0))

        if not symbol:
            return jsonify({'success': False, 'error': 'Invalid symbol'})

        market_data = fetch_data(symbol, period)
        if market_data is None:
            return jsonify({'success': False, 'error': f'Could not fetch data for {symbol}. Please check the symbol and try again.'})

        strategy = EnhancedEMAStrategy(market_data, ema_period=ema_period, risk_reward_ratio=risk_reward)
        results = strategy.analyze()

        # Store data for downloads
        analysis_cache[symbol] = {
            'strategy': strategy,
            'signals': strategy.signals,
            'trades': strategy.trades,
            'market_data': market_data,
            'timestamp': datetime.now()
        }

        chart_data = create_enhanced_chart(strategy, symbol)
        results['chart'] = chart_data

        return jsonify(results)

    except Exception as e:
        return jsonify({'success': False, 'error': f'Analysis error: {str(e)}'})

@app.route('/download/signals', methods=['POST'])
def download_signals():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()

        if symbol not in analysis_cache:
            return jsonify({'error': 'No analysis data found. Please run analysis first.'}), 400

        signals_df = analysis_cache[symbol]['signals']

        if signals_df.empty:
            return jsonify({'error': 'No signals found for this analysis.'}), 400

        # Create CSV
        output = StringIO()
        signals_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_signals.csv'
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

        if trades_df.empty:
            return jsonify({'error': 'No trades found for this analysis.'}), 400

        # Create CSV
        output = StringIO()
        trades_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_trades.csv'
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

        # Combine all data
        output = StringIO()

        # Write header information
        output.write(f"ENHANCED EMA ALERT CANDLE STRATEGY ANALYSIS\n")
        output.write(f"Symbol: {symbol}\n")
        output.write(f"Analysis Date: {cache_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        output.write(f"Total Data Points: {len(cache_data['market_data'])}\n\n")

        # Write signals
        output.write("=== TRADING SIGNALS ===\n")
        if not cache_data['signals'].empty:
            cache_data['signals'].to_csv(output, index=False)
        else:
            output.write("No signals generated\n")

        output.write("\n\n=== TRADE RESULTS ===\n")
        if not cache_data['trades'].empty:
            cache_data['trades'].to_csv(output, index=False)
        else:
            output.write("No trades executed\n")

        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol}_complete_analysis.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'message': 'Enhanced EMA Alert Candle Strategy API is running'
    })

@app.route('/test')
def test():
    try:
        test_data = fetch_data('AAPL', '5d')
        if test_data is not None:
            return jsonify({
                'status': 'ok',
                'data_points': len(test_data),
                'last_price': f"${test_data['Close'].iloc[-1]:.2f}",
                'message': f'Successfully fetched {len(test_data)} data points'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Could not fetch test data'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Test failed: {str(e)}'
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
