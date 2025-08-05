import os
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

class SimpleStrategy:
    def __init__(self, data):
        self.data = data.copy()

    def calculate_ema(self, data, period):
        return data.ewm(span=period, adjust=False).mean()

    def prepare_data(self):
        self.data['EMA_5'] = self.calculate_ema(self.data['Close'], 5)
        return self.data

    def identify_alerts(self):
        self.data['Long_Alert'] = (self.data['Close'] < self.data['EMA_5']) & (self.data['High'] < self.data['EMA_5'])
        self.data['Short_Alert'] = (self.data['Close'] > self.data['EMA_5']) & (self.data['Low'] > self.data['EMA_5'])
        return self.data

    def analyze(self):
        self.prepare_data()
        self.identify_alerts()

        return {
            'long_alerts': int(self.data['Long_Alert'].sum()),
            'short_alerts': int(self.data['Short_Alert'].sum()),
            'total_candles': len(self.data),
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

def create_chart(strategy, symbol):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(strategy.data.index, strategy.data['Close'], label='Close', linewidth=1)
        ax.plot(strategy.data.index, strategy.data['EMA_5'], label='EMA 5', color='orange', linewidth=2)

        long_alerts = strategy.data[strategy.data['Long_Alert']]
        short_alerts = strategy.data[strategy.data['Short_Alert']]

        if not long_alerts.empty:
            ax.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=30, label='Long Alert')
        if not short_alerts.empty:
            ax.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=30, label='Short Alert')

        ax.set_title(f'{symbol} - EMA Alert Candle Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=80)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return chart_url
    except Exception as e:
        print(f"Chart error: {e}")
        return None

@app.route('/')
def index():
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>EMA Alert Candle Strategy</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center">EMA Alert Candle Strategy</h1>
        <div class="row mt-4">
            <div class="col-md-6 mx-auto">
                <div class="card">
                    <div class="card-body">
                        <form id="analysisForm">
                            <div class="mb-3">
                                <label class="form-label">Stock Symbol</label>
                                <input type="text" class="form-control" id="symbol" value="AAPL">
                            </div>
                            <div class="mb-3">
                                <label class="form-label">Period</label>
                                <select class="form-select" id="period">
                                    <option value="3mo">3 Months</option>
                                    <option value="6mo">6 Months</option>
                                    <option value="1y" selected>1 Year</option>
                                </select>
                            </div>
                            <button type="submit" class="btn btn-primary w-100">Analyze</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <div id="loading" class="text-center mt-4" style="display:none;">
            <div class="spinner-border"></div>
            <p>Analyzing...</p>
        </div>

        <div id="results" class="mt-4" style="display:none;">
            <div class="card">
                <div class="card-body">
                    <h5>Results</h5>
                    <div id="metrics"></div>
                    <div id="chart" class="mt-3"></div>
                </div>
            </div>
        </div>

        <div id="error" class="alert alert-danger mt-4" style="display:none;"></div>
    </div>

    <script>
    document.getElementById('analysisForm').addEventListener('submit', async function(e) {
        e.preventDefault();

        const symbol = document.getElementById('symbol').value;
        const period = document.getElementById('period').value;

        document.getElementById('loading').style.display = 'block';
        document.getElementById('results').style.display = 'none';
        document.getElementById('error').style.display = 'none';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbol: symbol, period: period})
            });

            const data = await response.json();

            if (data.success) {
                document.getElementById('metrics').innerHTML = `
                    <p><strong>Long Alerts:</strong> ${data.long_alerts}</p>
                    <p><strong>Short Alerts:</strong> ${data.short_alerts}</p>
                    <p><strong>Total Candles:</strong> ${data.total_candles}</p>
                `;

                if (data.chart) {
                    document.getElementById('chart').innerHTML = 
                        '<img src="data:image/png;base64,' + data.chart + '" class="img-fluid">';
                }

                document.getElementById('results').style.display = 'block';
            } else {
                document.getElementById('error').textContent = data.error;
                document.getElementById('error').style.display = 'block';
            }
        } catch (error) {
            document.getElementById('error').textContent = 'Network error: ' + error.message;
            document.getElementById('error').style.display = 'block';
        } finally {
            document.getElementById('loading').style.display = 'none';
        }
    });
    </script>
</body>
</html>"""
    return html_content

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        period = data.get('period', '1y')

        market_data = fetch_data(symbol, period)
        if market_data is None:
            return jsonify({'success': False, 'error': f'Could not fetch data for {symbol}'})

        strategy = SimpleStrategy(market_data)
        results = strategy.analyze()

        chart_data = create_chart(strategy, symbol)
        results['chart'] = chart_data

        return jsonify(results)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/test')
def test():
    try:
        data = fetch_data('AAPL', '5d')
        return jsonify({
            'status': 'ok',
            'data_points': len(data) if data is not None else 0,
            'message': 'App is working!'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
