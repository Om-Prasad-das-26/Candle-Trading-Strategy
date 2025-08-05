
import os
from flask import Flask, render_template, request, jsonify, send_file
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import json

# Import our strategy
from ema_alert_candle_strategy import EMAAlertCandleStrategy, fetch_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_strategy():
    try:
        data = request.json
        symbol = data.get('symbol', 'AAPL').upper()
        period = data.get('period', '1y')
        ema_period = int(data.get('ema_period', 5))
        risk_reward = float(data.get('risk_reward', 3.0))
        use_filters = data.get('use_filters', True)

        # Fetch data
        market_data = fetch_data(symbol, period=period)
        if market_data is None:
            return jsonify({'error': f'Could not fetch data for {symbol}'}), 400

        # Initialize and run strategy
        strategy = EMAAlertCandleStrategy(
            market_data, 
            ema_period=ema_period,
            risk_reward_ratio=risk_reward,
            use_filters=use_filters
        )

        strategy.prepare_data()
        strategy.identify_alert_candles()
        signals = strategy.generate_signals()

        if not signals.empty:
            trades = strategy.backtest_strategy()
            metrics = strategy.calculate_performance_metrics()

            # Create chart
            chart_data = create_chart_data(strategy, symbol)

            return jsonify({
                'success': True,
                'metrics': metrics,
                'signals_count': len(signals),
                'trades_count': len(trades) if trades is not None else 0,
                'chart_data': chart_data
            })
        else:
            return jsonify({
                'success': True,
                'metrics': {'message': 'No signals generated'},
                'signals_count': 0,
                'trades_count': 0,
                'chart_data': None
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def create_chart_data(strategy, symbol):
    """Create chart data for frontend visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot price and EMA
    ax.plot(strategy.data.index, strategy.data['Close'], label='Close Price', linewidth=1)
    ax.plot(strategy.data.index, strategy.data['EMA_5'], label='EMA 5', color='orange', linewidth=2)

    # Mark Alert Candles
    long_alerts = strategy.data[strategy.data['Long_Alert']]
    short_alerts = strategy.data[strategy.data['Short_Alert']]

    if not long_alerts.empty:
        ax.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=50, label='Long Alert Candles')
    if not short_alerts.empty:
        ax.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=50, label='Short Alert Candles')

    # Mark trade entries
    if not strategy.signals.empty:
        long_signals = strategy.signals[strategy.signals['Type'] == 'LONG']
        short_signals = strategy.signals[strategy.signals['Type'] == 'SHORT']

        if not long_signals.empty:
            ax.scatter(long_signals['Date'], long_signals['Entry_Price'], color='green', marker='o', s=100, label='Long Entries')
        if not short_signals.empty:
            ax.scatter(short_signals['Date'], short_signals['Entry_Price'], color='red', marker='o', s=100, label='Short Entries')

    ax.set_title(f'{symbol} - EMA Alert Candle Strategy')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Convert plot to base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return chart_url

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
