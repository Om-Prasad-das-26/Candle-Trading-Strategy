
import os
from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import json

# Import our simplified strategy
from simplified_strategy import SimplifiedEMAStrategy, fetch_data_simple, create_simple_chart

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

        # Fetch data
        market_data = fetch_data_simple(symbol, period=period)
        if market_data is None:
            return jsonify({'error': f'Could not fetch data for {symbol}'}), 400

        # Initialize and run strategy
        strategy = SimplifiedEMAStrategy(
            market_data, 
            ema_period=ema_period,
            risk_reward_ratio=risk_reward
        )

        strategy.prepare_data()
        strategy.identify_alert_candles()
        signals = strategy.generate_signals()

        if not signals.empty:
            trades = strategy.backtest_strategy()
            metrics = strategy.calculate_metrics()

            # Create chart
            chart_data = create_simple_chart(strategy, symbol)

            return jsonify({
                'success': True,
                'metrics': metrics,
                'signals_count': len(signals),
                'trades_count': len(trades) if trades is not None else 0,
                'chart_data': chart_data,
                'data_points': len(market_data),
                'alert_candles': {
                    'long': int(strategy.data['Long_Alert'].sum()),
                    'short': int(strategy.data['Short_Alert'].sum())
                }
            })
        else:
            return jsonify({
                'success': True,
                'metrics': {'message': 'No signals generated for this period'},
                'signals_count': 0,
                'trades_count': 0,
                'chart_data': create_simple_chart(strategy, symbol),
                'data_points': len(market_data),
                'alert_candles': {
                    'long': int(strategy.data['Long_Alert'].sum()),
                    'short': int(strategy.data['Short_Alert'].sum())
                }
            })

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/test')
def test_endpoint():
    """Test endpoint to verify deployment"""
    try:
        # Test data fetching
        data = fetch_data_simple('AAPL', '5d')
        if data is not None:
            return jsonify({
                'status': 'ok',
                'test_data_points': len(data),
                'test_columns': list(data.columns),
                'last_price': float(data['Close'].iloc[-1])
            })
        else:
            return jsonify({'status': 'error', 'message': 'Could not fetch test data'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
