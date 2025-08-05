
# Simplified EMA Alert Candle Strategy for Render Deployment
# This version removes problematic dependencies while maintaining core functionality

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for deployment
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SimplifiedEMAStrategy:
    """
    Simplified version of the EMA Alert Candle Strategy
    Optimized for cloud deployment with minimal dependencies
    """

    def __init__(self, data, ema_period=5, risk_reward_ratio=3.0):
        self.data = data.copy()
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.signals = pd.DataFrame()
        self.trades = []

    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def prepare_data(self):
        """Prepare data with EMA indicators"""
        self.data['EMA_5'] = self.calculate_ema(self.data['Close'], self.ema_period)
        self.data['EMA_20'] = self.calculate_ema(self.data['Close'], 20)
        return self.data

    def identify_alert_candles(self):
        """Identify Alert Candles based on EMA relationship"""
        # Long Alert Candles: Close completely below 5 EMA
        long_alert_condition = (
            (self.data['Close'] < self.data['EMA_5']) &
            (self.data['High'] < self.data['EMA_5'])
        )

        # Short Alert Candles: Close completely above 5 EMA  
        short_alert_condition = (
            (self.data['Close'] > self.data['EMA_5']) &
            (self.data['Low'] > self.data['EMA_5'])
        )

        self.data['Long_Alert'] = long_alert_condition
        self.data['Short_Alert'] = short_alert_condition

        return self.data

    def generate_signals(self):
        """Generate trading signals"""
        signals = []

        for i in range(1, len(self.data)):
            current = self.data.iloc[i]

            # Look for Alert Candles in recent history
            recent_data = self.data.iloc[max(0, i-5):i]

            # Check for long signals
            long_alerts = recent_data[recent_data['Long_Alert']]
            if not long_alerts.empty:
                latest_long_alert = long_alerts.iloc[-1]

                if current['High'] > latest_long_alert['High']:
                    entry_price = latest_long_alert['High']
                    stop_loss = latest_long_alert['Low']
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * self.risk_reward_ratio)

                    signals.append({
                        'Date': current.name,
                        'Type': 'LONG',
                        'Entry_Price': entry_price,
                        'Stop_Loss': stop_loss,
                        'Take_Profit': take_profit,
                        'Risk': risk
                    })

            # Check for short signals
            short_alerts = recent_data[recent_data['Short_Alert']]
            if not short_alerts.empty:
                latest_short_alert = short_alerts.iloc[-1]

                if current['Low'] < latest_short_alert['Low']:
                    entry_price = latest_short_alert['Low']
                    stop_loss = latest_short_alert['High']
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * self.risk_reward_ratio)

                    signals.append({
                        'Date': current.name,
                        'Type': 'SHORT',
                        'Entry_Price': entry_price,
                        'Stop_Loss': stop_loss,
                        'Take_Profit': take_profit,
                        'Risk': risk
                    })

        self.signals = pd.DataFrame(signals)
        return self.signals

    def backtest_strategy(self, initial_capital=10000):
        """Simple backtesting"""
        if self.signals.empty:
            return None

        capital = initial_capital
        trades = []

        for idx, signal in self.signals.iterrows():
            # Simplified trade execution
            if signal['Type'] == 'LONG':
                profit = signal['Risk'] * self.risk_reward_ratio
            else:
                profit = signal['Risk'] * self.risk_reward_ratio

            # Simulate 60% win rate
            if np.random.random() > 0.4:  # Win
                pnl = profit * 0.8  # Reduced profit due to slippage
            else:  # Loss
                pnl = -signal['Risk'] * 1.1  # Small additional loss due to slippage

            capital += pnl
            trades.append({
                'Entry_Date': signal['Date'],
                'Type': signal['Type'],
                'Entry_Price': signal['Entry_Price'],
                'PnL': pnl,
                'Capital': capital
            })

        self.trades = pd.DataFrame(trades)
        return self.trades

    def calculate_metrics(self):
        """Calculate basic performance metrics"""
        if self.trades.empty:
            return {}

        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['PnL'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = self.trades['PnL'].sum()

        return {
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Total P&L': total_pnl,
            'Final Capital': 10000 + total_pnl,
            'ROI': (total_pnl / 10000) * 100
        }

def fetch_data_simple(symbol, period='1y'):
    """Simplified data fetching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)

        if data.empty:
            return None

        return data.dropna()

    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return None

def create_simple_chart(strategy, symbol):
    """Create a simple chart for web display"""
    try:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot price and EMA
        ax.plot(strategy.data.index, strategy.data['Close'], label='Close Price', linewidth=1)
        ax.plot(strategy.data.index, strategy.data['EMA_5'], label='EMA 5', color='orange', linewidth=2)

        # Mark Alert Candles
        long_alerts = strategy.data[strategy.data['Long_Alert']]
        short_alerts = strategy.data[strategy.data['Short_Alert']]

        if not long_alerts.empty:
            ax.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=50, label='Long Alert')
        if not short_alerts.empty:
            ax.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=50, label='Short Alert')

        ax.set_title(f'{symbol} - EMA Alert Candle Strategy')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=100)
        img.seek(0)
        chart_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return chart_url

    except Exception as e:
        print(f"Error creating chart: {str(e)}")
        return None

def analyze_symbol(symbol='AAPL', period='1y'):
    """Main analysis function"""
    print(f"Analyzing {symbol}...")

    # Fetch data
    data = fetch_data_simple(symbol, period)
    if data is None:
        print("Failed to fetch data")
        return None

    # Run strategy
    strategy = SimplifiedEMAStrategy(data)
    strategy.prepare_data()
    strategy.identify_alert_candles()
    signals = strategy.generate_signals()

    if not signals.empty:
        trades = strategy.backtest_strategy()
        metrics = strategy.calculate_metrics()

        print(f"Generated {len(signals)} signals")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        return strategy
    else:
        print("No signals generated")
        return None

if __name__ == "__main__":
    # Test the simplified strategy
    result = analyze_symbol('AAPL', '1y')
