
# Complete Trading Strategy Implementation
# EMA Alert Candle Strategy with Support/Resistance and Breakout Signals
# Author: Trading Strategy Framework
# Date: August 2025

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    print("pandas_ta not available. Using custom indicator implementations.")

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_LIB_AVAILABLE = True
except ImportError:
    BACKTESTING_LIB_AVAILABLE = False
    print("backtesting library not available. Using custom backtesting implementation.")

class TechnicalIndicators:
    """
    Custom implementation of technical indicators for the strategy
    """

    @staticmethod
    def ema(data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(data, period):
        """Calculate Simple Moving Average"""
        return data.rolling(window=period).mean()

    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower

    @staticmethod
    def rsi(data, period=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def support_resistance_levels(df, window=20, min_touches=2):
        """
        Identify support and resistance levels using pivot points
        """
        highs = df['High'].rolling(window=window, center=True).max()
        lows = df['Low'].rolling(window=window, center=True).min()

        # Find pivot highs and lows
        pivot_highs = df['High'][df['High'] == highs]
        pivot_lows = df['Low'][df['Low'] == lows]

        # Group similar levels together (within 1% tolerance)
        resistance_levels = []
        support_levels = []
        tolerance = 0.01

        for level in pivot_highs.dropna():
            similar_levels = pivot_highs[abs(pivot_highs - level) / level <= tolerance]
            if len(similar_levels) >= min_touches:
                resistance_levels.append(level)

        for level in pivot_lows.dropna():
            similar_levels = pivot_lows[abs(pivot_lows - level) / level <= tolerance]
            if len(similar_levels) >= min_touches:
                support_levels.append(level)

        return list(set(resistance_levels)), list(set(support_levels))

class EMAAlertCandleStrategy:
    """
    Implementation of the EMA Alert Candle Trading Strategy

    Strategy Rules:
    1. Alert Candle Identification:
       - Candle closes completely above 5 EMA = potential short Alert Candle
       - Candle closes completely below 5 EMA = potential long Alert Candle

    2. Trade Entry Conditions:
       - Short: Enter when Alert Candle low is breached
       - Long: Enter when Alert Candle high is breached

    3. Stop Loss:
       - Short: Above Alert Candle high
       - Long: Below Alert Candle low

    4. Take Profit: 1:3 risk-reward ratio minimum

    5. Additional Filter:
       - Long: Current close > Alert Candle close
       - Short: Current close < Alert Candle close
    """

    def __init__(self, data, ema_period=5, risk_reward_ratio=3.0, use_filters=True):
        self.data = data.copy()
        self.ema_period = ema_period
        self.risk_reward_ratio = risk_reward_ratio
        self.use_filters = use_filters
        self.signals = pd.DataFrame()
        self.trades = []

    def prepare_data(self):
        """Prepare data with technical indicators"""
        # Calculate 5-period EMA
        self.data['EMA_5'] = TechnicalIndicators.ema(self.data['Close'], self.ema_period)

        # Calculate additional indicators for context
        self.data['EMA_20'] = TechnicalIndicators.ema(self.data['Close'], 20)
        self.data['RSI'] = TechnicalIndicators.rsi(self.data['Close'])

        # Bollinger Bands for additional context
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(self.data['Close'])
        self.data['BB_Upper'] = bb_upper
        self.data['BB_Middle'] = bb_middle
        self.data['BB_Lower'] = bb_lower

        # Support and Resistance levels
        resistance_levels, support_levels = TechnicalIndicators.support_resistance_levels(self.data)
        self.resistance_levels = resistance_levels
        self.support_levels = support_levels

        return self.data

    def identify_alert_candles(self):
        """Identify Alert Candles based on EMA relationship"""
        # Long Alert Candles: Close completely below 5 EMA
        long_alert_condition = (
            (self.data['Close'] < self.data['EMA_5']) &
            (self.data['High'] < self.data['EMA_5'])  # Entire candle below EMA
        )

        # Short Alert Candles: Close completely above 5 EMA
        short_alert_condition = (
            (self.data['Close'] > self.data['EMA_5']) &
            (self.data['Low'] > self.data['EMA_5'])  # Entire candle above EMA
        )

        self.data['Long_Alert'] = long_alert_condition
        self.data['Short_Alert'] = short_alert_condition

        return self.data

    def generate_signals(self):
        """Generate trading signals based on Alert Candle breakouts"""
        signals = []

        for i in range(1, len(self.data)):
            current = self.data.iloc[i]
            previous = self.data.iloc[i-1]

            # Look for Alert Candles in recent history (last 5 candles)
            recent_data = self.data.iloc[max(0, i-5):i]

            # Check for long signals (breakout above Alert Candle high)
            long_alerts = recent_data[recent_data['Long_Alert']]
            if not long_alerts.empty:
                latest_long_alert = long_alerts.iloc[-1]

                # Entry condition: Current high breaks Alert Candle high
                if current['High'] > latest_long_alert['High']:
                    # Additional filter: Current close > Alert Candle close
                    if not self.use_filters or current['Close'] > latest_long_alert['Close']:
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
                            'Risk': risk,
                            'Alert_Candle_Date': latest_long_alert.name,
                            'Current_Price': current['Close']
                        })

            # Check for short signals (breakout below Alert Candle low)
            short_alerts = recent_data[recent_data['Short_Alert']]
            if not short_alerts.empty:
                latest_short_alert = short_alerts.iloc[-1]

                # Entry condition: Current low breaks Alert Candle low
                if current['Low'] < latest_short_alert['Low']:
                    # Additional filter: Current close < Alert Candle close
                    if not self.use_filters or current['Close'] < latest_short_alert['Close']:
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
                            'Risk': risk,
                            'Alert_Candle_Date': latest_short_alert.name,
                            'Current_Price': current['Close']
                        })

        self.signals = pd.DataFrame(signals)
        return self.signals

    def backtest_strategy(self, initial_capital=10000, position_size_pct=0.02):
        """
        Backtest the strategy with proper risk management
        """
        if self.signals.empty:
            print("No signals generated. Cannot backtest.")
            return None

        capital = initial_capital
        position = 0
        equity_curve = []
        trade_log = []

        # Process each signal
        for idx, signal in self.signals.iterrows():
            entry_date = signal['Date']
            entry_price = signal['Entry_Price']
            stop_loss = signal['Stop_Loss']
            take_profit = signal['Take_Profit']
            trade_type = signal['Type']

            # Calculate position size based on risk
            risk_amount = capital * position_size_pct
            if trade_type == 'LONG':
                shares = risk_amount / signal['Risk']
            else:
                shares = risk_amount / signal['Risk']

            # Find exit point
            future_data = self.data[self.data.index > entry_date]
            exit_price = None
            exit_date = None
            exit_reason = None

            for future_idx, future_candle in future_data.iterrows():
                if trade_type == 'LONG':
                    if future_candle['Low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_date = future_idx
                        exit_reason = 'Stop Loss'
                        break
                    elif future_candle['High'] >= take_profit:
                        exit_price = take_profit
                        exit_date = future_idx
                        exit_reason = 'Take Profit'
                        break
                else:  # SHORT
                    if future_candle['High'] >= stop_loss:
                        exit_price = stop_loss
                        exit_date = future_idx
                        exit_reason = 'Stop Loss'
                        break
                    elif future_candle['Low'] <= take_profit:
                        exit_price = take_profit
                        exit_date = future_idx
                        exit_reason = 'Take Profit'
                        break

            # If no exit found, use last available price
            if exit_price is None:
                exit_price = future_data.iloc[-1]['Close'] if not future_data.empty else entry_price
                exit_date = future_data.index[-1] if not future_data.empty else entry_date
                exit_reason = 'End of Data'

            # Calculate P&L
            if trade_type == 'LONG':
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            capital += pnl

            trade_log.append({
                'Entry_Date': entry_date,
                'Exit_Date': exit_date,
                'Type': trade_type,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'Shares': shares,
                'PnL': pnl,
                'Capital': capital,
                'Exit_Reason': exit_reason,
                'Duration': (exit_date - entry_date).days if exit_date else 0
            })

        self.trades = pd.DataFrame(trade_log)
        return self.trades

    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.trades.empty:
            return {}

        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(self.trades[self.trades['PnL'] > 0])
        losing_trades = len(self.trades[self.trades['PnL'] < 0])

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = self.trades['PnL'].sum()
        avg_win = self.trades[self.trades['PnL'] > 0]['PnL'].mean() if winning_trades > 0 else 0
        avg_loss = self.trades[self.trades['PnL'] < 0]['PnL'].mean() if losing_trades > 0 else 0

        # Risk metrics
        returns = self.trades['PnL'] / 10000  # Assuming initial capital of 10000
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Maximum drawdown
        equity_curve = (10000 + self.trades['PnL'].cumsum())
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf')

        return {
            'Total Trades': total_trades,
            'Winning Trades': winning_trades,
            'Losing Trades': losing_trades,
            'Win Rate': win_rate,
            'Total P&L': total_pnl,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Final Capital': 10000 + total_pnl
        }

def fetch_data(symbol, period='2y', interval='1d'):
    """
    Fetch historical data using yfinance
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)

        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        # Clean the data
        data = data.dropna()

        # Ensure we have the required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError("Data missing required OHLCV columns")

        print(f"Successfully fetched {len(data)} rows of data for {symbol}")
        return data

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def plot_strategy_analysis(strategy_obj, symbol):
    """
    Create comprehensive visualization of the strategy
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 20))

    # Plot 1: Price action with EMA and signals
    ax1 = axes[0]
    ax1.plot(strategy_obj.data.index, strategy_obj.data['Close'], label='Close Price', linewidth=1)
    ax1.plot(strategy_obj.data.index, strategy_obj.data['EMA_5'], label='EMA 5', color='orange', linewidth=2)
    ax1.plot(strategy_obj.data.index, strategy_obj.data['EMA_20'], label='EMA 20', color='red', linewidth=1)

    # Mark Alert Candles
    long_alerts = strategy_obj.data[strategy_obj.data['Long_Alert']]
    short_alerts = strategy_obj.data[strategy_obj.data['Short_Alert']]

    ax1.scatter(long_alerts.index, long_alerts['Low'], color='green', marker='^', s=50, label='Long Alert Candles')
    ax1.scatter(short_alerts.index, short_alerts['High'], color='red', marker='v', s=50, label='Short Alert Candles')

    # Mark trade entries
    if not strategy_obj.signals.empty:
        long_signals = strategy_obj.signals[strategy_obj.signals['Type'] == 'LONG']
        short_signals = strategy_obj.signals[strategy_obj.signals['Type'] == 'SHORT']

        ax1.scatter(long_signals['Date'], long_signals['Entry_Price'], color='green', marker='o', s=100, label='Long Entries')
        ax1.scatter(short_signals['Date'], short_signals['Entry_Price'], color='red', marker='o', s=100, label='Short Entries')

    ax1.set_title(f'{symbol} - EMA Alert Candle Strategy Analysis')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: RSI
    ax2 = axes[1]
    ax2.plot(strategy_obj.data.index, strategy_obj.data['RSI'], label='RSI', color='purple')
    ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    ax2.set_title('RSI Indicator')
    ax2.set_ylabel('RSI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Equity Curve
    if not strategy_obj.trades.empty:
        ax3 = axes[2]
        equity_curve = 10000 + strategy_obj.trades['PnL'].cumsum()
        ax3.plot(strategy_obj.trades['Exit_Date'], equity_curve, label='Equity Curve', linewidth=2)
        ax3.axhline(y=10000, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        ax3.set_title('Equity Curve')
        ax3.set_ylabel('Portfolio Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Trade Distribution
        ax4 = axes[3]
        ax4.hist(strategy_obj.trades['PnL'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        ax4.set_title('Trade P&L Distribution')
        ax4.set_xlabel('P&L')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax3 = axes[2]
        ax3.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Equity Curve - No Data')

        ax4 = axes[3]
        ax4.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Trade Distribution - No Data')

    plt.tight_layout()
    plt.show()

def main_analysis(symbol='AAPL', period='2y'):
    """
    Main function to run the complete strategy analysis
    """
    print(f"Starting EMA Alert Candle Strategy Analysis for {symbol}")
    print("="*60)

    # Fetch data
    print("1. Fetching market data...")
    data = fetch_data(symbol, period=period)
    if data is None:
        return None

    # Initialize strategy
    print("2. Initializing strategy...")
    strategy = EMAAlertCandleStrategy(data, ema_period=5, risk_reward_ratio=3.0, use_filters=True)

    # Prepare data with indicators
    print("3. Calculating technical indicators...")
    strategy.prepare_data()

    # Identify Alert Candles
    print("4. Identifying Alert Candles...")
    strategy.identify_alert_candles()

    long_alerts = strategy.data[strategy.data['Long_Alert']].shape[0]
    short_alerts = strategy.data[strategy.data['Short_Alert']].shape[0]
    print(f"   Found {long_alerts} Long Alert Candles")
    print(f"   Found {short_alerts} Short Alert Candles")

    # Generate signals
    print("5. Generating trading signals...")
    signals = strategy.generate_signals()
    print(f"   Generated {len(signals)} trading signals")

    if signals.empty:
        print("   No trading signals generated. Strategy may need adjustment.")
        return strategy

    # Backtest strategy
    print("6. Running backtest...")
    trades = strategy.backtest_strategy(initial_capital=10000, position_size_pct=0.02)

    if trades is not None and not trades.empty:
        print(f"   Executed {len(trades)} trades")

        # Calculate performance metrics
        print("7. Calculating performance metrics...")
        metrics = strategy.calculate_performance_metrics()

        # Display results
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE RESULTS")
        print("="*60)

        for key, value in metrics.items():
            if isinstance(value, float):
                if 'Rate' in key or 'Ratio' in key or 'Drawdown' in key:
                    print(f"{key:<20}: {value:.2%}")
                else:
                    print(f"{key:<20}: {value:.2f}")
            else:
                print(f"{key:<20}: {value}")

        # Create visualizations
        print("\n8. Creating visualizations...")
        plot_strategy_analysis(strategy, symbol)

    else:
        print("   No trades were executed during backtest period.")

    return strategy

# Example usage and testing
if __name__ == "__main__":
    # Test with different symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

    print("EMA Alert Candle Strategy - Complete Implementation")
    print("="*60)
    print("Available test symbols:", symbols)

    # Run analysis for Apple as default
    strategy_result = main_analysis('AAPL', period='1y')

    if strategy_result:
        print("\nStrategy analysis completed successfully!")
        print("\nTo test with different symbols, use:")
        print("strategy_result = main_analysis('SYMBOL', period='1y')")
