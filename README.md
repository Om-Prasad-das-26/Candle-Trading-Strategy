# EMA Alert Candle Trading Strategy - Complete Guide

## Overview
This is a comprehensive implementation of the EMA Alert Candle trading strategy in Python. The strategy combines support and resistance levels, breakout signals, and a 5-period Exponential Moving Average (EMA) Alert Candle approach.

## Strategy Rules

### 1. Alert Candle Identification
- **Long Alert Candle**: A candle that closes completely below the 5 EMA
- **Short Alert Candle**: A candle that closes completely above the 5 EMA

### 2. Trade Entry Conditions
- **Long Trade**: Enter when the high of the Alert Candle is breached by subsequent price action
- **Short Trade**: Enter when the low of the Alert Candle is breached by subsequent price action

### 3. Stop Loss Placement
- **Long Trades**: Place stop loss just below the low of the Alert Candle
- **Short Trades**: Place stop loss just above the high of the Alert Candle

### 4. Profit Target
- Set minimum profit target using a risk-reward ratio of at least 1:3 relative to stop loss distance

### 5. Additional Filter (Optional)
- **Long Trades**: Require current candle's close to be higher than Alert Candle's close
- **Short Trades**: Require current candle's close to be lower than Alert Candle's close

## Files Structure

### Core Strategy Files
- `ema_alert_candle_strategy.py` - Main strategy implementation
- `colab_setup.py` - Google Colab setup instructions
- `requirements.txt` - Python dependencies

### Web Application Files
- `app.py` - Flask web application for deployment
- `templates/index.html` - Web interface
- `render.yaml` - Render deployment configuration

## Installation and Setup

### Local Development
```bash
# Install required packages
pip install yfinance pandas numpy matplotlib seaborn pandas-ta backtesting

# Run the strategy
python ema_alert_candle_strategy.py
```

### Google Colab Setup
1. Open Google Colab (colab.research.google.com)
2. Create a new notebook
3. Copy and paste the code from `colab_setup.py`
4. Run each cell in sequence
5. Upload or paste the strategy code
6. Test with your preferred stock symbols

### Render Deployment
1. Create a new account on Render.com
2. Connect your GitHub repository
3. Create a new Web Service
4. Set the build command: `pip install -r requirements.txt`
5. Set the start command: `gunicorn app:app`
6. Deploy and access your web application

## Usage Examples

### Basic Usage
```python
from ema_alert_candle_strategy import main_analysis

# Analyze Apple stock for 1 year
strategy_result = main_analysis('AAPL', period='1y')
```

### Custom Configuration
```python
from ema_alert_candle_strategy import EMAAlertCandleStrategy, fetch_data

# Fetch data
data = fetch_data('TSLA', period='2y')

# Initialize custom strategy
strategy = EMAAlertCandleStrategy(
    data=data,
    ema_period=5,           # EMA period for alert candles
    risk_reward_ratio=4.0,  # Higher risk-reward ratio
    use_filters=False       # Disable additional filters
)

# Run analysis
strategy.prepare_data()
strategy.identify_alert_candles()
signals = strategy.generate_signals()
trades = strategy.backtest_strategy()
metrics = strategy.calculate_performance_metrics()

print(metrics)
```

## Key Features

### Technical Indicators
- 5-period and 20-period Exponential Moving Averages
- RSI (Relative Strength Index)
- Bollinger Bands
- Support and Resistance level detection

### Risk Management
- Position sizing based on risk percentage (default 2% of capital)
- Stop loss and take profit levels
- Risk-reward ratio enforcement
- Maximum drawdown calculation

### Performance Metrics
- Total return and Sharpe ratio
- Win rate and profit factor
- Average win/loss amounts
- Maximum drawdown
- Trade distribution analysis

### Visualization
- Price charts with EMA overlays
- Alert candle markers
- Trade entry/exit points
- RSI indicator
- Equity curve
- Trade P&L distribution

## Customization Options

### Strategy Parameters
- `ema_period`: EMA period for alert candles (default: 5)
- `risk_reward_ratio`: Minimum risk-reward ratio (default: 3.0)
- `use_filters`: Enable additional entry filters (default: True)
- `position_size_pct`: Risk percentage per trade (default: 2%)

### Data Parameters
- `symbol`: Stock symbol to analyze
- `period`: Data period ('1y', '2y', '5y', 'max')
- `interval`: Data interval ('1d', '1h', '5m')

## Performance Considerations

### Strengths
- Clear entry and exit rules
- Built-in risk management
- Comprehensive backtesting
- Visual analysis tools
- Web-based interface

### Limitations
- Strategy performance depends on market conditions
- May generate false signals in sideways markets
- Requires sufficient historical data
- Past performance doesn't guarantee future results

## Risk Disclaimer
This strategy is for educational purposes only. Trading involves risk and you should carefully consider your financial situation before trading. Past performance is not indicative of future results.

## Support and Customization
The code is fully customizable and can be modified to suit your specific needs. Key areas for customization:
- Alert candle detection logic
- Entry and exit conditions
- Risk management rules
- Performance metrics
- Visualization options

## Dependencies
- yfinance: Market data fetching
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib/seaborn: Visualization
- pandas-ta: Technical indicators (optional)
- backtesting: Professional backtesting (optional)
- flask: Web application framework
- gunicorn: WSGI HTTP Server
