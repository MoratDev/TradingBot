# S&P 500 ML Trading Bot

A machine learning-based trading bot that uses technical analysis and Random Forest classification to predict S&P 500 price movements.

## Overview

This project implements an algorithmic trading strategy using machine learning to predict daily S&P 500 price movements. The bot uses 22 technical indicators and a Random Forest classifier to generate buy/sell signals.

## Features

- **Data Source**: Yahoo Finance API for S&P 500 historical data
- **Machine Learning Model**: Random Forest Classifier
- **Technical Indicators**: 22 features including:
  - Moving averages (5, 10, 20, 50 days)
  - RSI, MACD, Bollinger Bands
  - Volume indicators and ratios
  - Average True Range (ATR)
  - Lagged price returns
- **Backtesting Engine**: Built-in performance evaluation
- **Visualization**: Performance charts and results analysis

## Performance Summary

- **Training Period**: 2015-2020 (6 years)
- **Testing Period**: 2021-2025 (4.5 years)
- **Total Return**: 32.86% vs 58.92% buy & hold
- **Sharpe Ratio**: 0.51
- **Number of Trades**: 350

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MoratDev/TradingBot.git
cd TradingBot
```

2. Create a virtual environment:
```bash
python -m venv trading_env
source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the trading bot:
```bash
python sp500_trading_bot.py
```

The script will:
1. Download S&P 500 data from Yahoo Finance
2. Create technical indicators and features
3. Train the Random Forest model
4. Run backtesting simulation
5. Generate performance visualization and summary

## Files

- `sp500_trading_bot.py` - Main trading algorithm and backtesting engine
- `requirements.txt` - Python package dependencies
- `trading_summary.md` - Detailed performance analysis and insights
- `backtest_results.png` - Performance visualization chart

## Key Insights

- The model achieved ~50% prediction accuracy, typical for financial markets
- Most important features were lagged returns and volume ratios
- Strategy underperformed buy & hold due to market efficiency and transaction costs
- Shows potential for improvement with additional features and optimization

## Disclaimer

This is an educational project for learning machine learning applications in finance. It is not intended as investment advice. Past performance does not guarantee future results. Always consult with financial professionals before making investment decisions.

## License

This project is open source and available under the MIT License.