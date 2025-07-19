import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SP500TradingBot:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.data = None
        
    def download_data(self, start_date='2015-01-01', end_date='2025-07-19'):
        """Download S&P 500 data"""
        print(f"Downloading S&P 500 data from {start_date} to {end_date}")
        self.data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        
        # Flatten multi-index columns if present
        if isinstance(self.data.columns, pd.MultiIndex):
            self.data.columns = [col[0] for col in self.data.columns]
        
        print(f"Downloaded {len(self.data)} trading days of data")
        print(f"Column names: {list(self.data.columns)}")
        return self.data
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Price-based features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Pct'] = (df['Close'] - df['Open']) / df['Open']
        
        # Moving averages
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        
        # Price relative to moving averages
        df['Price_SMA5_Ratio'] = df['Close'] / df['SMA_5']
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
        
        # Technical indicators
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['BB_upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['BB_lower'] = ta.volatility.bollinger_lband(df['Close'])
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Momentum indicators
        df['ROC'] = ta.momentum.roc(df['Close'], window=10)
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'Returns_lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Ratio_lag_{lag}'] = df['Volume_Ratio'].shift(lag)
        
        return df
    
    def create_target(self, data, holding_period=1):
        """Create target variable: 1 if price goes up in next holding_period days, 0 otherwise"""
        df = data.copy()
        df['Future_Return'] = df['Close'].shift(-holding_period) / df['Close'] - 1
        df['Target'] = (df['Future_Return'] > 0).astype(int)
        return df
    
    def prepare_training_data(self, start_date='2015-01-01', end_date='2020-12-31'):
        """Prepare training dataset"""
        print(f"Preparing training data from {start_date} to {end_date}")
        
        # Filter training period
        training_data = self.data[start_date:end_date].copy()
        
        # Create features
        training_data = self.create_features(training_data)
        training_data = self.create_target(training_data)
        
        # Select feature columns
        feature_cols = [col for col in training_data.columns if col not in 
                       ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 
                        'Future_Return', 'Target', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
                        'BB_upper', 'BB_lower', 'Volume_SMA']]
        
        self.feature_columns = feature_cols
        
        # Remove NaN values
        training_data = training_data.dropna()
        
        X = training_data[feature_cols]
        y = training_data['Target']
        
        print(f"Training features: {len(feature_cols)}")
        print(f"Training samples: {len(X)}")
        print(f"Positive samples: {y.sum()} ({y.mean():.2%})")
        
        return X, y, training_data
    
    def train_model(self, X, y):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Validation
        val_pred = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nValidation Classification Report:")
        print(classification_report(y_val, val_pred))
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10))
        
        return self.model
    
    def backtest(self, start_date='2021-01-01', end_date='2025-07-19', initial_capital=10000):
        """Backtest the strategy on out-of-sample data"""
        print(f"Backtesting from {start_date} to {end_date}")
        
        # Filter test period
        test_data = self.data[start_date:end_date].copy()
        test_data = self.create_features(test_data)
        test_data = self.create_target(test_data)
        test_data = test_data.dropna()
        
        # Generate predictions
        X_test = test_data[self.feature_columns]
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)[:, 1]
        
        # Create trading signals
        test_data['Signal'] = predictions
        test_data['Signal_Prob'] = probabilities
        
        # Backtest strategy
        capital = initial_capital
        position = 0
        portfolio_values = [capital]
        trades = []
        
        for i in range(1, len(test_data)):
            current_price = test_data['Close'].iloc[i]
            previous_price = test_data['Close'].iloc[i-1]
            signal = test_data['Signal'].iloc[i-1]  # Use previous day's signal
            
            if signal == 1 and position == 0:  # Buy signal
                shares = capital / current_price
                position = shares
                capital = 0
                trades.append(('BUY', test_data.index[i], current_price, shares))
                
            elif signal == 0 and position > 0:  # Sell signal
                capital = position * current_price
                trades.append(('SELL', test_data.index[i], current_price, position))
                position = 0
            
            # Calculate portfolio value
            if position > 0:
                portfolio_value = position * current_price
            else:
                portfolio_value = capital
                
            portfolio_values.append(portfolio_value)
        
        # Final sell if still holding
        if position > 0:
            final_price = test_data['Close'].iloc[-1]
            capital = position * final_price
            trades.append(('SELL', test_data.index[-1], final_price, position))
        
        # Calculate performance metrics
        # Ensure portfolio_values has the same length as test_data
        if len(portfolio_values) > len(test_data):
            portfolio_values = portfolio_values[:len(test_data)]
        elif len(portfolio_values) < len(test_data):
            portfolio_values.extend([portfolio_values[-1]] * (len(test_data) - len(portfolio_values)))
        
        test_data['Portfolio_Value'] = portfolio_values
        test_data['Buy_Hold_Value'] = initial_capital * (test_data['Close'] / test_data['Close'].iloc[0])
        
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        buy_hold_return = (test_data['Close'].iloc[-1] - test_data['Close'].iloc[0]) / test_data['Close'].iloc[0]
        
        # Calculate Sharpe ratio
        daily_returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
        
        print(f"\n=== Backtest Results ===")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"Excess Return: {total_return - buy_hold_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Number of Trades: {len(trades)}")
        
        return test_data, trades, portfolio_values
    
    def plot_results(self, test_data, portfolio_values):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value vs Buy & Hold
        axes[0, 0].plot(test_data.index, test_data['Portfolio_Value'], label='ML Strategy', linewidth=2)
        axes[0, 0].plot(test_data.index, test_data['Buy_Hold_Value'], label='Buy & Hold', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Comparison')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # S&P 500 price with signals
        buy_signals = test_data[test_data['Signal'] == 1]
        sell_signals = test_data[test_data['Signal'] == 0]
        
        axes[0, 1].plot(test_data.index, test_data['Close'], label='S&P 500', color='black', alpha=0.7)
        axes[0, 1].scatter(buy_signals.index, buy_signals['Close'], color='green', marker='^', s=30, label='Buy Signal', alpha=0.7)
        axes[0, 1].scatter(sell_signals.index, sell_signals['Close'], color='red', marker='v', s=30, label='Sell Signal', alpha=0.7)
        axes[0, 1].set_title('S&P 500 Price with Trading Signals')
        axes[0, 1].set_ylabel('Price ($)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily returns distribution
        strategy_returns = test_data['Portfolio_Value'].pct_change().dropna()
        market_returns = test_data['Close'].pct_change().dropna()
        
        axes[1, 0].hist(strategy_returns, bins=50, alpha=0.7, label='ML Strategy', density=True)
        axes[1, 0].hist(market_returns, bins=50, alpha=0.7, label='Buy & Hold', density=True)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance (top 10)
        if self.model:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 1].barh(range(len(importance_df)), importance_df['importance'])
            axes[1, 1].set_yticks(range(len(importance_df)))
            axes[1, 1].set_yticklabels(importance_df['feature'])
            axes[1, 1].set_title('Top 10 Feature Importance')
            axes[1, 1].set_xlabel('Importance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/andresmorat/Documents/dev/TradingBot/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Initialize trading bot
    bot = SP500TradingBot()
    
    # Download data
    data = bot.download_data()
    
    # Prepare training data (2015-2020)
    X_train, y_train, train_data = bot.prepare_training_data()
    
    # Train model
    model = bot.train_model(X_train, y_train)
    
    # Backtest on 2021-2025
    test_data, trades, portfolio_values = bot.backtest()
    
    # Plot results
    bot.plot_results(test_data, portfolio_values)

if __name__ == "__main__":
    main()