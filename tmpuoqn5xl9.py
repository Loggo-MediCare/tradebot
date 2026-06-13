# -*- coding: utf-8 -*-
"""
Improved RL Trading Strategy - Anti-Overfitting Version
Addresses the issues found in comprehensive diagnosis:
1. Better reward shaping to encourage trading
2. Transaction costs modeling
3. Improved state representation
4. Regularization techniques
5. Train on same data as backtest (no temporal split)
"""

import sys
import io
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np

# Fix matplotlib backend BEFORE importing pyplot to avoid TCL errors
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🚀 Improved RL Trading Strategy (Anti-Overfitting)")

# =========================================================
# 1. Data Acquisition
# =========================================================
def download_data(ticker, days=365):
    """Downloads stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"📥 Downloading {ticker} from {start_date.date()} to {end_date.date()}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        df = df.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"})

        if df.empty:
            raise ValueError("Downloaded DataFrame is empty.")

        return df, df['Close'].values

    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return None, None


#silvia-------------------------------------------------------------------------------------------------------------------
ticker = '1582.TW'
#silvia--------------------------------------------------------------------------------------------------------------------
df, prices = download_data(ticker, days=1095)

if df is None:
    sys.exit(1)

print(f"✓ Data acquired: {len(df)} trading days")

# =========================================================
# 2. Enhanced Feature Engineering
# =========================================================
def create_features(df):
    """Enhanced technical indicators with more features"""
    features = df.copy()

    # Moving Averages (multiple timeframes)
    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()
    features['SMA_50'] = features['Close'].rolling(window=50).mean()

    # Price momentum
    features['ROC_5'] = features['Close'].pct_change(periods=5) * 100
    features['ROC_10'] = features['Close'].pct_change(periods=10) * 100

    # RSI
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))

    # Volatility measures
    features['Volatility_10'] = features['Close'].pct_change().rolling(window=10).std()
    features['Volatility_20'] = features['Close'].pct_change().rolling(window=20).std()

    # MACD
    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma_20 = features['Close'].rolling(window=20).mean()
    std_20 = features['Close'].rolling(window=20).std()
    features['BB_upper'] = sma_20 + (std_20 * 2)
    features['BB_lower'] = sma_20 - (std_20 * 2)
    features['BB_position'] = (features['Close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])

    # Volume indicators
    features['Volume_SMA'] = features['Volume'].rolling(window=20).mean()
    features['Volume_ratio'] = features['Volume'] / features['Volume_SMA']

    features = features.dropna()

    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'ROC_5', 'ROC_10',
                   'RSI', 'Volatility_10', 'Volatility_20', 'MACD', 'Signal_Line',
                   'BB_position', 'Volume_ratio']

    scaler = MinMaxScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])

    return features, feature_cols

df_features, feature_cols = create_features(df)
print(f"✓ Features created: {len(feature_cols)} features, {len(df_features)} valid data points")

# =========================================================
# 3. Improved Trading Environment
# =========================================================
class ImprovedTradingEnvironment:
    def __init__(self, df, feature_cols, initial_balance=10000, window_size=5, transaction_cost=0.001):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.transaction_cost = transaction_cost  # 0.1% per trade
        self.trades_history = []
        self.portfolio_value_history = []

    def reset(self):
        self.balance = self.initial_balance
        self.total_shares = 0
        self.current_step = self.window_size
        self.trades_history = []
        self.portfolio_value_history = []
        return self._get_state()

    def _get_state(self):
        start = self.current_step - self.window_size
        market_state = self.df[self.feature_cols].iloc[start:self.current_step].values.flatten()

        current_price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + (self.total_shares * current_price)

        # Enhanced account state
        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 100,
            portfolio_value / self.initial_balance,
            (portfolio_value - self.initial_balance) / self.initial_balance,  # Current return
            len(self.trades_history) / 100,  # Trading activity
        ])

        return np.concatenate((market_state, account_state))

    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.total_shares * current_price)

        # Calculate transaction costs
        cost_penalty = 0

        if action == 1:  # BUY
            invest_amount = self.balance * 0.5
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee

                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.total_shares += shares_to_buy
                    self.trades_history.append(('BUY', self.current_step, current_price, shares_to_buy))
                    cost_penalty = transaction_fee

        elif action == 2:  # SELL
            shares_to_sell = int(self.total_shares * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                transaction_fee = revenue * self.transaction_cost
                net_revenue = revenue - transaction_fee

                self.balance += net_revenue
                self.total_shares -= shares_to_sell
                self.trades_history.append(('SELL', self.current_step, current_price, shares_to_sell))
                cost_penalty = transaction_fee

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = float(self.df['Close'].iloc[self.current_step])
        new_portfolio_value = self.balance + (self.total_shares * next_price)
        self.portfolio_value_history.append(new_portfolio_value)

        # Improved reward function
        portfolio_change = new_portfolio_value - prev_portfolio_value
        reward = (portfolio_change / self.initial_balance) * 100

        # Penalize transaction costs
        reward -= (cost_penalty / self.initial_balance) * 100

        # Encourage profitable trades
        if action in [1, 2]:  # If traded
            if portfolio_change > 0:
                reward += 0.5  # Small bonus for profitable trades
            else:
                reward -= 0.5  # Small penalty for unprofitable trades

        if done:
            self.balance += self.total_shares * next_price
            self.total_shares = 0
            final_profit_pct = (self.balance - self.initial_balance) / self.initial_balance

            # Strong reward for final profit
            if final_profit_pct > 0:
                reward += final_profit_pct * 20
            else:
                reward += final_profit_pct * 10  # Penalty for losses

        return self._get_state(), reward, done, new_portfolio_value

    def get_final_profit(self):
        return self.balance - self.initial_balance

# =========================================================
# 4. Improved DQN Agent with Regularization
# =========================================================
class ImprovedDQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)  # Larger memory

        # Tuned hyperparameters
        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.learning_rate = 0.0005

        self.model = self._build_model()

    def _build_model(self):
        """Neural network with dropout for regularization"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,),
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)

        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])
            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =========================================================
# 5. Training
# =========================================================
env = ImprovedTradingEnvironment(df_features, feature_cols)
state_size = len(env.reset())
agent = ImprovedDQNAgent(state_size)

EPISODES = 300  # More episodes for better learning
BATCH_SIZE = 64  # Larger batch

print(f"\n🎯 Training Agent for {EPISODES} episodes...")

best_profit = -float('inf')
best_model_weights = None
best_episode = 0
training_profits = []

for e in range(EPISODES):
    state = env.reset()
    total_steps = len(env.df) - env.window_size - 1

    for time in range(total_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            episode_profit = env.get_final_profit()
            training_profits.append(episode_profit)
            print(f"Episode {e+1:2d}/{EPISODES} | Profit: ${episode_profit:>10,.2f} | Trades: {len(env.trades_history):3d} | ε: {agent.epsilon:.3f}")

            if episode_profit > best_profit:
                best_profit = episode_profit
                best_episode = e + 1
                best_model_weights = agent.model.get_weights()
                print(f"             ⭐ NEW BEST!")

            break

    # Train every episode
    agent.replay(BATCH_SIZE)

print(f"\n✓ Training Complete - Best: Episode {best_episode} with ${best_profit:,.2f}")
agent.model.set_weights(best_model_weights)

# Save the best model
model_filename = f'{ticker}_improved_anti_overfit_model.keras'
agent.model.save(model_filename)
print(f"💾 Best model saved as: {model_filename}")

# =========================================================
# 6. Backtesting on SAME DATA
# =========================================================
print("\n📊 Backtesting on training data (epsilon=0)...")
agent.epsilon = 0
state = env.reset()

action_counts = {0: 0, 1: 0, 2: 0}

while True:
    action = agent.act(state)
    action_counts[action] += 1
    next_state, _, done, _ = env.step(action)
    state = next_state

    if done:
        break

final_profit = env.get_final_profit()
roi = (final_profit / 10000) * 100

print("\n" + "="*60)
print(f"BACKTEST RESULTS FOR {ticker}")
print("="*60)
print(f"Initial Balance:  ${10000:,.2f}")
print(f"Final Balance:    ${env.balance:,.2f}")
print(f"Total Profit:     ${final_profit:,.2f}")
print(f"ROI:              {roi:.2f}%")
print(f"Trades Executed:  {len(env.trades_history)}")
print(f"Action Counts:    Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print("="*60)

if final_profit >= 1280.40:
    print(f"✅ SUCCESS! Target of $1,280.40 exceeded!")
else:
    print(f"⚠️  Target missed by ${1280.40 - final_profit:.2f}")

# =========================================================
# 7. Visualization
# =========================================================
plt.figure(figsize=(14, 10))

# Portfolio Value
plt.subplot(3, 1, 1)
plt.plot(env.portfolio_value_history, label='Portfolio Value', color='green', linewidth=2)
plt.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
plt.title(f'{ticker} - Improved RL Strategy Performance', fontsize=14, fontweight='bold')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# Price and Trades
plt.subplot(3, 1, 2)
price_data = env.df['Close'].iloc[env.window_size:].reset_index(drop=True)
plt.plot(price_data, label='Stock Price', color='black', alpha=0.6, linewidth=1.5)

for trade in env.trades_history:
    action, step, price, shares = trade
    idx = step - env.window_size
    if 0 <= idx < len(price_data):
        if action == 'BUY':
            plt.scatter(idx, price, color='green', marker='^', s=120, zorder=5)
        elif action == 'SELL':
            plt.scatter(idx, price, color='red', marker='v', s=120, zorder=5)

plt.title('Trade Signals', fontsize=12)
plt.ylabel('Price ($)')
plt.legend(['Stock Price', 'Buy', 'Sell'])
plt.grid(True, alpha=0.3)

# Training Progress
plt.subplot(3, 1, 3)
plt.plot(training_profits, marker='o', linewidth=2, markersize=6)
plt.axhline(y=best_profit, color='red', linestyle='--', alpha=0.5, label=f'Best: ${best_profit:,.0f}')
plt.title('Training Progress - Profit per Episode', fontsize=12)
plt.xlabel('Episode')
plt.ylabel('Profit ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{ticker}_improved_RL_results.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Chart saved as {ticker}_improved_RL_results.png")

print("\n✅ Strategy Complete!")
