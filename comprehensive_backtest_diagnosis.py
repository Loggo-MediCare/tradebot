# -*- coding: utf-8 -*-
"""
Comprehensive Backtesting Diagnosis for RL Trading Bot
This script tests for overfitting and generalization issues
"""

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import from the working Gemini version
import importlib.util
spec = importlib.util.spec_from_file_location("gemini_bot",
    r"c:\Users\Silvi\Projects\trading-bot\rl_trading_strategy_1280_by gemini 20251212.py")
# We'll just copy the classes instead

# =========================================================
# Copy necessary classes from Gemini version
# =========================================================

def download_data(ticker, days=730):
    """Downloads stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"📥 Downloading data for {ticker} from {start_date.date()} to {end_date.date()}...")

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

def create_features(df):
    """Generates technical indicators"""
    features = df.copy()

    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()

    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))

    features['Volatility'] = features['Close'].pct_change().rolling(window=20).std()

    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()

    features = features.dropna()

    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'Volatility', 'MACD', 'Signal_Line']

    scaler = MinMaxScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])

    return features, feature_cols

class TradingEnvironment:
    def __init__(self, df, feature_cols, initial_balance=10000, window_size=5):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.trades_history = []
        self.portfolio_value_history = []
        self.feature_cols = feature_cols

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

        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 100,
            portfolio_value / self.initial_balance
        ])

        return np.concatenate((market_state, account_state))

    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.total_shares * current_price)

        if action == 1:  # BUY
            invest_amount = self.balance * 0.5
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.total_shares += shares_to_buy
                self.trades_history.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # SELL
            shares_to_sell = int(self.total_shares * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.balance += revenue
                self.total_shares -= shares_to_sell
                self.trades_history.append(('SELL', self.current_step, current_price, shares_to_sell))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = float(self.df['Close'].iloc[self.current_step])
        new_portfolio_value = self.balance + (self.total_shares * next_price)
        self.portfolio_value_history.append(new_portfolio_value)

        reward = (new_portfolio_value - prev_portfolio_value) / self.initial_balance * 100

        if done:
            self.balance += self.total_shares * next_price
            self.total_shares = 0
            final_profit_pct = (self.balance - self.initial_balance) / self.initial_balance
            if final_profit_pct > 0:
                reward += final_profit_pct * 10

        return self._get_state(), reward, done, new_portfolio_value

    def get_final_profit(self):
        return self.balance - self.initial_balance

class DQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
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

    def get_q_values(self, state):
        """Get Q-values without taking action"""
        return self.model.predict(state.reshape(1, -1), verbose=0)[0]

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
# DIAGNOSTIC TESTS
# =========================================================

print("="*70)
print("COMPREHENSIVE RL TRADING BOT DIAGNOSIS")
print("="*70)

# 1. Download and prepare data
ticker = 'NVDA'
df, prices = download_data(ticker, days=730)

if df is None:
    sys.exit(1)

print(f"✓ Data acquired: {len(df)} trading days")

# 2. Create features
df_features, feature_cols = create_features(df)
print(f"✓ Features created: {len(df_features)} valid data points")

# 3. Split into train/test (80/20)
train_size = int(len(df_features) * 0.8)
train_data = df_features.iloc[:train_size].copy()
test_data = df_features.iloc[train_size:].copy()

print(f"\n📊 Data Split:")
print(f"  Training: {len(train_data)} days ({train_data.index[0]} to {train_data.index[-1]})")
print(f"  Testing:  {len(test_data)} days ({test_data.index[0]} to {test_data.index[-1]})")

# 4. Train model with best model tracking
print("\n" + "="*70)
print("PHASE 1: TRAINING")
print("="*70)

env_train = TradingEnvironment(train_data, feature_cols)
state_size = len(env_train.reset())
agent = DQNAgent(state_size)

EPISODES = 10
BATCH_SIZE = 32
best_profit = -float('inf')
best_model_weights = None
best_episode = 0
training_profits = []

for e in range(EPISODES):
    state = env_train.reset()
    total_steps = len(env_train.df) - env_train.window_size - 1

    for time in range(total_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env_train.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            episode_profit = env_train.get_final_profit()
            training_profits.append(episode_profit)
            print(f"Episode {e+1}/{EPISODES} | Profit: ${episode_profit:,.2f} | Epsilon: {agent.epsilon:.3f}")

            if episode_profit > best_profit:
                best_profit = episode_profit
                best_episode = e + 1
                best_model_weights = agent.model.get_weights()
                print(f"  ⭐ NEW BEST MODEL!")

            break

    agent.replay(BATCH_SIZE)

print(f"\n✓ Training Complete - Best Episode: {best_episode} with ${best_profit:,.2f}")
agent.model.set_weights(best_model_weights)

# =========================================================
# DIAGNOSTIC TEST 1: Train Data with Exploitation Only
# =========================================================
print("\n" + "="*70)
print("TEST 1: Performance on TRAINING data (epsilon=0)")
print("="*70)

agent.epsilon = 0
env_train_test = TradingEnvironment(train_data, feature_cols)
state = env_train_test.reset()

q_values_train = []
actions_train = []

while True:
    q_vals = agent.get_q_values(state)
    action = agent.act(state)
    q_values_train.append(q_vals)
    actions_train.append(action)

    next_state, _, done, _ = env_train_test.step(action)
    state = next_state

    if done:
        break

train_exploit_profit = env_train_test.get_final_profit()
train_action_counts = {0: actions_train.count(0), 1: actions_train.count(1), 2: actions_train.count(2)}

print(f"Training Data Exploitation Profit: ${train_exploit_profit:,.2f}")
print(f"Actions: Hold={train_action_counts[0]}, Buy={train_action_counts[1]}, Sell={train_action_counts[2]}")
print(f"Trades Executed: {len(env_train_test.trades_history)}")

if train_exploit_profit < best_profit * 0.5:
    print("⚠️  WARNING: Model performs much worse without exploration on training data!")
    print("   This suggests the model hasn't learned a good exploitative policy.")

# =========================================================
# DIAGNOSTIC TEST 2: Test Data Performance
# =========================================================
print("\n" + "="*70)
print("TEST 2: Performance on UNSEEN TEST data (epsilon=0)")
print("="*70)

env_test = TradingEnvironment(test_data.reset_index(drop=True), feature_cols)
state = env_test.reset()

q_values_test = []
actions_test = []

while True:
    q_vals = agent.get_q_values(state)
    action = agent.act(state)
    q_values_test.append(q_vals)
    actions_test.append(action)

    next_state, _, done, _ = env_test.step(action)
    state = next_state

    if done:
        break

test_profit = env_test.get_final_profit()
test_action_counts = {0: actions_test.count(0), 1: actions_test.count(1), 2: actions_test.count(2)}

print(f"Test Data Profit: ${test_profit:,.2f}")
print(f"Actions: Hold={test_action_counts[0]}, Buy={test_action_counts[1]}, Sell={test_action_counts[2]}")
print(f"Trades Executed: {len(env_test.trades_history)}")

# =========================================================
# DIAGNOSTIC TEST 3: Q-Value Analysis
# =========================================================
print("\n" + "="*70)
print("TEST 3: Q-Value Analysis")
print("="*70)

# Analyze first 10 steps of test data
print("\nFirst 10 steps of TEST data:")
for i in range(min(10, len(q_values_test))):
    q = q_values_test[i]
    a = actions_test[i]
    print(f"  Step {i}: Q=[{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}] → Action={a} ({'Hold' if a==0 else 'Buy' if a==1 else 'Sell'})")

# Calculate average Q-values
avg_q_train = np.mean(q_values_train, axis=0)
avg_q_test = np.mean(q_values_test, axis=0)

print(f"\nAverage Q-values:")
print(f"  Training: Hold={avg_q_train[0]:.4f}, Buy={avg_q_train[1]:.4f}, Sell={avg_q_train[2]:.4f}")
print(f"  Test:     Hold={avg_q_test[0]:.4f}, Buy={avg_q_test[1]:.4f}, Sell={avg_q_test[2]:.4f}")

# =========================================================
# FINAL SUMMARY AND DIAGNOSIS
# =========================================================
print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print(f"\n1. TRAINING PERFORMANCE:")
print(f"   Best Episode Profit: ${best_profit:,.2f}")
print(f"   Exploitation-only:   ${train_exploit_profit:,.2f}")
print(f"   Degradation:         {((best_profit - train_exploit_profit) / best_profit * 100):.1f}%")

print(f"\n2. GENERALIZATION (Test Performance):")
print(f"   Test Profit: ${test_profit:,.2f}")
print(f"   Success: {'✓ YES' if test_profit > 1280.40 else '✗ NO'}")

print(f"\n3. OVERFITTING INDICATORS:")
performance_drop = ((train_exploit_profit - test_profit) / max(train_exploit_profit, 1)) * 100
print(f"   Performance drop: {performance_drop:.1f}%")

if performance_drop > 50:
    print("   ⚠️  HIGH OVERFITTING DETECTED")
elif performance_drop > 20:
    print("   ⚠️  MODERATE OVERFITTING")
else:
    print("   ✓ Good generalization")

print(f"\n4. ACTION DISTRIBUTION:")
print(f"   Training: Hold={train_action_counts[0]}, Buy={train_action_counts[1]}, Sell={train_action_counts[2]}")
print(f"   Test:     Hold={test_action_counts[0]}, Buy={test_action_counts[1]}, Sell={test_action_counts[2]}")

if test_action_counts[1] == 0 and test_action_counts[2] == 0:
    print("   ⚠️  CRITICAL: Model only chooses HOLD on test data!")
    print("      This indicates severe overfitting or collapsed policy.")

print("\n" + "="*70)
print("END OF DIAGNOSIS")
print("="*70)
