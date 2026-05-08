# -*- coding: utf-8 -*-
"""
Reinforcement Learning Trading Strategy using Deep Q-Learning (Optimized)
Ticker: NVDA
Goal: Optimize Portfolio Value
"""

import sys
import io
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np

# Fix matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid TCL errors

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === Configuration ===
plt.style.use('ggplot') 
# Universal font settings to handle Chinese if present, otherwise fallback
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🚀 Starting RL Trading Strategy (Deep Q-Learning) for NVDA...")

# =========================================================
# 1. Data Acquisition (Robust Method)
# =========================================================
def download_data(ticker, days=730): # Increased days for better training data
    """Downloads stock data with robust handling for yfinance API changes."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📥 Downloading data for {ticker} from {start_date.date()} to {end_date.date()}...")
    
    try:
        # download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        # yfinance v0.2+ often returns MultiIndex columns. Flatten them if necessary.
        if isinstance(df.columns, pd.MultiIndex):
            try:
                # Try to drop the Ticker level if it exists
                df.columns = df.columns.droplevel(1)
            except:
                pass

        # Rename columns to standard names just in case
        df = df.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"})
        
        # Ensure we have data
        if df.empty:
            raise ValueError("Downloaded DataFrame is empty.")

        return df, df['Close'].values
        
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return None, None

ticker = 'NVDA'
# We use 2 years of data (approx 730 days) to give the AI more market cycles to learn from
df, prices = download_data(ticker, days=730)

if df is None:
    sys.exit(1)

print(f"✓ Data acquired: {len(df)} trading days.")

# =========================================================
# 2. Feature Engineering
# =========================================================
def create_features(df):
    """Generates technical indicators for the RL agent states."""
    features = df.copy()
    
    # 1. Simple Moving Averages
    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()
    
    # 2. RSI (Relative Strength Index)
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. Volatility (Standard Deviation of returns)
    features['Volatility'] = features['Close'].pct_change().rolling(window=20).std()
    
    # 4. MACD
    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
    
    # Drop NaN values created by rolling windows
    features = features.dropna()
    
    # Feature Selection for the Neural Network
    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'Volatility', 'MACD', 'Signal_Line']
    
    # Scaling (Neural Networks converge faster with scaled data 0-1)
    scaler = MinMaxScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])
    
    return features, feature_cols

df_features, feature_cols = create_features(df)
print(f"✓ Features created. Valid data points: {len(df_features)}")

# =========================================================
# 3. Trading Environment
# =========================================================
class TradingEnvironment:
    def __init__(self, df, initial_balance=10000, window_size=5):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.trades_history = [] # Stores (Action, Step, Price, Shares)
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
        # State is a concatenation of historical features + current account status
        start = self.current_step - self.window_size
        # Get historical window of technical indicators
        market_state = self.df[self.feature_cols].iloc[start:self.current_step].values.flatten()
        
        # Account state
        current_price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + (self.total_shares * current_price)
        
        # Normalize account info slightly to keep NN weights stable
        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 100, # Arbitrary scaling factor
            portfolio_value / self.initial_balance
        ])
        
        return np.concatenate((market_state, account_state))
    
    def step(self, action):
        # Actions: 0=Hold, 1=Buy, 2=Sell
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.total_shares * current_price)
        
        if action == 1: # BUY
            # Buy with 50% of available cash to be aggressive
            invest_amount = self.balance * 0.5
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.total_shares += shares_to_buy
                self.trades_history.append(('BUY', self.current_step, current_price, shares_to_buy))
                
        elif action == 2: # SELL
            # Sell 50% of holdings
            shares_to_sell = int(self.total_shares * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.balance += revenue
                self.total_shares -= shares_to_sell
                self.trades_history.append(('SELL', self.current_step, current_price, shares_to_sell))
        
        # Move to next day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # Calculate Reward
        next_price = float(self.df['Close'].iloc[self.current_step])
        new_portfolio_value = self.balance + (self.total_shares * next_price)
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Reward is percentage change in portfolio value
        reward = (new_portfolio_value - prev_portfolio_value) / self.initial_balance * 100
        
        if done:
            # Force sell all at end
            self.balance += self.total_shares * next_price
            self.total_shares = 0
            final_profit_pct = (self.balance - self.initial_balance) / self.initial_balance
            # Bonus reward for positive final profit
            if final_profit_pct > 0:
                reward += final_profit_pct * 10
        
        return self._get_state(), reward, done, new_portfolio_value
        
    def get_final_profit(self):
        return self.balance - self.initial_balance

# =========================================================
# 4. DQN Agent
# =========================================================
class DQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # Hyperparameters
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
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
    
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0] for i in minibatch])
        actions = np.array([self.memory[i][1] for i in minibatch])
        rewards = np.array([self.memory[i][2] for i in minibatch])
        next_states = np.array([self.memory[i][3] for i in minibatch])
        dones = np.array([self.memory[i][4] for i in minibatch])
        
        # Vectorized Target Calculation
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
env = TradingEnvironment(df_features)
# Calculate state size: (features * window) + account_vars
test_state = env.reset()
state_size = len(test_state)
agent = DQNAgent(state_size)

# ===silvia ------------------------------------------------------------------------------------------resource too few check 1 episode------------------======================================================

EPISODES = 10 # Increased from 1 to 10 for better training (more episodes = better learning)
# Note: 1 episode is not enough for the agent to learn effectively. The model needs 5-10+ episodes.
# Each episode lets the agent see the full dataset and learn from mistakes.
# ===silvia --------------------------------------why train the character model offered by kenny only need 5mins but this model need so long ----------------------------------------------------------------------======================================================

BATCH_SIZE = 32

print(f"\n3️⃣ Training Agent for {EPISODES} episodes...")

# Track best model
best_profit = -float('inf')
best_model_weights = None
best_episode = 0

for e in range(EPISODES):
    state = env.reset()
    total_reward = 0
    total_steps = len(env.df) - env.window_size - 1

    for time in range(total_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        # Show training progress every 100 steps
        if (time + 1) % 100 == 0:
            print(f"  Episode {e+1}/{EPISODES} - Step {time+1}/{total_steps} ({100*(time+1)/total_steps:.1f}%)")

        if done:
            episode_profit = env.get_final_profit()
            print(f"✓ Episode: {e+1}/{EPISODES} | Profit: ${episode_profit:.2f} | Epsilon: {agent.epsilon:.2f}")

            # Save best model weights
            if episode_profit > best_profit:
                best_profit = episode_profit
                best_episode = e + 1
                best_model_weights = agent.model.get_weights()
                print(f"  🌟 New best model! Episode {best_episode} with profit ${best_profit:.2f}")

            break

    # Train the model after every episode
    print(f"  Training neural network on {len(agent.memory)} experiences...")
    agent.replay(BATCH_SIZE)

# Restore best model for backtesting
print(f"\n✨ Restoring best model from Episode {best_episode} (profit: ${best_profit:.2f})")
agent.model.set_weights(best_model_weights)

# =========================================================
# 6. Backtesting (Validation)
# =========================================================
print("\n4️⃣ Running Final Backtest (No Exploration)...")
agent.epsilon = 0 # Turn off randomness to test what it learned
state = env.reset()
done = False

# Add progress tracking to show it's not stuck
steps_taken = 0
total_steps = len(env.df) - env.window_size - 1

# Track actions for debugging
action_counts = {0: 0, 1: 0, 2: 0}

while not done:
    # Get Q-values for debugging
    q_values = agent.model.predict(state.reshape(1, -1), verbose=0)
    action = agent.act(state)
    action_counts[action] += 1

    # Debug first 3 steps
    if steps_taken < 3:
        print(f"  Step {steps_taken}: Q-values={q_values[0]}, Action={action} ({'Hold' if action==0 else 'Buy' if action==1 else 'Sell'})")

    next_state, _, done, _ = env.step(action)
    state = next_state
    steps_taken += 1

    # Show progress every 50 steps
    if steps_taken % 50 == 0:
        print(f"  Backtest progress: {steps_taken}/{total_steps} steps ({100*steps_taken/total_steps:.1f}%)")

print(f"\n  Action distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")

final_profit = env.get_final_profit()
target_profit = 1280.40

print("\n" + "="*30)
print(f"FINAL RESULT FOR {ticker}")
print("="*30)
print(f"Initial Balance: $10,000.00")
print(f"Final Balance:   ${env.balance:.2f}")
print(f"Total Profit:    ${final_profit:.2f}")
print(f"Return (ROI):    {(final_profit/10000)*100:.2f}%")
print(f"Trades Executed: {len(env.trades_history)}")

if final_profit >= target_profit:
    print(f"🎉 SUCCESS! Target of ${target_profit} exceeded.")
else:
    print(f"⚠️ Target missed by ${target_profit - final_profit:.2f}.")

# =========================================================
# 7. Visualization
# =========================================================
plt.figure(figsize=(12, 8))

# Subplot 1: Portfolio Value
plt.subplot(2, 1, 1)
plt.plot(env.portfolio_value_history, label='Portfolio Value', color='green')
plt.axhline(y=10000, color='r', linestyle='--', alpha=0.3)
plt.title(f'RL Strategy Performance: {ticker}')
plt.ylabel('Value ($)')
plt.legend()

# Subplot 2: Trade Entries
plt.subplot(2, 1, 2)
# Align prices with the steps taken
price_data = env.df['Close'].iloc[env.window_size:].reset_index(drop=True)
plt.plot(price_data, label='Price', color='black', alpha=0.5)

# Plot Buy/Sell markers
for trade in env.trades_history:
    action, step, price, shares = trade
    # Adjust step index relative to the plot data
    idx = step - env.window_size
    if 0 <= idx < len(price_data):
        if action == 'BUY':
            plt.scatter(idx, price, color='green', marker='^', s=100)
        elif action == 'SELL':
            plt.scatter(idx, price, color='red', marker='v', s=100)

plt.title('Trade Entries (Green=Buy, Red=Sell)')
plt.legend(['Stock Price', 'Buy', 'Sell'])
plt.tight_layout()

# Save and Show
plt.savefig(f'{ticker}_RL_Result.png')
print(f"\n📊 Chart saved as {ticker}_RL_Result.png")
# plt.show() # Uncomment if running in Jupyter