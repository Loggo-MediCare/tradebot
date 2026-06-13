# -*- coding: utf-8 -*-
"""
Reinforcement Learning Trading Bot (Based on Hariom Chap 9 Case Study 1)
Reference: HARIOM_CHAP9_Total Profit 1280.40_ENG.pdf
Model: Deep Q-Network (DQN) with Experience Replay
"""
# !pip install yfinance pandas numpy matplotlib tensorflow -q

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# --- GPU Configuration ---
print("=" * 50)
print("GPU Configuration")
print("=" * 50)

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(gpus) > 0}")
print(f"Number of GPUs: {len(gpus)}")

if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus[0].name}")
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found, using CPU")
    print("To enable GPU:")
    print("  1. Install CUDA Toolkit")
    print("  2. Install cuDNN")
    print("  3. Install: pip install tensorflow[and-cuda]")

print("=" * 50)
print()

# --- 1. Helper Functions (PDF Page 21) ---

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    """
    Sigmoid function to normalize price differences.
    Maps values to range (0, 1).
    """
    return 1 / (1 + math.exp(-x))

def getState(data, t, n):
    """
    State representation:
    Input: price data, current time t, window size n
    Output: A vector of sigmoid(price_diff) for the window
    """
    d = t - n + 1
    # Handle the beginning of the time series (padding with the first price)
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]

    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

# --- 2. Agent Class (PDF Page 19-20) ---

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size # n-1 (window size - 1)
        self.action_size = 3 # Sit (Hold), Buy, Sell
        self.memory = deque(maxlen=1000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Load existing model or create new one
        if is_eval:
            try:
                self.model = load_model(model_name)
                print(f"Loaded model: {model_name}")
            except:
                print("Model not found, creating new one for evaluation.")
                self.model = self._model()
        else:
            self.model = self._model()

    def _model(self):
        """
        Deep Q-Network Architecture as described in PDF
        Input -> Dense(64) -> Dense(32) -> Dense(8) -> Output(3)
        """
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
        model.add(Dense(units=32, activation="relu"))
        model.add(Dense(units=8, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
        return model

    def act(self, state):
        """Action Selection: Epsilon-Greedy"""
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    def expReplay(self, batch_size):
        """Experience Replay: Training the model from memory"""
        mini_batch = []
        l = len(self.memory)

        # Sample the most recent memories (as per PDF logic snippet)
        # Note: Standard DQN usually samples randomly, but PDF snippet iterates range
        for i in range(l - batch_size + 1, l):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                # Bellman Equation
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            # Train the model for one epoch
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 3. Data Loading (Replaced CSV with yfinance) ---
# ----------------------SILVIA--------------------------
print("1. 下載數據 (S&P 500) ...")
# 為了節省時間，我們下載最近 3 年的數據 (PDF 範例是用 2010-2019)
ticker = "^GSPC"
df = yf.download(ticker, start="2020-01-01", end="2025-12-01", progress=False)

#---------------------------------------------------------
df.to_csv('my_stock_data.csv')
# The yfinance output for a single ticker usually has a flat column index,
# but the kernel state shows a MultiIndex. We will handle both cases.
#
data_prices = []

if isinstance(df.columns, pd.MultiIndex):
    # Handle MultiIndex columns
    # Prioritize 'Adj Close' then 'Close' under the 'Price' level
    if ('Price', 'Adj Close') in df.columns:
        data_prices = df[('Price', 'Adj Close')].tolist()
    elif ('Price', 'Close') in df.columns:
        data_prices = df[('Price', 'Close')].tolist()
    else:
        # Fallback if standard MultiIndex price columns are not found
        # Try to find any numeric column under 'Price'
        found = False
        for col_tuple in df.columns:
            if isinstance(col_tuple, tuple) and col_tuple[0] == 'Price' and pd.api.types.is_numeric_dtype(df[col_tuple]):
                data_prices = df[col_tuple].tolist()
                found = True
                break
        if not found:
            print("Warning: No suitable 'Price' column found in MultiIndex. Attempting to use the first column.")
            # Fallback to the first column, assuming it's the intended price data
            data_prices = df.iloc[:,0].tolist()
else:
    # Handle standard flat DataFrame columns
    if 'Adj Close' in df.columns:
        data_prices = df['Adj Close'].tolist()
    elif 'Close' in df.columns:
        data_prices = df['Close'].tolist()
    else:
        print("Warning: No 'Adj Close' or 'Close' column found in flat DataFrame. Attempting to use the first column.")
        # Fallback to the first column
        data_prices = df.iloc[:,0].tolist()

# Ensure all elements are floats. This step will now handle actual price data.
# The previous `data = [float(x) for x in data]` line caused the error because `data` was `['^GSPC']`.
# With `data_prices` now correctly populated with numerical values, this conversion should pass.
data = [float(x) for x in data_prices]
print(f"Data loaded: {len(data)} points")

# --- 4. Training (PDF Page 22-23) ---

print("\n2. 開始訓練 (Training) ...")
window_size = 10
agent = Agent(window_size)
batch_size = 32
episode_count = 10 # 10回合，平衡训练质量和时间

# Using a subset for training to save time in this demo
train_len = int(len(data) * 0.8)
train_data = data[:train_len]

# 跟踪最佳利潤以保存最佳模型
best_profit = float('-inf')
profit_history = []

for e in range(1, episode_count + 1):
    print(f"\nEpisode {e}/{episode_count}")
    state = getState(train_data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    # 統計交易次數
    buy_count = 0
    sell_count = 0

    # Iterate over the time series
    for t in range(len(train_data) - 1):
        action = agent.act(state)
        next_state = getState(train_data, t + 1, window_size + 1)
        reward = 0

        # Action Logic
        if action == 1: # Buy
            agent.inventory.append(train_data[t])
            buy_count += 1
            # print("Buy: " + formatPrice(train_data[t]))

        elif action == 2 and len(agent.inventory) > 0: # Sell
            bought_price = agent.inventory.pop(0)
            reward = max(train_data[t] - bought_price, 0)
            total_profit += train_data[t] - bought_price
            sell_count += 1
            # print("Sell: " + formatPrice(train_data[t]) + " | Profit: " + formatPrice(train_data[t] - bought_price))

        done = True if t == len(train_data) - 2 else False

        # Add to memory
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print(f"買入次數: {buy_count}, 賣出次數: {sell_count}")
            print(f"Epsilon (探索率): {agent.epsilon:.4f}")
            print("--------------------------------")

            # 保存利潤歷史
            profit_history.append(total_profit)

            # 如果這是最好的結果，保存模型
            if total_profit > best_profit:
                best_profit = total_profit
                agent.model.save('my_trading_bot.h5')
                print(f">>> 新的最佳利潤! 模型已保存: ${formatPrice(total_profit)}")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    # 每10個回合顯示平均進度
    if e % 10 == 0:
        recent_avg = sum(profit_history[-10:]) / min(10, len(profit_history))
        print(f"\n[進度報告] 最近10回合平均利潤: {formatPrice(recent_avg)}")
        print(f"[進度報告] 歷史最佳利潤: {formatPrice(best_profit)}\n")

print("\n" + "="*50)
print("訓練完成！")
print("="*50)
print(f"總回合數: {episode_count}")
print(f"最佳利潤: {formatPrice(best_profit)}")
print(f"平均利潤: {formatPrice(sum(profit_history) / len(profit_history))}")
print(f"最終探索率: {agent.epsilon:.4f}")
print("最佳模型已保存至: my_trading_bot.h5")
print("="*50)

# --- 5. Evaluation / Testing (PDF Page 26) ---

print("\n3. 開始回測 (Evaluation) - 就像 PDF 第 26 頁的 $1280.40 ...")

test_data = data[train_len:]
agent.is_eval = True # Turn off exploration
agent.epsilon = 0    # No random moves

state = getState(test_data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

buy_signals = []
sell_signals = []

for t in range(len(test_data) - 1):
    action = agent.act(state)
    next_state = getState(test_data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # Buy
        agent.inventory.append(test_data[t])
        buy_signals.append(t)
        # print("Buy: " + formatPrice(test_data[t]))

    elif action == 2 and len(agent.inventory) > 0: # Sell
        bought_price = agent.inventory.pop(0)
        profit = test_data[t] - bought_price
        total_profit += profit
        sell_signals.append(t)
        # print("Sell: " + formatPrice(test_data[t]) + " | Profit: " + formatPrice(profit))

    state = next_state

print("--------------------------------")
print(f"Test Total Profit: {formatPrice(total_profit)}")
print("--------------------------------")

# --- 6. Visualization ---
plt.figure(figsize=(12, 6))
plt.plot(test_data, color='gray', label='Price', alpha=0.5)
plt.plot(buy_signals, [test_data[i] for i in buy_signals], '^', markersize=8, color='green', label='Buy')
plt.plot(sell_signals, [test_data[i] for i in sell_signals], 'v', markersize=8, color='red', label='Sell')
plt.title(f'RL Trading Bot Evaluation - Profit: {formatPrice(total_profit)}')
plt.legend()
plt.show()