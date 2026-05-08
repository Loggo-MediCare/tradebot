# -*- coding: utf-8 -*-
"""
Reinforcement Learning Trading Bot (Based on Hariom Chap 9 Case Study 1)
Reference: HARIOM_CHAP9_Total Profit 1280.40_ENG.pdf
Model: Deep Q-Network (DQN) with Experience Replay (Vectorized / Faster Version)
"""

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
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]

    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    # 原始實作回傳 shape = (1, state_size)
    return np.array([res])

# --- 2. Agent Class (PDF Page 19-20) ---

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # n-1 (window size - 1)
        self.action_size = 3  # Sit (Hold), Buy, Sell
        self.memory = deque(maxlen=2000)   # 稍微放大 buffer
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95        # Discount factor
        self.epsilon = 1.0       # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

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

        # state shape 原本是 (1, state_size)，這邊沿用
        options = self.model.predict(state, verbose=0)
        return np.argmax(options[0])

    # ===================== 這是重構後的 expReplay =====================
    def expReplay(self, batch_size):
        """
        Vectorized Experience Replay:
        - 使用隨機抽樣 mini-batch
        - 一次 predict 一整個 batch
        - 一次 fit 一整個 batch
        速度會比逐筆呼叫 model.fit() 快非常多
        """
        if len(self.memory) == 0:
            return

        sample_size = min(len(self.memory), batch_size)
        mini_batch = random.sample(self.memory, sample_size)

        # memory 中的 state / next_state shape 原本是 (1, state_size)
        states = np.vstack([m[0] for m in mini_batch])         # (batch, 1, state_size)
        next_states = np.vstack([m[3] for m in mini_batch])    # (batch, 1, state_size)

        # 攤平成 (batch, state_size)
        states = states.reshape(sample_size, -1)
        next_states = next_states.reshape(sample_size, -1)

        actions = np.array([m[1] for m in mini_batch])
        rewards = np.array([m[2] for m in mini_batch])
        dones = np.array([m[4] for m in mini_batch])

        # 一次算 Q(s,a) 和 Q(s',a')
        q_values = self.model.predict(states, verbose=0)
        q_next = self.model.predict(next_states, verbose=0)

        for i in range(sample_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(q_next[i])
            q_values[i][actions[i]] = target

        # 一次 fit 整個 batch
        self.model.fit(states, q_values, epochs=1, verbose=0)

        # 更新 epsilon（探索率）
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# --- 3. Data Loading (Replaced CSV with yfinance) ---
# ----------------------SILVIA--------------------------
print("1. 下載數據 (S&P 500) ...")

ticker = "^GSPC"
df = yf.download(ticker, start="2020-01-01", end="2025-12-01", progress=False)

df.to_csv('my_stock_data.csv')

data_prices = []

if isinstance(df.columns, pd.MultiIndex):
    if ('Price', 'Adj Close') in df.columns:
        data_prices = df[('Price', 'Adj Close')].tolist()
    elif ('Price', 'Close') in df.columns:
        data_prices = df[('Price', 'Close')].tolist()
    else:
        found = False
        for col_tuple in df.columns:
            if isinstance(col_tuple, tuple) and col_tuple[0] == 'Price' and pd.api.types.is_numeric_dtype(df[col_tuple]):
                data_prices = df[col_tuple].tolist()
                found = True
                break
        if not found:
            print("Warning: No suitable 'Price' column found in MultiIndex. Attempting to use the first column.")
            data_prices = df.iloc[:, 0].tolist()
else:
    if 'Adj Close' in df.columns:
        data_prices = df['Adj Close'].tolist()
    elif 'Close' in df.columns:
        data_prices = df['Close'].tolist()
    else:
        print("Warning: No 'Adj Close' or 'Close' column found in flat DataFrame. Attempting to use the first column.")
        data_prices = df.iloc[:, 0].tolist()

data = [float(x) for x in data_prices]
print(f"Data loaded: {len(data)} points")

# --- 4. Training (PDF Page 22-23) ---

print("\n2. 開始訓練 (Training) ...")
window_size = 10
agent = Agent(window_size)
batch_size = 32

# Episode 數量
episode_count = 10  # Changed from 1 to 10 for better learning

# 使用 80% 資料當訓練
train_len = int(len(data) * 0.8)
train_data = data[:train_len]

best_profit = float('-inf')
profit_history = []

for e in range(1, episode_count + 1):
    print(f"\nEpisode {e}/{episode_count}")
    state = getState(train_data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    buy_count = 0
    sell_count = 0

    for t in range(len(train_data) - 1):
        action = agent.act(state)
        next_state = getState(train_data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # Buy
            agent.inventory.append(train_data[t])
            buy_count += 1

        elif action == 2 and len(agent.inventory) > 0:  # Sell
            bought_price = agent.inventory.pop(0)
            reward = max(train_data[t] - bought_price, 0)
            total_profit += train_data[t] - bought_price
            sell_count += 1

        done = True if t == len(train_data) - 2 else False

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        # ✅ 不再每一步都訓練，只在每 10 步訓練一次，加速很多
        if len(agent.memory) >= batch_size and (t % 10 == 0 or done):
            agent.expReplay(batch_size)

        if done:
            print("--------------------------------")
            print(f"Total Profit: {formatPrice(total_profit)}")
            print(f"買入次數: {buy_count}, 賣出次數: {sell_count}")
            print(f"Epsilon (探索率): {agent.epsilon:.4f}")
            print("--------------------------------")

            profit_history.append(total_profit)

            if total_profit > best_profit:
                best_profit = total_profit
                agent.model.save('my_trading_bot.h5')
                print(f">>> 新的最佳利潤! 模型已保存: {formatPrice(total_profit)}")

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
agent.is_eval = True
agent.epsilon = 0  # 關掉隨機行為

state = getState(test_data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

buy_signals = []
sell_signals = []

# Track actions for debugging
action_counts = {0: 0, 1: 0, 2: 0}

for t in range(len(test_data) - 1):
    action = agent.act(state)
    action_counts[action] += 1

    # Debug first 5 steps
    if t < 5:
        q_values = agent.model.predict(state, verbose=0)
        print(f"Step {t}: Q={q_values[0]}, Action={action} ({'Hold' if action==0 else 'Buy' if action==1 else 'Sell'}), Price=${test_data[t]:.2f}")

    next_state = getState(test_data, t + 1, window_size + 1)

    if action == 1:  # Buy
        agent.inventory.append(test_data[t])
        buy_signals.append(t)

    elif action == 2 and len(agent.inventory) > 0:  # Sell
        bought_price = agent.inventory.pop(0)
        profit = test_data[t] - bought_price
        total_profit += profit
        sell_signals.append(t)

    state = next_state

print("--------------------------------")
print(f"Test Total Profit: {formatPrice(total_profit)}")
print(f"Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print(f"Buy signals: {len(buy_signals)}, Sell signals: {len(sell_signals)}")
print("--------------------------------")

# --- 6. Visualization ---

plt.figure(figsize=(12, 6))
plt.plot(test_data, color='gray', label='Price', alpha=0.5)
plt.plot(buy_signals, [test_data[i] for i in buy_signals], '^', markersize=8, color='green', label='Buy')
plt.plot(sell_signals, [test_data[i] for i in sell_signals], 'v', markersize=8, color='red', label='Sell')
plt.title(f'RL Trading Bot Evaluation - Profit: {formatPrice(total_profit)}')
plt.legend()
plt.tight_layout()
plt.show()
