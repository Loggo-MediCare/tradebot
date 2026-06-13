# -*- coding: utf-8 -*-
"""
Reinforcement Learning Trading Strategy using Deep Q-Learning
Goal: Achieve profit of 1280.40+ using intelligent trading signals
Based on Chapter 9: Reinforcement Learning for Trading
"""

# !pip install yfinance pandas numpy matplotlib tensorflow scikit-learn -q

import sys
import io
# Fix Windows console encoding for emojis and Chinese characters
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# === 設定 ===
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🚀 啟動強化學習交易策略 (Deep Q-Learning)...")

# =========================================================
# 1. 下載並準備數據
# =========================================================
def download_data(ticker, days=365):
    """下載股票數據"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            df.columns = df.columns.get_level_values(0)

        # Get required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df_clean = df[required_cols].copy()

        # Get prices for returns calculation
        prices = df['Close'].values

        return df_clean, prices
    except Exception as e:
        print(f"數據下載失敗: {e}")
        import traceback
        traceback.print_exc()
        return None, None

ticker = 'NVDA'  # 台灣的例子
df, prices = download_data(ticker, days=365)

# Check if data was downloaded successfully
if df is None or len(prices) == 0:
    raise ValueError(f"Failed to download data for {ticker}. Please check your internet connection and ticker symbol.")

print(f"✓ 數據已下載，共 {len(prices)} 筆交易日數據")

# =========================================================
# 2. 特徵工程 (Feature Engineering)
# =========================================================
def create_features(df, window_size=5):
    """創建技術指標特徵"""
    features = df.copy()
    
    # 簡單移動平均 (SMA)
    features['SMA_5'] = df['Close'].rolling(5).mean()
    features['SMA_10'] = df['Close'].rolling(10).mean()
    features['SMA_20'] = df['Close'].rolling(20).mean()
    
    # 相對強度指數 (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))
    
    # 波動率 (Volatility)
    features['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # 價格變化率 (ROC)
    features['ROC'] = df['Close'].pct_change(periods=5) * 100
    
    # MACD
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9).mean()
    
    # 標準化特徵
    features = features.dropna()

    # Validate we have enough data after dropping NaN values
    if len(features) == 0:
        raise ValueError("Not enough data to create features. All rows contain NaN values after feature calculation.")

    scaler = MinMaxScaler()
    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'Volatility', 'ROC', 'MACD', 'Signal_Line']

    # Ensure we have data before fitting the scaler
    if len(features[feature_cols]) > 0:
        features[feature_cols] = scaler.fit_transform(features[feature_cols])
    else:
        raise ValueError("No valid feature data available for scaling.")

    return features, feature_cols

df_features, feature_cols = create_features(df)
print(f"✓ 特徵工程完成，共 {len(feature_cols)} 個特徵")
print(f"✓ 特徵數據量: {len(df_features)} 筆 (原始數據: {len(df)} 筆)")

# =========================================================
# 3. 強化學習環境 (RL Trading Environment)
# =========================================================
class TradingEnvironment:
    def __init__(self, df, initial_balance=10000, window_size=5):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.trades_history = []
        self.portfolio_value_history = []
        
    def reset(self):
        """重置環境"""
        self.balance = self.initial_balance
        self.total_shares = 0
        self.current_step = self.window_size
        self.trades_history = []
        self.portfolio_value_history = []
        return self._get_state()
    
    def _get_state(self):
        """獲取當前狀態 (最近5天的特徵)"""
        start = max(0, self.current_step - self.window_size)
        state = self.df[feature_cols].iloc[start:self.current_step].values.flatten()
        # 加入帳戶信息
        price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + self.total_shares * price
        state = np.append(state, [self.balance, self.total_shares, price, portfolio_value])
        return state
    
    def step(self, action):
        """執行動作 (0=Hold, 1=Buy, 2=Sell)"""
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + self.total_shares * current_price
        reward = 0
        done = False

        if action == 1:  # 買入
            # 使用部分資金買入（買入多股）
            shares_to_buy = int(self.balance * 0.3 / current_price)  # 使用30%資金
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.total_shares += shares_to_buy
                self.balance -= cost
                self.trades_history.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # 賣出
            # 賣出持有股票（賣出30%）
            shares_to_sell = max(1, int(self.total_shares * 0.3))
            if self.total_shares > 0:
                shares_to_sell = min(shares_to_sell, self.total_shares)
                revenue = shares_to_sell * current_price
                self.total_shares -= shares_to_sell
                self.balance += revenue
                self.trades_history.append(('SELL', self.current_step, current_price, shares_to_sell))

        # 進行下一步
        self.current_step += 1

        # 計算新的投資組合價值
        new_price = float(self.df['Close'].iloc[self.current_step]) if self.current_step < len(self.df) else current_price
        portfolio_value = self.balance + self.total_shares * new_price
        self.portfolio_value_history.append(portfolio_value)

        # 計算基於投資組合價值變化的獎勵
        reward = (portfolio_value - prev_portfolio_value) / self.initial_balance * 100

        # 檢查是否結束
        if self.current_step >= len(self.df) - 1:
            done = True
            # 結束時賣出所有股票
            final_price = float(self.df['Close'].iloc[self.current_step])
            self.balance += self.total_shares * final_price
            self.total_shares = 0
            # 給予最終獎勵
            final_profit = self.balance - self.initial_balance
            reward += final_profit / self.initial_balance * 10  # 額外的最終獎勵

        return self._get_state(), reward, done, portfolio_value
    
    def get_final_profit(self):
        """獲取最終利潤"""
        return self.balance - self.initial_balance

# =========================================================
# 4. 深度Q學習代理 (Deep Q-Learning Agent)
# =========================================================
class DQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        
        # 超參數
        self.gamma = 0.98  # 折扣因子（提高以重視長期回報）
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.05  # 保留更多探索
        self.epsilon_decay = 0.996  # 更慢的衰減
        self.learning_rate = 0.0005  # 降低學習率以更穩定
        
        # 神經網絡
        self.model = self._build_model()
    
    def _build_model(self):
        """構建神經網絡模型"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """存儲經驗"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """選擇動作 (ε-貪心策略)"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, self.action_size)  # 探索
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])  # 利用
    
    def replay(self, batch_size):
        """經驗回放"""
        if len(self.memory) < batch_size:
            return

        # 從記憶中隨機抽取批次
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] for exp in batch])
        dones = np.array([exp[4] for exp in batch])
        
        # 預測Q值
        target_qs = self.model.predict(states, verbose=0)
        next_qs = self.model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_qs[i][actions[i]] = rewards[i]
            else:
                target_qs[i][actions[i]] = rewards[i] + self.gamma * np.max(next_qs[i])
        
        self.model.fit(states, target_qs, epochs=1, verbose=0)
        
        # 衰減探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =========================================================
# 5. 訓練代理
# =========================================================
print("\n3️⃣ 正在訓練深度Q學習代理...")

env = TradingEnvironment(df_features, initial_balance=10000)
state_size = len(feature_cols) * 5 + 4  # 特徵 + 帳戶信息
agent = DQNAgent(state_size)

episodes = 100  # 增加訓練次數
batch_size = 32

best_profit = -float('inf')

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(len(env.df) - env.window_size - 1):
        action = agent.act(state)
        next_state, reward, done, portfolio_value = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

        if done:
            break

    agent.replay(batch_size)

    profit = env.get_final_profit()
    if profit > best_profit:
        best_profit = profit

    if (episode + 1) % 10 == 0:
        print(f"  Episode {episode+1}/{episodes} | Profit: ${profit:.2f} | Best: ${best_profit:.2f} | Epsilon: {agent.epsilon:.3f}")

print("✓ 訓練完成")

# =========================================================
# 6. 回測與評估
# =========================================================
print("\n4️⃣ 執行回測...")

env = TradingEnvironment(df_features, initial_balance=10000)
state = env.reset()
agent.epsilon = 0  # 關閉探索，純利用

action_counts = {0: 0, 1: 0, 2: 0}  # Count actions: Hold, Buy, Sell

for step in range(len(env.df) - env.window_size - 1):
    action = agent.act(state)
    action_counts[action] += 1
    next_state, reward, done, portfolio_value = env.step(action)
    state = next_state
    if done:
        break

final_profit = env.get_final_profit()
final_portfolio_value = env.balance + env.total_shares * float(env.df['Close'].iloc[-1])

print(f"\n📊 回測結果:")
print(f"  動作統計: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print(f"  初始資本: $10,000.00")
print(f"  最終資產: ${final_portfolio_value:.2f}")
print(f"  淨利潤: ${final_profit:.2f}")
print(f"  回報率: {(final_profit/10000)*100:.2f}%")
print(f"  交易次數: {len(env.trades_history)}")

if final_profit >= 1280.40:
    print(f"\n🎉 成功達成目標利潤 1280.40 元！")
else:
    print(f"\n⚠️ 當前利潤 ${final_profit:.2f}，距離目標還需 ${1280.40 - final_profit:.2f}")

# =========================================================
# 7. 可視化結果
# =========================================================
plt.figure(figsize=(15, 10))

# 投資組合價值
plt.subplot(2, 1, 1)
plt.plot(env.portfolio_value_history, label='RL策略投資組合', color='green', linewidth=2)
plt.axhline(y=10000, color='gray', linestyle='--', label='初始資本', alpha=0.5)
plt.title(f'{ticker} - 強化學習交易策略 (最終利潤: ${final_profit:.2f})', fontsize=14, fontweight='bold')
plt.ylabel('投資組合價值 ($)')
plt.legend()
plt.grid(True, alpha=0.3)

# 交易信號
plt.subplot(2, 1, 2)
prices_plot = df_features['Close'].iloc[env.window_size:env.current_step].values
plt.plot(prices_plot, label='股價', color='blue', linewidth=1.5)

# 標記買入和賣出
buys = [t for t in env.trades_history if t[0] == 'BUY']
sells = [t for t in env.trades_history if t[0] == 'SELL']

for buy in buys:
    idx = buy[1] - env.window_size
    if 0 <= idx < len(prices_plot):
        plt.scatter(idx, prices_plot[idx], color='green', marker='^', s=100, label='買入' if buy == buys[0] else '')

for sell in sells:
    idx = sell[1] - env.window_size
    if 0 <= idx < len(prices_plot):
        plt.scatter(idx, prices_plot[idx], color='red', marker='v', s=100, label='賣出' if sell == sells[0] else '')

plt.title('交易信號 (綠色^=買入, 紅色v=賣出)', fontsize=12)
plt.xlabel('交易日')
plt.ylabel('股價 ($)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('rl_trading_results.png', dpi=150, bbox_inches='tight')
print(f"\n📊 圖表已保存到: rl_trading_results.png")

print("\n✅ 策略回測完成！")
