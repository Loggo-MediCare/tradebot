# -*- coding: utf-8 -*-
"""
台灣股票 2344.TW 強化學習交易策略
基於改進版防過擬合模型
Taiwan Stock 2344.TW RL Trading Strategy
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
from roi_control import print_roi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("🚀 台灣股票 2344.TW 強化學習交易策略")

# =========================================================
# 1. 數據下載
# =========================================================
def download_data(ticker, days=365):
    """下載股票數據"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"📥 下載 {ticker} 數據：{start_date.date()} 到 {end_date.date()}...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        df = df.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"})

        if df.empty:
            raise ValueError("下載的數據為空")

        return df, df['Close'].values

    except Exception as e:
        print(f"❌ 數據下載失敗: {e}")
        return None, None

# 台灣股票代碼（需要加 .TW 後綴）
ticker = '2344.TW'
df, prices = download_data(ticker, days=365)

if df is None:
    print("⚠️  無法下載數據，請檢查：")
    print("   1. 網路連線是否正常")
    print("   2. 股票代碼是否正確（台股需要加 .TW 後綴）")
    print("   3. 該股票是否存在於 Yahoo Finance")
    sys.exit(1)

print(f"✓ 數據已下載：{len(df)} 個交易日")

# =========================================================
# 2. 技術指標特徵工程
# =========================================================
def create_features(df):
    """創建技術指標"""
    features = df.copy()

    # 移動平均線
    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()
    features['SMA_50'] = features['Close'].rolling(window=50).mean()

    # 動量指標
    features['ROC_5'] = features['Close'].pct_change(periods=5) * 100
    features['ROC_10'] = features['Close'].pct_change(periods=10) * 100

    # RSI
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))

    # 波動率
    features['Volatility_10'] = features['Close'].pct_change().rolling(window=10).std()
    features['Volatility_20'] = features['Close'].pct_change().rolling(window=20).std()

    # MACD
    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()

    # 布林通道
    sma_20 = features['Close'].rolling(window=20).mean()
    std_20 = features['Close'].rolling(window=20).std()
    features['BB_upper'] = sma_20 + (std_20 * 2)
    features['BB_lower'] = sma_20 - (std_20 * 2)
    features['BB_position'] = (features['Close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])

    # 成交量指標
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
print(f"✓ 特徵已創建：{len(feature_cols)} 個特徵，{len(df_features)} 個有效數據點")

# =========================================================
# 3. 交易環境
# =========================================================
class TradingEnvironment:
    def __init__(self, df, feature_cols, initial_balance=100000, window_size=5, transaction_cost=0.001425):
        """
        initial_balance: 初始資金（台股通常至少10萬）
        transaction_cost: 交易成本（台股約 0.1425% = 手續費0.1425% + 證交稅0.3%）
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.transaction_cost = transaction_cost  # 台股交易成本
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

        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 1000,  # 台股通常以張（1000股）為單位
            portfolio_value / self.initial_balance,
            (portfolio_value - self.initial_balance) / self.initial_balance,
            len(self.trades_history) / 100,
        ])

        return np.concatenate((market_state, account_state))

    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.total_shares * current_price)

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

        # Reward function
        portfolio_change = new_portfolio_value - prev_portfolio_value
        reward = (portfolio_change / self.initial_balance) * 100
        reward -= (cost_penalty / self.initial_balance) * 100

        if action in [1, 2]:
            if portfolio_change > 0:
                reward += 0.5
            else:
                reward -= 0.5

        if done:
            self.balance += self.total_shares * next_price
            self.total_shares = 0
            final_profit_pct = (self.balance - self.initial_balance) / self.initial_balance

            if final_profit_pct > 0:
                reward += final_profit_pct * 20
            else:
                reward += final_profit_pct * 10

        return self._get_state(), reward, done, new_portfolio_value

    def get_final_profit(self):
        return self.balance - self.initial_balance

# =========================================================
# 4. DQN Agent with Regularization
# =========================================================
class ImprovedDQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=3000)

        self.gamma = 0.97
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.996
        self.learning_rate = 0.0005

        self.model = self._build_model()

    def _build_model(self):
        """Neural network with dropout and L2 regularization"""
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
env = TradingEnvironment(df_features, feature_cols, initial_balance=100000)
state_size = len(env.reset())
agent = ImprovedDQNAgent(state_size)

EPISODES = 20
BATCH_SIZE = 64

print(f"\n🎯 開始訓練 {EPISODES} 個回合...")

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
            print(f"回合 {e+1:2d}/{EPISODES} | 利潤: ${episode_profit:>10,.2f} | 交易: {len(env.trades_history):3d} | ε: {agent.epsilon:.3f}")

            if episode_profit > best_profit:
                best_profit = episode_profit
                best_episode = e + 1
                best_model_weights = agent.model.get_weights()
                print(f"             ⭐ 新的最佳模型！")
            break

    agent.replay(BATCH_SIZE)

print(f"\n✓ 訓練完成 - 最佳回合：第 {best_episode} 回合，利潤 ${best_profit:,.2f}")
agent.model.set_weights(best_model_weights)

# Save model
model_filename = f'{ticker.replace(".", "_")}_model.keras'
agent.model.save(model_filename)
print(f"💾 模型已保存：{model_filename}")

# =========================================================
# 6. Backtesting
# =========================================================
print("\n📊 回測中（epsilon=0）...")
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
roi = (final_profit / 100000) * 100

print("\n" + "="*60)
print(f"台灣股票 {ticker} 回測結果")
print("="*60)
print(f"初始資金:    ${100000:,.2f}")
print(f"最終資金:    ${env.balance:,.2f}")
print(f"總利潤:      ${final_profit:,.2f}")
print_roi(f"投資回報率:  {roi:.2f}%")
print(f"執行交易:    {len(env.trades_history)} 筆")
print(f"動作分布:    Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print("="*60)

if final_profit > 0:
    print(f"✅ 成功！獲利 ${final_profit:,.2f}")
else:
    print(f"⚠️  虧損 ${abs(final_profit):,.2f}")

# =========================================================
# 7. Visualization
# =========================================================
plt.figure(figsize=(14, 10))

# Portfolio Value
plt.subplot(3, 1, 1)
plt.plot(env.portfolio_value_history, label='投資組合價值', color='blue', linewidth=2)
plt.axhline(y=100000, color='red', linestyle='--', alpha=0.5, label='初始資金')
plt.title(f'{ticker} - 強化學習策略表現', fontsize=14, fontweight='bold')
plt.ylabel('價值 (TWD)')
plt.legend()
plt.grid(True, alpha=0.3)

# Price and Trades
plt.subplot(3, 1, 2)
price_data = env.df['Close'].iloc[env.window_size:].reset_index(drop=True)
plt.plot(price_data, label='股價', color='black', alpha=0.6, linewidth=1.5)

for trade in env.trades_history:
    action, step, price, shares = trade
    idx = step - env.window_size
    if 0 <= idx < len(price_data):
        if action == 'BUY':
            plt.scatter(idx, price, color='green', marker='^', s=120, zorder=5)
        elif action == 'SELL':
            plt.scatter(idx, price, color='red', marker='v', s=120, zorder=5)

plt.title('交易信號', fontsize=12)
plt.ylabel('股價 (TWD)')
plt.legend(['股價', '買入', '賣出'])
plt.grid(True, alpha=0.3)

# Training Progress
plt.subplot(3, 1, 3)
plt.plot(training_profits, marker='o', linewidth=2, markersize=6, color='purple')
plt.axhline(y=best_profit, color='red', linestyle='--', alpha=0.5, label=f'最佳: ${best_profit:,.0f}')
plt.title('訓練進度 - 每回合利潤', fontsize=12)
plt.xlabel('回合')
plt.ylabel('利潤 (TWD)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
chart_filename = f'{ticker.replace(".", "_")}_results.png'
plt.savefig(chart_filename, dpi=150, bbox_inches='tight')
print(f"\n📊 圖表已保存：{chart_filename}")

print("\n✅ 完成！")
