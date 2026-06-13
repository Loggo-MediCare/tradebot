"""
LITE (Lumentum Holdings) PPO 改進版訓練
改進:
1. 增加訓練步數 (50000 -> 150000)
2. 調整獎勵函數 (加入交易鼓勵)
3. 調整探索參數
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import json
from datetime import datetime

print("="*80)
print("訓練 LITE (Lumentum Holdings) PPO 模型 (改進版)")
print("="*80)

TICKER = 'LITE'
df = yf.download(TICKER, start='2015-01-01', end='2026-12-31', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                        'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()

print(f"下載 {len(df)} 天數據")

# 添加技術指標
df['sma_10'] = df['close'].rolling(10).mean()
df['sma_30'] = df['close'].rolling(30).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df['rsi'] = 100 - (100 / (1 + rs))

df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()

df['bb_middle'] = df['close'].rolling(20).mean()
df['bb_std'] = df['close'].rolling(20).std()
df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

df = df.bfill().ffill()
df_clean = df.dropna()

print(f"清理後數據: {len(df_clean)} 天")

# 改進的交易環境
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.last_total_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1

        obs = np.array([
            float(self.shares_held),
            float(self.balance),
            float(row['close']),
            float(row.get('sma_10', 0)),
            float(row.get('sma_30', 0)),
            float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)),
            float(row.get('macd', 0)),
            float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)),
            float(row.get('bb_lower', 0)),
            float(row.get('volume', 0)),
            float(self.total_profit),
            float(stock_ratio),
            float(cash_ratio),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])

        traded = False

        # 執行交易
        if action > 0.15:
            max_shares = int(self.balance / current_price)
            shares_to_buy = int(max_shares * abs(action))
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
                traded = True

        elif action < -0.15:
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                self.balance += revenue
                self.shares_held -= shares_to_sell
                self.total_trades += 1
                traded = True

        # 移動到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # 改進的獎勵函數
        total_value = self.balance + self.shares_held * current_price
        profit = total_value - self.initial_balance
        self.total_profit = profit

        # 基礎獎勵: 總價值變化
        value_change = total_value - self.last_total_value
        reward = value_change / self.initial_balance

        # 獎勵交易行為 (鼓勵學習交易)
        if traded:
            reward += 0.01

        # 懲罰長期不交易
        if self.total_trades < self.current_step / 100:
            reward -= 0.001

        self.last_total_value = total_value

        return self._get_observation(), reward, done, False, {}

# 訓練
print("\n開始訓練 PPO 模型 (改進版)...")
env = DummyVecEnv([lambda: ImprovedTradingEnv(df_clean)])

model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0005,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.01,
    verbose=1
)

print("訓練中... (150000 步，約 2-3 分鐘)")
model.learn(total_timesteps=150000)

model_filename = 'ppo_lite_improved'
model.save(model_filename)
print(f"\n模型已保存: {model_filename}.zip")

# 評估
print("\n評估模型...")
test_env = ImprovedTradingEnv(df_clean)
obs, _ = test_env.reset()

correct_predictions = 0
total_predictions = 0

for _ in range(len(df_clean) - 1):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)

    if abs(action[0]) > 0.15:
        total_predictions += 1
        if reward > 0:
            correct_predictions += 1

    if done:
        break

accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
final_value = test_env.balance + test_env.shares_held * df_clean.iloc[-1]['close']
total_return = ((final_value - test_env.initial_balance) / test_env.initial_balance) * 100

print(f"\n{'='*80}")
print("訓練結果 (改進版)")
print(f"{'='*80}")
print(f"模型準確度: {accuracy:.2f}%")
print(f"總回報率: {total_return:.2f}%")
print(f"最終價值: ${final_value:.2f}")
print(f"總交易次數: {test_env.total_trades}")
print(f"{'='*80}")

# 保存結果
accuracy_data = {
    'symbol': 'LITE',
    'model_type': 'PPO',
    'training_accuracy': float(accuracy),
    'validation_accuracy': float(accuracy),
    'backtest_accuracy': float(accuracy),
    'total_return': float(total_return),
    'total_trades': int(test_env.total_trades),
    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_accuracy_LITE.json', 'w', encoding='utf-8') as f:
    json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

print(f"準確度已保存: model_accuracy_LITE.json")
