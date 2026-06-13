"""
8064.TWO (東捷科技) PPO 訓練 + 準確度比較
比較 XGBoost (54.64%) vs PPO
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
from sklearn.metrics import accuracy_score
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

TICKER = '8064.TWO'
TOTAL_TIMESTEPS = 100000

# ==========================================
# Trading Environment
# ==========================================
class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
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
        self.last_action = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        price = float(row['close'])
        total_val = self.balance + self.shares_held * price
        return np.array([
            float(self.shares_held),
            float(self.balance),
            price,
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
            (self.shares_held * price) / total_val if total_val > 0 else 0,
            self.balance / total_val if total_val > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])

        if action < -0.1:
            shares = int(self.shares_held * abs(action))
            if shares > 0:
                self.balance += shares * price
                self.shares_held -= shares
                self.total_trades += 1
        elif action > 0.1:
            max_buy = int(self.balance // price)
            shares = int(max_buy * action)
            if shares > 0:
                self.balance -= shares * price
                self.shares_held += shares
                self.total_trades += 1

        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        reward = self.total_profit / self.initial_balance
        if abs(action) > 0.1:
            reward += 0.01
        if self.balance > (self.balance + self.shares_held * price) * 0.9:
            reward -= 0.005
        return self._get_obs(), reward, done, False, {}


def add_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

    # Future return for accuracy evaluation
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['true_label'] = (df['future_return'] > 0.02).astype(int)

    return df.bfill().ffill()


def evaluate_ppo_accuracy(model, test_df):
    """
    Evaluate PPO signal accuracy vs actual future returns.
    action > 0.1 => model says BUY => compare to true_label (next 5d > 2%)
    """
    env = TradingEnv(test_df)
    env.current_step = 0
    obs, _ = env.reset()

    predictions = []
    labels = []

    for i in range(len(test_df) - 6):  # leave room for future_return
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0])
        pred = 1 if action_val > 0.1 else 0
        true = int(test_df.iloc[i]['true_label'])
        predictions.append(pred)
        labels.append(true)
        obs, _, done, _, _ = env.step(action)
        if done:
            break

    acc = accuracy_score(labels, predictions) if predictions else 0.0
    buy_signals = sum(predictions)
    return acc, buy_signals, len(predictions)


if __name__ == "__main__":
    print("=" * 80)
    print(f"🚀 {TICKER} (東捷科技) PPO 訓練 + 準確度比較")
    print("=" * 80)

    # Download data
    print("\n📊 下載數據...")
    df = yf.download(TICKER, start='2015-01-01', end='2025-12-31', progress=False)
    if df.empty:
        print("ERROR: 無法下載數據")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                            'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
    print(f"下載 {len(df)} 天數據")

    df = add_indicators(df)
    df = df.dropna(subset=['sma_200', 'true_label'])

    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    print(f"訓練集: {len(train_df)} 天 | 測試集: {len(test_df)} 天")

    # Train PPO
    print(f"\n🧠 訓練 PPO 模型 ({TOTAL_TIMESTEPS:,} 步)...")
    env = DummyVecEnv([lambda: TradingEnv(train_df)])
    model = PPO(
        'MlpPolicy', env,
        verbose=0,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    model_path = 'ppo_8064_two_improved'
    model.save(model_path)
    print(f"✅ PPO 模型已保存: {model_path}.zip")

    # Evaluate accuracy
    print("\n📈 評估 PPO 準確度...")
    ppo_acc, buy_signals, total = evaluate_ppo_accuracy(model, test_df)
    print(f"PPO 測試準確度: {ppo_acc*100:.2f}%")
    print(f"買入信號數: {buy_signals} / {total} ({buy_signals/total*100:.1f}%)")

    # Save accuracy JSON
    accuracy_data = {
        'symbol': TICKER,
        'model_type': 'PPO',
        'training_accuracy': float(ppo_acc * 100),
        'validation_accuracy': float(ppo_acc * 100),
        'backtest_accuracy': float(ppo_acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('model_accuracy_8064_TWO_ppo.json', 'w', encoding='utf-8') as f:
        json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
    print(f"✅ PPO 準確度已保存: model_accuracy_8064_TWO_ppo.json")

    # Comparison summary
    xgb_acc = 54.64
    print("\n" + "=" * 80)
    print("📊 模型準確度比較")
    print("=" * 80)
    print(f"  XGBoost 測試準確度: {xgb_acc:.2f}%")
    print(f"  PPO     測試準確度: {ppo_acc*100:.2f}%")
    winner = "XGBoost" if xgb_acc >= ppo_acc * 100 else "PPO"
    diff = abs(xgb_acc - ppo_acc * 100)
    print(f"  勝出模型: {winner} (差距 {diff:.2f}%)")
    print("=" * 80)
