"""
批量训练第7批股票 PPO 模型
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

STOCKS_TO_TRAIN = [
    '3017.TW',   # 奇鋐
    '2368.TW',   # 金像電
    '6669.TW',   # 緯穎
    '2317.TW',   # 鴻海
    '2383.TW',   # 台光電
    '6274.TW',   # 台燿
    '6285.TW',   # 啟碁
    '6282.TW',   # 康舒
    '1795.TW',   # 美時
    '6446.TW',   # 藥華藥
    '3443.TW',   # 創意
]

class ImprovedTradingEnv(gym.Env):
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
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = float(row['close'])
        total = self.balance + self.shares_held * price
        return np.array([
            self.shares_held, self.balance, price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            self.total_profit, (self.shares_held * price) / total if total > 0 else 0,
            self.balance / total if total > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])
        old_val = self.balance + self.shares_held * price

        if action < -0.1 and self.shares_held > 0:
            sell = int(self.shares_held * abs(action))
            self.balance += sell * price
            self.shares_held -= sell
        elif action > 0.1 and self.balance > price:
            buy = int((self.balance // price) * action)
            self.balance -= buy * price
            self.shares_held += buy

        new_val = self.balance + self.shares_held * price
        self.total_profit = new_val - self.initial_balance
        reward = self.total_profit / self.initial_balance
        self.current_step += 1
        return self._get_observation(), reward, self.current_step >= len(self.df) - 1, False, {}

def download_data(ticker):
    print(f"\n下载 {ticker} ...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start='2015-01-01', end='2025-07-31', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            raise ValueError("Empty")
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"  ✅ {len(df)} 天")
        return df
    except Exception as e:
        print(f"  ❌ {e}")
        return None

def add_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / ((-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_mid + bb_std * 2
    df['bb_lower'] = bb_mid - bb_std * 2
    return df.fillna(method='bfill').fillna(method='ffill')

def train_model(df, ticker):
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=100000)
    path = f"ppo_{ticker.lower().replace('.', '_')}_improved"
    model.save(path)
    print(f"  ✅ {path}.zip")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 批量训练第7批 (11 stocks)")
    print("=" * 50)

    success, failed = 0, []
    for i, ticker in enumerate(STOCKS_TO_TRAIN, 1):
        print(f"[{i}/{len(STOCKS_TO_TRAIN)}] {ticker}")
        df = download_data(ticker)
        if df is None:
            failed.append(ticker)
            continue
        df = add_indicators(df)
        train_model(df.iloc[:int(len(df) * 0.8)].copy(), ticker)
        success += 1

    print(f"\n✅ {success}/{len(STOCKS_TO_TRAIN)}")
    if failed:
        print(f"❌ {', '.join(failed)}")
