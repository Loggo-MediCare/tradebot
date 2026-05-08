"""
台股 5289 (宜鼎 Innodisk) 交易 AI 训练 - Industrial SSD/Memory Modules
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
        return np.array([float(self.shares_held), float(self.balance), price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), (self.shares_held * price) / total if total > 0 else 0,
            self.balance / total if total > 0 else 1], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])
        if action < -0.1 and self.shares_held > 0:
            sell = int(self.shares_held * abs(action))
            if sell > 0: self.balance += sell * price; self.shares_held -= sell
        elif action > 0.1:
            buy = int((self.balance // price) * action)
            if buy > 0: self.balance -= buy * price; self.shares_held += buy
        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance
        reward = self.total_profit / self.initial_balance + (0.01 if abs(action) > 0.1 else 0)
        self.current_step += 1
        return self._get_observation(), float(reward), self.current_step >= len(self.df) - 1, False, {}

def download_data(ticker):
    import yfinance as yf
    print(f"下载 {ticker}...")
    df = yf.download(ticker, start='2015-01-01', end='2025-07-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
    print(f"下载 {len(df)} 天数据")
    return df

def add_indicators(df):
    df['sma_10'], df['sma_30'], df['sma_50'] = df['close'].rolling(10).mean(), df['close'].rolling(30).mean(), df['close'].rolling(50).mean()
    df['ema_12'], df['ema_26'] = df['close'].ewm(span=12).mean(), df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    df['rsi'] = 100 - (100 / (1 + (delta.where(delta > 0, 0)).rolling(14).mean() / ((-delta.where(delta < 0, 0)).rolling(14).mean() + 1e-10)))
    df['macd'], df['macd_signal'] = df['ema_12'] - df['ema_26'], (df['ema_12'] - df['ema_26']).ewm(span=9).mean()
    df['bb_middle'], df['bb_std'] = df['close'].rolling(20).mean(), df['close'].rolling(20).std()
    df['bb_upper'], df['bb_lower'] = df['bb_middle'] + 2*df['bb_std'], df['bb_middle'] - 2*df['bb_std']
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    return df.fillna(method='bfill').fillna(method='ffill')

def analyze_features(df, ticker):
    import json
    features = ['rsi', 'macd', 'macd_signal', 'sma_10', 'sma_30', 'sma_50']
    ml_data = df.dropna(subset=features + ['future_direction'])
    X, y = ml_data[features], ml_data['future_direction']
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"預測準確率: {acc:.4f}")
    with open(f'{ticker}_feature_importance.json', 'w') as f:
        json.dump({'ticker': ticker, 'accuracy': acc}, f, indent=2)

if __name__ == "__main__":
    TICKER = '5289.TW'
    print(f"训练 {TICKER} (宜鼎 Innodisk - Industrial Memory)")
    df = download_data(TICKER)
    df = add_indicators(df)
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df.iloc[:int(len(df)*0.8)].copy())])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64)
    model.learn(total_timesteps=100000)
    model.save(f"ppo_{TICKER.lower().replace('.', '_')}_improved")
    analyze_features(df, TICKER)
    print(f"完成!")
