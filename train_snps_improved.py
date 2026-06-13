"""
SNPS (SNPS) Trading AI Training
==========================================
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
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
        self.last_action = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        return np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(stock_ratio), float(cash_ratio),
        ], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        action = np.clip(float(action), -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price
        if action < -0.1:
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            shares_to_buy = int((self.balance // current_price) * action)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        reward = (self.total_profit / self.initial_balance) + (0.01 if abs(action) > 0.1 else 0) + (-0.005 if self.balance > old_total_value * 0.9 else 0)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}

if __name__ == "__main__":
    import yfinance as yf
    import json
    from datetime import datetime

    TICKER = 'SNPS'
    START_DATE = '2015-01-01'  # SNPS IPO was May 2012 (as Facebook)
    END_DATE = '2026-01-01'
    TOTAL_TIMESTEPS = 100000

    print(f"Training {TICKER} (SNPS)")
    print(f"Data range: {START_DATE} - {END_DATE}")

    df = yf.download(TICKER, start=START_DATE, end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
    df = df.reset_index()
    print(f"Downloaded {len(df)} days of data")

    # Technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    true_range = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    df = df.bfill().ffill()

    # Feature importance
    features = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D', 'OBV', 'OBV_MA', 'MA_20', 'MA_50', 'MA_200', 'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope']
    ml_data = df.dropna(subset=features + ['future_direction'])
    X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(ml_data[features]), ml_data['future_direction'], test_size=0.2, shuffle=False)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, rf.predict(X_test))
    print(f"Feature importance accuracy: {accuracy:.4f}")
    with open(f'{TICKER}_feature_importance.json', 'w') as f:
        json.dump({'ticker': TICKER, 'analysis_date': datetime.now().strftime('%Y-%m-%d'), 'model_accuracy': float(accuracy), 'feature_importance': dict(zip(features, rf.feature_importances_.tolist()))}, f, indent=2)

    # Train PPO
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    print(f"Training on {len(train_df)} days")
    env = DummyVecEnv([lambda: ImprovedTradingEnv(train_df)])
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(f"ppo_{TICKER.lower()}_improved")
    print(f"\n✅ Training complete! Model: ppo_{TICKER.lower()}_improved.zip")
