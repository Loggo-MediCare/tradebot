"""
台股 2417 A2C 交易 AI 训练
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
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

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
        self.last_action = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        obs = np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(stock_ratio), float(cash_ratio),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)
        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price
        if action < -0.1:
            shares_to_sell = int(self.shares_held * abs(action))
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * action)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * current_price
                self.shares_held += shares_to_buy
                self.total_trades += 1
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        profit_reward = self.total_profit / self.initial_balance
        trade_incentive = 0.01 if abs(action) > 0.1 else 0.0
        cash_penalty = -0.005 if self.balance > old_total_value * 0.9 else 0.0
        reward = profit_reward + trade_incentive + cash_penalty
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_observation(), float(reward), done, False, {}

def download_and_prepare_data(ticker, start_date='2015-01-01', end_date='2025-07-31'):
    print(f"下载 {ticker} 股票数据...")
    import yfinance as yf
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        raise ValueError(f"无法下载 {ticker} 的数据")
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'})
    df = df.reset_index()
    print(f"成功下载 {len(df)} 天的数据")
    return df

def add_technical_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

if __name__ == "__main__":
    TICKER = '2417.TW'
    print(f"🚀 A2C 訓練: {TICKER}")
    df = download_and_prepare_data(TICKER)
    df = add_technical_indicators(df)
    train_df = df.iloc[:int(len(df) * 0.8)].copy()

    env = DummyVecEnv([lambda: ImprovedTradingEnv(train_df)])
    model = A2C(
        'MlpPolicy', env, verbose=1,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        ent_coef=0.01,
    )
    print("開始訓練 A2C (100,000 步)...")
    model.learn(total_timesteps=100000)
    model_path = f"a2c_2417_tw_improved"
    model.save(model_path)
    print(f"✅ A2C 模型已保存: {model_path}.zip")
