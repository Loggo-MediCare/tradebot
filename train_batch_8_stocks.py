"""
批量训练第8批股票 PPO 模型
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
import warnings
warnings.filterwarnings('ignore')

STOCKS_TO_TRAIN = [
    '6285.TW',   # 啟碁
    '2489.TW',   # 瑞軒
    '3149.TW',   # 正達
    '8046.TW',   # 南電
    '7795.TW',   # 瀚昕科技
    '3049.TW',   # 和鑫
    '6435.TW',   # 大中
    '4563.TW',   # 百德
    '1514.TW',   # 亞力
    '1471.TW',   # 首利
    '3037.TW',   # 欣興
    '6548.TW',   # 長佳智能
    '6510.TW',   # 精測
    '2313.TW',   # 華通
    '2367.TW',   # 燿華
]

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0
        self.balance = 10000
        self.shares = 0
        self.profit = 0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.step_idx]
        p = float(r['close'])
        t = self.balance + self.shares * p
        return np.array([self.shares, self.balance, p,
            float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),
            float(r.get('rsi',50)), float(r.get('macd',0)), float(r.get('macd_signal',0)),
            float(r.get('bb_upper',0)), float(r.get('bb_lower',0)), float(r.get('volume',0)),
            self.profit, (self.shares*p)/t if t>0 else 0, self.balance/t if t>0 else 1], dtype=np.float32)

    def step(self, action):
        a = np.clip(float(action[0]) if isinstance(action, np.ndarray) else float(action), -1, 1)
        p = float(self.df.iloc[self.step_idx]['close'])
        if a < -0.1 and self.shares > 0:
            s = int(self.shares * abs(a))
            self.balance += s * p
            self.shares -= s
        elif a > 0.1 and self.balance > p:
            b = int((self.balance // p) * a)
            self.balance -= b * p
            self.shares += b
        self.profit = self.balance + self.shares * p - 10000
        self.step_idx += 1
        return self._obs(), self.profit/10000, self.step_idx >= len(self.df)-1, False, {}

def download(ticker):
    print(f"\n下载 {ticker} ...")
    try:
        import yfinance as yf
        df = yf.download(ticker, start='2015-01-01', end='2025-07-31', progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty: raise ValueError("Empty")
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  ✅ {len(df)} 天")
        return df
    except Exception as e:
        print(f"  ❌ {e}")
        return None

def indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    df['rsi'] = 100 - (100/(1+(d.where(d>0,0)).rolling(14).mean()/((-d.where(d<0,0)).rolling(14).mean()+1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    m = df['close'].rolling(20).mean()
    s = df['close'].rolling(20).std()
    df['bb_upper'] = m + s*2
    df['bb_lower'] = m - s*2
    return df.fillna(method='bfill').fillna(method='ffill')

def train(df, ticker):
    env = DummyVecEnv([lambda: TradingEnv(df)])
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=100000)
    path = f"ppo_{ticker.lower().replace('.','_')}_improved"
    model.save(path)
    print(f"  ✅ {path}.zip")

if __name__ == "__main__":
    print("=" * 50)
    print("🚀 批量训练第8批 (15 stocks)")
    print("=" * 50)
    ok, fail = 0, []
    for i, t in enumerate(STOCKS_TO_TRAIN, 1):
        print(f"[{i}/{len(STOCKS_TO_TRAIN)}] {t}")
        df = download(t)
        if df is None: fail.append(t); continue
        df = indicators(df)
        train(df.iloc[:int(len(df)*0.8)].copy(), t)
        ok += 1
    print(f"\n✅ {ok}/{len(STOCKS_TO_TRAIN)}")
    if fail: print(f"❌ {', '.join(fail)}")
