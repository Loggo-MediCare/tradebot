"""Train XGBoost + PPO for 2356 英業達"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

CODE    = '2356'
TICKER  = '2356.TW'
NAME    = '英業達'
SUFFIX  = 'tw'
END     = '2026-05-30'

# ── Features ─────────────────────────────────────────────────────────────────
def add_features(df):
    df = df.copy()
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_mid']   = df['close'].rolling(20).mean()
    df['bb_std']   = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['future_return'] = df['close'].pct_change(5).shift(-5)
    df['label'] = (df['future_return'] > 0).astype(int)
    df = df.bfill().ffill().dropna()
    return df

FEATURES = ['sma_10','sma_30','sma_50','rsi','macd','macd_signal',
            'bb_upper','bb_lower','obv']

# ── Env ───────────────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df, init_balance=100000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.init_balance = init_balance
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx   = 0
        self.balance    = self.init_balance
        self.shares     = 0
        self.total_pnl  = 0
        return self._obs(), {}

    def _obs(self):
        row = self.df.iloc[self.step_idx]
        price = float(row.get('close', 1))
        equity = self.balance + self.shares * price
        return np.array([
            self.shares, self.balance,
            price,
            float(row.get('sma_10', price)), float(row.get('sma_30', price)),
            float(row.get('sma_50', price)), float(row.get('rsi', 50)),
            float(row.get('macd', 0)),       float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', price)), float(row.get('bb_lower', price)),
            float(row.get('volume', 0)),
            self.total_pnl, equity / self.init_balance, self.balance / self.init_balance,
        ], dtype=np.float32)

    def step(self, action):
        row   = self.df.iloc[self.step_idx]
        price = float(row.get('close', 1))
        act   = float(action[0])
        if act > 0.3 and self.balance >= price:
            buy = int(self.balance // price)
            self.shares  += buy
            self.balance -= buy * price
        elif act < -0.3 and self.shares > 0:
            self.balance += self.shares * price
            self.shares   = 0
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        pnl  = self.balance + self.shares * price - self.init_balance
        self.total_pnl = pnl
        return self._obs(), pnl / self.init_balance * 0.01, done, False, {}


# ── Download ──────────────────────────────────────────────────────────────────
print(f'\n{"="*60}')
print(f'Training {CODE} {NAME}')
print(f'{"="*60}')

df_raw = yf.download(TICKER, start='2018-01-01', end=END, progress=False, auto_adjust=True)
print(f'Downloaded {len(df_raw)} rows')
df = add_features(df_raw)

# ── XGBoost ───────────────────────────────────────────────────────────────────
print('\n[XGBoost]')
X = df[FEATURES].values
y = df['label'].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_tr, y_tr)
acc = accuracy_score(y_te, xgb.predict(X_te))
print(f'  Accuracy: {acc*100:.2f}%')
joblib.dump(xgb, f'xgb_{CODE}_{SUFFIX}_model.pkl')
print(f'  Saved: xgb_{CODE}_{SUFFIX}_model.pkl')

# ── PPO ───────────────────────────────────────────────────────────────────────
print('\n[PPO]')
split = int(len(df) * 0.8)
df_train = df.iloc[:split].reset_index(drop=True)
env = TradingEnv(df_train)
model = PPO('MlpPolicy', env, verbose=0,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, ent_coef=0.01)
model.learn(total_timesteps=50000)
model.save(f'ppo_{CODE}_{SUFFIX}_improved')
print(f'  Saved: ppo_{CODE}_{SUFFIX}_improved')

# Quick backtest
df_test = df.iloc[split:].reset_index(drop=True)
env_t = TradingEnv(df_test)
obs, _ = env_t.reset()
env_t.step_idx = 0
done = False
while not done:
    act, _ = model.predict(obs, deterministic=True)
    obs, _, done, _, _ = env_t.step(act)
final = env_t.balance + env_t.shares * float(df_test.iloc[-1]['close'])
roi = (final - 100000) / 100000 * 100
print(f'  PPO Test ROI: {roi:+.2f}%')

print(f'\nDone! Models saved for {CODE} {NAME}')
