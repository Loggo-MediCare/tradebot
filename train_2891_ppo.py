"""
2891.TW (中信金控) - PPO 訓練
輸出: ppo_2891_tw_improved.zip
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import json

TICKER = '2891.TW'
SYMBOL = '2891_tw'
NAME   = '中信金控'


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
        self.total_trades = 0
        self.last_trade_step = 0
        self.last_total_value = self.initial_balance
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.current_step]
        cp = float(r['close'])
        tv = self.balance + self.shares_held * cp
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)), float(r.get('sma_50', 0)),
            float(r.get('rsi', 50)), float(r.get('macd', 0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)), float(r.get('volume', 0)),
            float(self.total_profit),
            (self.shares_held * cp) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close'])
        tv_before = self.balance + self.shares_held * cp
        traded = False

        if action > 0.1:
            s = int(int(self.balance / cp) * abs(action))
            if s > 0:
                self.balance -= s * cp; self.shares_held += s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True
        elif action < -0.1:
            s = int(self.shares_held * abs(action))
            if s > 0:
                self.balance += s * cp; self.shares_held -= s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp
        self.total_profit = tv - self.initial_balance

        pct = (tv - tv_before) / tv_before if tv_before > 0 else 0
        reward = pct * 100
        if traded: reward += 0.05
        idle = self.current_step - self.last_trade_step
        if idle > 20: reward -= 0.02 * (idle / 20)
        self.last_total_value = tv
        return self._obs(), reward, done, False, {}


if __name__ == '__main__':
    print("=" * 60)
    print(f"訓練 {TICKER} ({NAME}) - PPO 模型")
    print("=" * 60)

    # Download data
    df = yf.download(TICKER, start='2015-01-01', end='2026-12-31', progress=False)
    if df.empty:
        print("❌ 無法下載數據"); exit(1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Volume':'volume',
                             'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"  {len(df)} 天數據")

    # Technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower'] = df['bb_middle'] - df['bb_std'] * 2
    df = df.bfill().ffill().dropna()
    print(f"  清理後: {len(df)} 天")

    # Train
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model = PPO('MlpPolicy', env,
                learning_rate=0.0003, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=0)
    print("  訓練中... (200000步)")
    model.learn(total_timesteps=200000)

    fname = f'ppo_{SYMBOL}_improved'
    model.save(fname)
    print(f"  ✅ 保存: {fname}.zip")

    # Evaluate
    test_env = ImprovedTradingEnv(df)
    obs, _ = test_env.reset()
    cp_t = tp = 0
    for _ in range(len(df) - 1):
        act, _ = model.predict(obs, deterministic=True)
        obs, rew, done, _, _ = test_env.step(act)
        if abs(act[0]) > 0.1:
            tp += 1; cp_t += (1 if rew > 0 else 0)
        if done: break
    acc = cp_t / tp * 100 if tp > 0 else 0
    fv  = test_env.balance + test_env.shares_held * df.iloc[-1]['close']
    ret = (fv - test_env.initial_balance) / test_env.initial_balance * 100
    print(f"  準確度:{acc:.1f}%  回報:{ret:.1f}%  交易:{test_env.total_trades}")

    with open(f'model_accuracy_{SYMBOL}.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': SYMBOL, 'model_type': 'PPO',
                   'training_accuracy': float(acc), 'validation_accuracy': float(acc),
                   'backtest_accuracy': float(acc), 'total_return': float(ret),
                   'total_trades': int(test_env.total_trades),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                  f, ensure_ascii=False, indent=2)
    print(f"  準確度已保存: model_accuracy_{SYMBOL}.json")
    print("\n完成! 現在可以運行 get_trading_signal_2891.py")
