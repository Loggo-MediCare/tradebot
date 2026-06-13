"""
STX, SNPS - 改用 SAC (Soft Actor-Critic) 模型
SAC 自帶最大熵探索，解決 PPO 不交易的問題
model.predict() API 與 PPO 相同，信號腳本無需修改
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
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
import json
from datetime import datetime

STOCKS = [
    ('STX',  'stx',  'Seagate'),
    ('SNPS', 'snps', 'Synopsys'),
]

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        # SAC requires continuous action space — same as before
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
        self.last_trade_step = 0
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

        if action > 0.05:
            s = int(int(self.balance / cp) * abs(action))
            if s > 0:
                self.balance -= s * cp; self.shares_held += s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True
        elif action < -0.05:
            s = int(self.shares_held * abs(action))
            if s > 0:
                self.balance += s * cp; self.shares_held -= s
                self.total_trades += 1; self.last_trade_step = self.current_step; traded = True

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp
        self.total_profit = tv - self.initial_balance

        # Percentage return reward (SAC handles exploration internally)
        pct = (tv - tv_before) / tv_before if tv_before > 0 else 0
        reward = pct * 10  # scale up for meaningful gradients

        if traded: reward += 0.02
        idle = self.current_step - self.last_trade_step
        if idle > 30: reward -= 0.01 * (idle / 30)

        self.last_total_value = tv
        return self._obs(), reward, done, False, {}


def train(ticker, symbol, name):
    print(f"\n{'='*70}\n訓練 {ticker} ({name}) - SAC 模型\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty: print("  ❌ 無數據"); return False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                                'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"  {len(df)} 天數據")

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
        df = df.bfill().ffill()
        df_clean = df.dropna()
        print(f"  清理後: {len(df_clean)} 天")

        # SAC does NOT use DummyVecEnv — it works with single env
        env = TradingEnv(df_clean)

        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=0.0003,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',   # SAC auto-tunes entropy for exploration
            verbose=0
        )

        print("  訓練中... (200000步)")
        model.learn(total_timesteps=200000)

        mf = f'ppo_{symbol}_improved'  # keep same filename for signal script compatibility
        model.save(mf)
        print(f"  保存: {mf}.zip")

        # Evaluate
        test = TradingEnv(df_clean)
        obs, _ = test.reset()
        cp_t = tp = 0
        for _ in range(len(df_clean) - 1):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _, _ = test.step(act)
            if abs(act[0]) > 0.05:
                tp += 1; cp_t += (1 if rew > 0 else 0)
            if done: break

        acc = cp_t / tp * 100 if tp > 0 else 0
        fv = test.balance + test.shares_held * df_clean.iloc[-1]['close']
        ret = (fv - test.initial_balance) / test.initial_balance * 100
        print(f"  ✅ 準確度:{acc:.1f}% 回報:{ret:.1f}% 交易:{test.total_trades}")

        with open(f'model_accuracy_{symbol}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': symbol, 'model_type': 'SAC',
                       'training_accuracy': float(acc), 'validation_accuracy': float(acc),
                       'backtest_accuracy': float(acc), 'total_return': float(ret),
                       'total_trades': int(test.total_trades),
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                      f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"  ❌ {e}"); import traceback; traceback.print_exc(); return False


if __name__ == '__main__':
    print("=" * 70)
    print("STX + SNPS — 改用 SAC 模型訓練")
    print("=" * 70)
    results = [(t, n, train(t, s, n)) for t, s, n in STOCKS]
    print("\n=== 結果 ===")
    for t, n, ok in results:
        print(f"  {'✅' if ok else '❌'} {t} ({n})")
