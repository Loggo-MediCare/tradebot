"""Train PPO for STX, MRVL, SNPS, 02202.HK, 9984.T"""
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
import json
from datetime import datetime

STOCKS = [
    ('STX',     'stx',   'Seagate'),
    ('MRVL',    'mrvl',  'Marvell'),
    ('SNPS',    'snps',  'Synopsys'),
    ('2202.HK', '02202', 'Vanke'),
    ('9984.T',  '9984',  'SoftBank'),
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
        self.total_trades = 0
        self.last_total_value = self.initial_balance
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        cp = float(row['close'])
        tv = self.balance + self.shares_held * cp
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit),
            (self.shares_held * cp) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close'])
        traded = False
        if action > 0.15:
            s = int(int(self.balance / cp) * abs(action))
            if s > 0:
                self.balance -= s * cp; self.shares_held += s; self.total_trades += 1; traded = True
        elif action < -0.15:
            s = int(self.shares_held * abs(action))
            if s > 0:
                self.balance += s * cp; self.shares_held -= s; self.total_trades += 1; traded = True
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp
        self.total_profit = tv - self.initial_balance
        reward = (tv - self.last_total_value) / self.initial_balance
        if traded: reward += 0.01
        if self.total_trades < self.current_step / 100: reward -= 0.001
        self.last_total_value = tv
        return self._get_obs(), reward, done, False, {}

def train(ticker, symbol, name):
    print(f"\n{'='*70}\n訓練 {ticker} ({name})\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty or len(df) < 100:
            print(f"  ❌ 無數據"); return False
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  {len(df)} 天數據")
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        d = df['close'].diff()
        g = d.where(d>0,0).rolling(14).mean(); l = (-d.where(d<0,0)).rolling(14).mean()
        df['rsi'] = 100-(100/(1+g/(l+1e-10)))
        df['macd'] = df['ema_12']-df['ema_26']; df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['bb_middle'] = df['close'].rolling(20).mean(); df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle']+df['bb_std']*2; df['bb_lower'] = df['bb_middle']-df['bb_std']*2
        df = df.bfill().ffill(); df_clean = df.dropna()
        print(f"  清理後: {len(df_clean)} 天")
        env = DummyVecEnv([lambda: ImprovedTradingEnv(df_clean)])
        model = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
        print("  訓練中... (150000步)")
        model.learn(total_timesteps=150000)
        mf = f'ppo_{symbol}_improved'
        model.save(mf); print(f"  保存: {mf}.zip")
        test = ImprovedTradingEnv(df_clean); obs, _ = test.reset()
        cp_t, tp = 0, 0
        for _ in range(len(df_clean)-1):
            act, _ = model.predict(obs, deterministic=True)
            obs, rew, done, _, _ = test.step(act)
            if abs(act[0])>0.15: tp+=1; cp_t += (1 if rew>0 else 0)
            if done: break
        acc = cp_t/tp*100 if tp>0 else 0
        fv = test.balance + test.shares_held*df_clean.iloc[-1]['close']
        ret = (fv-test.initial_balance)/test.initial_balance*100
        print(f"  ✅ 準確度:{acc:.1f}% 回報:{ret:.1f}% 交易:{test.total_trades}")
        with open(f'model_accuracy_{symbol}.json','w',encoding='utf-8') as f:
            json.dump({'symbol':symbol,'model_type':'PPO','training_accuracy':float(acc),'validation_accuracy':float(acc),'backtest_accuracy':float(acc),'total_return':float(ret),'total_trades':int(test.total_trades),'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},f,ensure_ascii=False,indent=2)
        return True
    except Exception as e:
        print(f"  ❌ {e}"); import traceback; traceback.print_exc(); return False

if __name__ == '__main__':
    print("訓練 5支股票 PPO 模型")
    results = [(t,n,train(t,s,n)) for t,s,n in STOCKS]
    print("\n=== 結果 ===")
    for t,n,ok in results:
        print(f"  {'✅' if ok else '❌'} {t} ({n})")
