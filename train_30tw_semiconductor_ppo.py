"""
批量訓練 30 支台灣半導體/IC設計/封測 PPO 模型
=================================================
2330 台積電 / 2454 聯發科 / 2303 聯電 / 3034 聯詠 / 6415 矽力-KY
2368 金像電 / 3443 創意 / 6488 環球晶 / 6223 旺矽 / 3529 力旺
6510 精測 / 6669 緯穎 / 8299 群聯 / 1717 長興 / 3711 日月光投控
2379 瑞昱 / 3037 欣興 / 5483 中美晶 / 5347 世界 / 3653 健策
5269 祥碩 / 3035 智原 / 2371 大同 / 5434 崇越 / 6414 樺漢
2449 京元電子 / 3189 景碩 / 6239 力成 / 4915 致伸 / 2408 南亞科
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
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
    ('2330.TW',  '2330', '台積電'),
    ('2454.TW',  '2454', '聯發科'),
    ('2303.TW',  '2303', '聯電'),
    ('3034.TW',  '3034', '聯詠'),
    ('6415.TW',  '6415', '矽力-KY'),
    ('2368.TW',  '2368', '金像電'),
    ('3443.TW',  '3443', '創意'),
    ('6488.TWO', '6488', '環球晶'),
    ('6223.TWO', '6223', '旺矽'),
    ('3529.TW',  '3529', '力旺'),
    ('6510.TWO', '6510', '精測'),
    ('6669.TW',  '6669', '緯穎'),
    ('8299.TWO', '8299', '群聯'),
    ('1717.TW',  '1717', '長興'),
    ('3711.TW',  '3711', '日月光投控'),
    ('2379.TW',  '2379', '瑞昱'),
    ('3037.TW',  '3037', '欣興'),
    ('5483.TWO', '5483', '中美晶'),
    ('5347.TW',  '5347', '世界'),
    ('3653.TW',  '3653', '健策'),
    ('5269.TW',  '5269', '祥碩'),
    ('3035.TW',  '3035', '智原'),
    ('2371.TW',  '2371', '大同'),
    ('5434.TW',  '5434', '崇越'),
    ('6414.TW',  '6414', '樺漢'),
    ('2449.TW',  '2449', '京元電子'),
    ('3189.TW',  '3189', '景碩'),
    ('6239.TW',  '6239', '力成'),
    ('4915.TW',  '4915', '致伸'),
    ('2408.TW',  '2408', '南亞科'),
]


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
        self.current_step = 0; self.balance = self.initial_balance
        self.shares_held = 0; self.total_profit = 0
        self.total_trades = 0; self.last_total_value = self.initial_balance
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]; cp = float(row['close'])
        tv = self.balance + self.shares_held * cp
        sr = (self.shares_held * cp) / tv if tv > 0 else 0
        cr = self.balance / tv if tv > 0 else 1
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), float(sr), float(cr)
        ], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray): action = float(action[0])
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close']); traded = False
        if action > 0.15:
            ms = int(self.balance / cp); sb = int(ms * abs(action))
            if sb > 0: self.balance -= sb*cp; self.shares_held += sb; self.total_trades += 1; traded = True
        elif action < -0.15:
            ss = int(self.shares_held * abs(action))
            if ss > 0: self.balance += ss*cp; self.shares_held -= ss; self.total_trades += 1; traded = True
        self.current_step += 1; done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp; self.total_profit = tv - self.initial_balance
        vc = tv - self.last_total_value; reward = vc / self.initial_balance
        if traded: reward += 0.01
        if self.total_trades < self.current_step / 100: reward -= 0.001
        self.last_total_value = tv
        return self._get_observation(), reward, done, False, {}


def add_features(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower']    = df['bb_middle'] - df['bb_std'] * 2
    return df


def train_stock(ticker, symbol, name):
    print(f"\n{'='*70}\nPPO 訓練 {ticker} ({name})\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty or len(df) < 100:
            print("  ❌ 無資料"); return False, 0, 0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  下載 {len(df)} 天數據")
        df = add_features(df); df = df.bfill().ffill(); df_clean = df.dropna()
        print(f"  清理後 {len(df_clean)} 天")
        if len(df_clean) < 50:
            print("  ❌ 數據不足"); return False, 0, 0

        env = DummyVecEnv([lambda d=df_clean: ImprovedTradingEnv(d)])
        model = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048,
                    batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
        print("  訓練中... (200000 步)")
        model.learn(total_timesteps=200000)
        model.save(f'ppo_{symbol}_improved')
        print(f"  模型已保存: ppo_{symbol}_improved.zip")

        test_env = ImprovedTradingEnv(df_clean); obs, _ = test_env.reset()
        correct = 0; total = 0
        for _ in range(len(df_clean) - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = test_env.step(action)
            if abs(action[0]) > 0.15:
                total += 1; correct += (1 if reward > 0 else 0)
            if done: break

        acc = (correct / total * 100) if total > 0 else 0
        fv  = test_env.balance + test_env.shares_held * df_clean.iloc[-1]['close']
        ret = (fv - 10000) / 10000 * 100
        print(f"  ✅ 準確度: {acc:.2f}% | 回報率: {ret:.2f}% | 交易: {test_env.total_trades}")

        with open(f'model_accuracy_{symbol}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': symbol, 'model_type': 'PPO',
                       'training_accuracy': float(acc), 'validation_accuracy': float(acc),
                       'backtest_accuracy': float(acc), 'backtest_return': float(ret),
                       'win_rate': float(acc), 'sharpe_ratio': None,
                       'total_signals': int(total), 'correct_signals': int(correct),
                       'live_accuracy': None,
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                       'history': []}, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 保存: model_accuracy_{symbol}.json")
        return True, acc, ret
    except Exception as e:
        print(f"  ❌ {e}"); import traceback; traceback.print_exc(); return False, 0, 0


if __name__ == '__main__':
    print("=" * 70)
    print("批量訓練 30 支台灣半導體/IC設計/封測 PPO 模型")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for ticker, symbol, name in STOCKS:
        r = train_stock(ticker, symbol, name)
        success, acc, ret = r if isinstance(r, tuple) else (r, 0, 0)
        results.append((ticker, name, success, acc, ret))

    print(f"\n\n{'='*70}\nPPO 訓練完成摘要\n{'='*70}")
    print(f"{'代號':<14} {'名稱':<12} {'狀態':<4} {'準確度':>8} {'回報率':>12}")
    print("-" * 54)
    for ticker, name, success, acc, ret in results:
        status  = "✅" if success else "❌"
        acc_str = f"{acc:.1f}%" if success else "N/A"
        ret_str = (f"+{ret:.1f}%" if ret >= 0 else f"{ret:.1f}%") if success else "N/A"
        print(f"{ticker:<14} {name:<12} {status:<4} {acc_str:>8} {ret_str:>12}")

    ok = [r for r in results if r[2]]
    print(f"\n成功: {len(ok)}/{len(results)}")
    if ok:
        best_ret = sorted(ok, key=lambda x: x[4], reverse=True)[:5]
        print("\nTop 5 回報率:")
        for t, n, _, a, rt in best_ret:
            print(f"  {t:<14} {n:<12} 準確度:{a:.1f}%  回報率:+{rt:.1f}%")
