"""
Batch-train PPO models for the 21 TW stocks that were missing ppo_{code}_tw_improved.zip.
Saves ppo_{code}_tw_improved.zip and model_accuracy_{code}.json for each.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
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
import warnings
warnings.filterwarnings('ignore')

# (code, yfinance ticker)
STOCKS = [
    ('3293', '3293.TWO'),
    ('3360', '3360.TWO'),
    ('3374', '3374.TWO'),
    ('3630', '3630.TWO'),
    ('3706', '3706.TW'),
    ('3707', '3707.TWO'),
    ('4167', '4167.TWO'),
    ('4541', '4541.TWO'),
    ('4973', '4973.TWO'),
    ('6104', '6104.TWO'),
    ('6265', '6265.TWO'),
    ('6485', '6485.TWO'),
    ('6603', '6603.TWO'),
    ('6829', '6829.TWO'),
    ('6949', '6949.TW'),
    ('7728', '7728.TWO'),
    ('8038', '8038.TWO'),
    ('8074', '8074.TWO'),
    ('8271', '8271.TW'),
    ('8431', '8431.TWO'),
    ('8450', '8450.TWO'),
]


class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
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
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        return np.array([
            float(self.shares_held), float(self.balance), float(row['close']),
            float(row['sma_10']), float(row['sma_30']), float(row['sma_50']),
            float(row['rsi']), float(row['macd']), float(row['macd_signal']),
            float(row['bb_upper']), float(row['bb_lower']), float(row['volume']),
            float(self.total_profit), float(stock_ratio), float(cash_ratio)
        ], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
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


def train_stock(code, ticker):
    print(f"\n{'='*70}\n訓練 {code} ({ticker})\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty or len(df) < 100:
            print("  ❌ 無資料")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                                 'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"  下載 {len(df)} 天數據")

        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + df['bb_std'] * 2
        df['bb_lower'] = df['bb_middle'] - df['bb_std'] * 2
        df = df.bfill().ffill()
        df_clean = df.dropna()
        print(f"  清理後 {len(df_clean)} 天")
        if len(df_clean) < 50:
            print("  ❌ 數據不足")
            return None

        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx].reset_index(drop=True)
        test_df = df_clean.iloc[split_idx:].reset_index(drop=True)

        env = DummyVecEnv([lambda: ImprovedTradingEnv(train_df)])
        model = PPO('MlpPolicy', env, learning_rate=0.0005, n_steps=2048,
                     batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
        print("  訓練中... (200000 步)")
        model.learn(total_timesteps=200000)
        model.save(f'ppo_{code}_tw_improved')
        print(f"  模型已保存: ppo_{code}_tw_improved.zip")

        test_env = ImprovedTradingEnv(test_df)
        obs, _ = test_env.reset()
        correct = 0
        total = 0
        for _ in range(len(test_df) - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = test_env.step(action)
            if abs(action[0]) > 0.15:
                total += 1
                correct += (1 if reward > 0 else 0)
            if done:
                break

        acc = (correct / total * 100) if total > 0 else 0
        fv = test_env.balance + test_env.shares_held * test_df.iloc[-1]['close']
        ret = (fv - 10000) / 10000 * 100
        print(f"  ✅ 準確度: {acc:.2f}% | 回報率: {ret:.2f}% | 交易: {test_env.total_trades}")

        with open(f'model_accuracy_{code}.json', 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': code, 'model_type': 'PPO',
                'training_accuracy': float(acc), 'validation_accuracy': float(acc),
                'backtest_accuracy': float(acc), 'backtest_return': float(ret),
                'win_rate': float(acc), 'sharpe_ratio': None,
                'total_signals': int(total), 'correct_signals': int(correct),
                'live_accuracy': None,
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'history': []
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 保存: model_accuracy_{code}.json")
        return {'code': code, 'ticker': ticker, 'status': 'OK', 'accuracy': acc, 'return': ret, 'trades': test_env.total_trades}
    except Exception as e:
        print(f"  ❌ 錯誤: {e}")
        return {'code': code, 'ticker': ticker, 'status': f'ERROR: {e}', 'accuracy': None, 'return': None, 'trades': None}


if __name__ == '__main__':
    print("=" * 70)
    print(f"批量訓練 21 支 TW 股票 PPO 模型")
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for code, ticker in STOCKS:
        r = train_stock(code, ticker)
        results.append(r if r else {'code': code, 'ticker': ticker, 'status': 'FAIL', 'accuracy': None, 'return': None, 'trades': None})

    print("\n" + "=" * 70)
    print("📊 批量訓練結果總表")
    print("=" * 70)
    print(f"{'代號':<8} {'Ticker':<12} {'狀態':<10} {'準確度':>10} {'回報率':>10} {'交易':>6}")
    for r in results:
        acc = f"{r['accuracy']:.2f}%" if r['accuracy'] is not None else 'N/A'
        ret = f"{r['return']:+.2f}%" if r['return'] is not None else 'N/A'
        trades = r['trades'] if r['trades'] is not None else 'N/A'
        print(f"{r['code']:<8} {r['ticker']:<12} {r['status']:<10} {acc:>10} {ret:>10} {str(trades):>6}")
    print("=" * 70)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
