"""
批量訓練 3 支缺少模型的台股
1815 富喬工業 / 3013 晟鈦 / 3413 京鼎
XGBoost + PPO
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Try .TW first, fall back to .TWO
STOCKS = [
    ('1815.TW',  '1815_tw', '富喬工業'),
    ('3013.TW',  '3013_tw', '晟鈦'),
    ('3413.TW',  '3413_tw', '京鼎'),
]

FEATURE_COLUMNS = [
    'sma_10', 'sma_30', 'sma_50',
    'rsi', 'macd', 'macd_signal',
    'bb_upper', 'bb_lower', 'obv', 'volume_ratio',
]


def download_data(ticker):
    df = yf.download(ticker, start='2018-01-01', end='2026-12-31', progress=False)
    if df.empty or len(df) < 100:
        alt = ticker.replace('.TW', '.TWO')
        if alt != ticker:
            print(f"  .TW 無資料, 嘗試 {alt}")
            df = yf.download(alt, start='2018-01-01', end='2026-12-31', progress=False)
    return df


def add_features(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std']    = df['close'].rolling(20).std()
    df['bb_upper']  = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower']  = df['bb_middle'] - df['bb_std'] * 2

    df['obv']          = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['vol_ma20']     = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)

    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)
    return df


class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=100000):
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
        return self._obs(), {}

    def _obs(self):
        row = self.df.iloc[self.current_step]
        cp = float(row['close'])
        tv = self.balance + self.shares_held * cp
        sr = (self.shares_held * cp) / tv if tv > 0 else 0
        cr = self.balance / tv if tv > 0 else 1
        return np.array([
            float(self.shares_held), float(self.balance), cp,
            float(row.get('sma_10', cp)), float(row.get('sma_30', cp)), float(row.get('sma_50', cp)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', cp)), float(row.get('bb_lower', cp)), float(row.get('volume', 0)),
            float(self.total_profit), float(sr), float(cr),
        ], dtype=np.float32)

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        action = np.clip(action, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close'])
        traded = False
        if action > 0.15:
            ms = int(self.balance / cp)
            sb = int(ms * abs(action))
            if sb > 0:
                self.balance -= sb * cp; self.shares_held += sb
                self.total_trades += 1; traded = True
        elif action < -0.15:
            ss = int(self.shares_held * abs(action))
            if ss > 0:
                self.balance += ss * cp; self.shares_held -= ss
                self.total_trades += 1; traded = True
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        tv = self.balance + self.shares_held * cp
        self.total_profit = tv - self.initial_balance
        vc = tv - self.last_total_value
        reward = vc / self.initial_balance
        if traded:
            reward += 0.01
        if self.total_trades < self.current_step / 100:
            reward -= 0.001
        self.last_total_value = tv
        return self._obs(), reward, done, False, {}


def train_one(ticker, symbol, name):
    print(f"\n{'='*60}")
    print(f"  訓練 {name} ({ticker})")
    print('='*60)

    df = download_data(ticker)
    if df.empty or len(df) < 100:
        print(f"  ❌ 無法下載資料")
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={
        'Close': 'close', 'Volume': 'volume',
        'Open': 'open', 'High': 'high', 'Low': 'low',
    }).reset_index()
    print(f"  下載 {len(df)} 天數據  ({df['Date'].iloc[0].date()} → {df['Date'].iloc[-1].date()})")

    df = add_features(df)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])
    print(f"  清理後 {len(df_clean)} 天")
    if len(df_clean) < 100:
        print("  ❌ 數據不足"); return None

    # ── XGBoost ─────────────────────────────────────────────────────────
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=300,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss',
    )
    xgb_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, xgb_model.predict(X_train)) * 100
    test_acc  = accuracy_score(y_test,  xgb_model.predict(X_test))  * 100
    print(f"  XGBoost  train={train_acc:.1f}%  test={test_acc:.1f}%")

    # XGB backtest ROI
    bh_start = float(df_clean['close'].iloc[int(len(df_clean)*0.8)])
    bh_end   = float(df_clean['close'].iloc[-1])
    bh_roi   = (bh_end / bh_start - 1) * 100

    proba = xgb_model.predict_proba(X_test)
    pred  = xgb_model.predict(X_test)
    wins  = sum(1 for p, a in zip(pred, y_test.values) if p == a)
    xgb_win = wins / len(pred) * 100

    model_file = f'xgb_{symbol}_model.pkl'
    joblib.dump(xgb_model, model_file)
    print(f"  Saved: {model_file}")

    # ── PPO ─────────────────────────────────────────────────────────────
    env = DummyVecEnv([lambda: TradingEnv(df_clean)])
    ppo_model = PPO('MlpPolicy', env, learning_rate=3e-4, n_steps=2048,
                    batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
    print(f"  PPO 訓練中... (60000 步)")
    ppo_model.learn(total_timesteps=60000)

    ppo_save = f'ppo_{symbol}_improved'
    ppo_model.save(ppo_save)
    print(f"  Saved: {ppo_save}.zip")

    # PPO backtest
    test_env = TradingEnv(df_clean)
    obs, _ = test_env.reset()
    for _ in range(len(df_clean) - 1):
        action, _ = ppo_model.predict(obs, deterministic=True)
        obs, _, done, _, _ = test_env.step(action)
        if done: break
    fv  = test_env.balance + test_env.shares_held * float(df_clean.iloc[-1]['close'])
    ppo_roi = (fv - 100000) / 100000 * 100

    print(f"\n  XGBoost  acc={test_acc:.1f}%  win={xgb_win:.0f}%")
    print(f"  PPO      ROI={ppo_roi:+.1f}%  trades={test_env.total_trades}")
    print(f"  Buy&Hold ROI={bh_roi:+.1f}%")

    return {
        'ticker': ticker, 'symbol': symbol, 'name': name,
        'xgb_acc': test_acc, 'xgb_win': xgb_win,
        'ppo_roi': ppo_roi, 'bh_roi': bh_roi,
        'rows': len(df_clean),
    }


if __name__ == '__main__':
    print("=" * 60)
    print(f"  批量訓練 缺少模型的台股  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results = []
    for ticker, symbol, name in STOCKS:
        r = train_one(ticker, symbol, name)
        results.append((ticker, name, r))

    print(f"\n\n{'='*60}")
    print(f"  完成摘要  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print('='*60)
    print(f"  {'代號':<10} {'名稱':<12} {'XGB':<8} {'PPO ROI':<10} {'B&H ROI'}")
    print("  " + "-"*50)
    for ticker, name, r in results:
        if r:
            tag = '✅'
            print(f"  {tag} {ticker:<8} {name:<12} {r['xgb_acc']:.1f}%   {r['ppo_roi']:+.1f}%      {r['bh_roi']:+.1f}%")
        else:
            print(f"  ❌ {ticker:<8} {name:<12} FAILED")
    print()
