"""
Batch training: XGBoost + PPO for multiple stocks with accuracy comparison.
Stocks: 2345.TW, 3163.TWO, 3081.TWO, 6442.TW, 8046.TW
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

STOCKS = [
    # Newly identified stocks
    {'code': '8027', 'ticker': '8027.TWO', 'name': '鈦昇'},
    {'code': '5351', 'ticker': '5351.TWO', 'name': '鈺創'},
    {'code': '4720', 'ticker': '4720.TW',  'name': '德淵'},
    {'code': '6176', 'ticker': '6176.TW',  'name': '瑞儀'},
    {'code': '3380', 'ticker': '3380.TW',  'name': '明泰'},
    {'code': '6672', 'ticker': '6672.TW',  'name': '騰輝電子KY'},
    {'code': '6213', 'ticker': '6213.TW',  'name': '聯茂'},
    {'code': '8103', 'ticker': '8103.TW',  'name': '瀚荃'},
    {'code': '1569', 'ticker': '1569.TWO', 'name': '濱川'},
    {'code': '1595', 'ticker': '1595.TWO', 'name': '川寶'},
    {'code': '6108', 'ticker': '6108.TW',  'name': '競國'},
    {'code': '4951', 'ticker': '4951.TWO', 'name': '精拓科'},
    {'code': '1727', 'ticker': '1727.TW',  'name': '中華化'},
    {'code': '3105', 'ticker': '3105.TWO', 'name': '穩懋'},
    # Image 1 - need training
    {'code': '3583', 'ticker': '3583.TW',  'name': '辛耘'},
    {'code': '8028', 'ticker': '8028.TW',  'name': '昇陽半導體'},
    {'code': '3680', 'ticker': '3680.TWO', 'name': '家登'},
    {'code': '4749', 'ticker': '4749.TWO', 'name': '新應材'},
    {'code': '4772', 'ticker': '4772.TWO', 'name': '台特化'},
    {'code': '6788', 'ticker': '6788.TWO', 'name': '華景電'},
    {'code': '7703', 'ticker': '7703.TWO', 'name': '銳澤'},
    {'code': '3131', 'ticker': '3131.TWO', 'name': '弘塑'},
    {'code': '8147', 'ticker': '8147.TWO', 'name': '大綜'},
    {'code': '6223', 'ticker': '6223.TWO', 'name': '旺矽'},
    {'code': '6515', 'ticker': '6515.TW',  'name': '穎崴'},
    # Image 2 - new stocks
    {'code': '2404', 'ticker': '2404.TW',  'name': '漢唐'},
    {'code': '6196', 'ticker': '6196.TW',  'name': '帆宣'},
    {'code': '6605', 'ticker': '6605.TW',  'name': '信紘科'},
    {'code': '6139', 'ticker': '6139.TW',  'name': '亞翔'},
    {'code': '8071', 'ticker': '8071.TWO', 'name': '竹陸科技'},
    {'code': '1560', 'ticker': '1560.TW',  'name': '中砂'},
    {'code': '6438', 'ticker': '6438.TW',  'name': '迅得'},
    {'code': '6449', 'ticker': '6449.TW',  'name': '倍利科'},
]

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]

# ─── Technical indicators ──────────────────────────────────────────────────────
def add_indicators(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower']    = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = ((df['close'] - df['bb_lower']) /
                         (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)

    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    hl  = df['high'] - df['low']
    hc  = np.abs(df['high'] - df['close'].shift())
    lc  = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target']        = (df['future_return'] > 0.02).astype(int)
    return df.bfill().ffill()


# ─── XGBoost ──────────────────────────────────────────────────────────────────
def train_xgboost(df, code, ticker):
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])
    if len(df_clean) < 100:
        return None, 0.0

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=200,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, model.predict(X_te))

    suffix = 'two' if '.TWO' in ticker else 'tw'
    pkl_file = f'xgb_{code}_{suffix}_model.pkl'
    joblib.dump(model, pkl_file)

    acc_data = {
        'symbol': ticker, 'model_type': 'XGBoost',
        'training_accuracy': float(accuracy_score(y_tr, model.predict(X_tr)) * 100),
        'validation_accuracy': float(acc * 100),
        'backtest_accuracy':   float(acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    suffix_up = 'TWO' if '.TWO' in ticker else 'TW'
    with open(f'model_accuracy_{code}_{suffix_up}.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)

    return model, acc


# ─── PPO environment ──────────────────────────────────────────────────────────
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
        self.balance = 10000.0
        self.shares = 0
        self.profit = 0.0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.step_idx]
        p = float(r['close'])
        tv = self.balance + self.shares * p
        return np.array([
            float(self.shares), float(self.balance), p,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)),
            float(r.get('sma_50', 0)), float(r.get('rsi', 50)),
            float(r.get('macd', 0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)),
            float(r.get('volume', 0)), float(self.profit),
            (self.shares * p) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1.0, 1.0)
        p = float(self.df.iloc[self.step_idx]['close'])
        if a < -0.1:
            s = int(self.shares * abs(a))
            if s > 0:
                self.balance += s * p; self.shares -= s
        elif a > 0.1:
            s = int((self.balance // p) * a)
            if s > 0:
                self.balance -= s * p; self.shares += s
        self.profit = (self.balance + self.shares * p) - 10000.0
        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        reward = self.profit / 10000.0
        if abs(a) > 0.1: reward += 0.01
        if self.balance > (self.balance + self.shares * p) * 0.9: reward -= 0.005
        return self._obs(), reward, done, False, {}


def train_ppo(train_df, test_df, code, ticker, timesteps=100000):
    env = DummyVecEnv([lambda: TradingEnv(train_df)])
    model = PPO('MlpPolicy', env, verbose=0,
                learning_rate=0.0003, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)

    suffix = 'two' if '.TWO' in ticker else 'tw'
    model_path = f'ppo_{code}_{suffix}_improved'
    model.save(model_path)

    # Evaluate accuracy on test set
    env2 = TradingEnv(test_df.reset_index(drop=True))
    obs, _ = env2.reset()
    preds, labels = [], []
    for i in range(len(test_df) - 6):
        act, _ = model.predict(obs, deterministic=True)
        preds.append(1 if float(act[0]) > 0.1 else 0)
        labels.append(int(test_df.iloc[i]['target']))
        obs, _, done, _, _ = env2.step(act)
        if done: break

    acc = accuracy_score(labels, preds) if preds else 0.0

    suffix_up = 'TWO' if '.TWO' in ticker else 'TW'
    acc_data = {
        'symbol': ticker, 'model_type': 'PPO',
        'training_accuracy': float(acc * 100),
        'validation_accuracy': float(acc * 100),
        'backtest_accuracy':   float(acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f'model_accuracy_{code}_{suffix_up}_ppo.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)

    return acc


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    results = []

    for s in STOCKS:
        code, ticker, name = s['code'], s['ticker'], s['name']
        print(f"\n{'='*70}")
        print(f"  {ticker} ({name})")
        print(f"{'='*70}")

        # Download
        df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
        if df.empty:
            print(f"  ERROR: no data for {ticker}")
            results.append({'ticker': ticker, 'name': name, 'xgb': None, 'ppo': None})
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  Data: {len(df)} days")

        df = add_indicators(df)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])

        split = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split].copy()
        test_df  = df_clean.iloc[split:].copy()

        # XGBoost
        print(f"  [XGBoost] Training...")
        _, xgb_acc = train_xgboost(df_clean, code, ticker)
        print(f"  [XGBoost] Test accuracy: {xgb_acc*100:.2f}%")

        # PPO
        print(f"  [PPO] Training (100k steps)...")
        ppo_acc = train_ppo(train_df, test_df, code, ticker)
        print(f"  [PPO] Test accuracy: {ppo_acc*100:.2f}%")

        results.append({'ticker': ticker, 'name': name,
                        'xgb': xgb_acc*100, 'ppo': ppo_acc*100})

    # Summary table
    print(f"\n{'='*70}")
    print("  📊 ACCURACY COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Ticker':<12} {'Name':<12} {'XGBoost':>10} {'PPO':>10} {'Winner':>10}")
    print(f"  {'-'*56}")
    for r in results:
        if r['xgb'] is None:
            print(f"  {r['ticker']:<12} {r['name']:<12} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        else:
            winner = 'XGBoost' if r['xgb'] >= r['ppo'] else 'PPO'
            print(f"  {r['ticker']:<12} {r['name']:<12} {r['xgb']:>9.2f}% {r['ppo']:>9.2f}% {winner:>10}")
    print(f"{'='*70}")
