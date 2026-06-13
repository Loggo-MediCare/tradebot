"""Batch train XGBoost + PPO for all requested US stocks (XGBoost + PPO comparison)"""
import os, json, warnings, sys, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── All stocks requested across all batches ──────────────────────────────────
STOCKS = [
    # (ticker, company,                  start_date)
    ('ARM',  'ARM Holdings',             '2023-09-14'),  # IPO Sep 2023
    ('AMD',  'Advanced Micro Devices',   '2015-01-01'),
    ('TXN',  'Texas Instruments',        '2015-01-01'),
    ('INTC', 'Intel Corporation',        '2015-01-01'),
    ('MRVL', 'Marvell Technology',       '2015-01-01'),
    ('BKR',  'Baker Hughes',             '2017-07-01'),
    ('MCHP', 'Microchip Technology',     '2015-01-01'),
    ('NXPI', 'NXP Semiconductors',       '2015-01-01'),
    ('SNPS', 'Synopsys',                 '2015-01-01'),
    ('MPWR', 'Monolithic Power Systems', '2015-01-01'),
    ('URI',  'United Rentals',           '2015-01-01'),
    ('ON',   'ON Semiconductor',         '2015-01-01'),
    ('VRK',  'VRK',                      '2015-01-01'),
    ('GEV',  'GE Vernova',               '2024-04-02'),  # Spinoff Apr 2024
    ('STLD', 'Steel Dynamics',           '2015-01-01'),
]

END_DATE        = '2026-05-01'
TOTAL_TIMESTEPS = 100_000
FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','sma_200','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']


def build_features(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi']          = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']         = df['ema_12'] - df['ema_26']
    df['macd_signal']  = df['macd'].ewm(span=9).mean()
    df['macd_hist']    = df['macd'] - df['macd_signal']
    df['bb_m']         = df['close'].rolling(20).mean()
    df['bb_s']         = df['close'].rolling(20).std()
    df['bb_u']         = df['bb_m'] + 2 * df['bb_s']
    df['bb_l']         = df['bb_m'] - 2 * df['bb_s']
    df['bb_position']  = ((df['close'] - df['bb_l']) / (df['bb_u'] - df['bb_l']) * 100).fillna(50)
    lo14               = df['low'].rolling(14).min()
    hi14               = df['high'].rolling(14).max()
    df['K']            = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D']            = df['K'].rolling(3).mean()
    df['obv']          = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20']     = df['obv'].rolling(20).mean()
    df['volatility']   = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl  = df['high'] - df['low']
    hc  = np.abs(df['high'] - df['close'].shift())
    lc  = np.abs(df['low']  - df['close'].shift())
    df['atr']            = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    df['future_return']    = df['close'].shift(-5) / df['close'] - 1
    df['target']           = (df['future_return'] > 0.02).astype(int)
    return df.bfill().ffill()


class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0; self.bal = 10_000.0; self.sh = 0; self.profit = 0.0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.i]; p = float(r['close']); tv = self.bal + self.sh * p
        return np.array([
            float(self.sh), float(self.bal), p,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)), float(r.get('sma_50', 0)),
            float(r.get('rsi', 50)), float(r.get('macd', 0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_u', 0)), float(r.get('bb_l', 0)), float(r.get('volume', 0)),
            float(self.profit),
            (self.sh * p) / tv if tv > 0 else 0,
            self.bal / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1, 1); p = float(self.df.iloc[self.i]['close'])
        if a < -0.1:
            s = int(self.sh * abs(a))
            if s > 0: self.bal += s * p; self.sh -= s
        elif a > 0.1:
            s = int((self.bal // p) * a)
            if s > 0: self.bal -= s * p; self.sh += s
        self.profit = (self.bal + self.sh * p) - 10_000.0
        self.i += 1; done = self.i >= len(self.df) - 1
        rew = self.profit / 10_000.0 + (0.01 if abs(a) > 0.1 else 0)
        return self._obs(), rew, done, False, {}


def train_stock(ticker, company, start_date):
    print(f"\n{'='*60}")
    print(f"  Training {ticker} — {company}")
    print(f"{'='*60}")

    # Download
    try:
        df = yf.download(ticker, start=start_date, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 50:
            print(f"  [SKIP] Only {len(df)} rows — not enough data")
            return None
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                 'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  Data: {len(df)} trading days")
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return None

    df = build_features(df)
    dc = df.dropna(subset=FEAT + ['target'])
    if len(dc) < 50:
        print(f"  [SKIP] Only {len(dc)} clean rows after feature build")
        return None

    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy()
    te = dc.iloc[split:].copy()
    print(f"  Train: {len(tr)} | Test: {len(te)}")

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print(f"  [XGBoost] Training...")
    X = dc[FEAT]; y = dc['target']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
    xgb_model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=200,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss',
    )
    xgb_model.fit(Xt, yt)
    xgb_train_acc = accuracy_score(yt, xgb_model.predict(Xt))
    xgb_acc       = accuracy_score(ye, xgb_model.predict(Xe))
    t_low = ticker.lower()
    joblib.dump(xgb_model, f'xgb_{t_low}_model.pkl')
    with open(f'model_accuracy_{ticker}.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': ticker, 'model_type': 'XGBoost',
                   'training_accuracy': round(xgb_train_acc * 100, 2),
                   'validation_accuracy': round(xgb_acc * 100, 2),
                   'backtest_accuracy': round(xgb_acc * 100, 2),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f"  [XGBoost] Accuracy: {xgb_acc*100:.2f}%")

    # ── PPO ──────────────────────────────────────────────────────────────────
    print(f"  [PPO] Training ({TOTAL_TIMESTEPS//1000}k steps)...")
    env = DummyVecEnv([lambda: TradingEnv(tr)])
    ppo_model = PPO('MlpPolicy', env, verbose=0,
                    learning_rate=0.0003, n_steps=2048, batch_size=64,
                    n_epochs=10, gamma=0.99, ent_coef=0.01)
    ppo_model.learn(total_timesteps=TOTAL_TIMESTEPS)
    ppo_model.save(f'ppo_{t_low}_improved')

    env2 = TradingEnv(te.reset_index(drop=True))
    obs, _ = env2.reset()
    preds, labels = [], []
    for i in range(len(te) - 6):
        act, _ = ppo_model.predict(obs, deterministic=True)
        preds.append(1 if float(act[0]) > 0.1 else 0)
        labels.append(int(te.iloc[i]['target']))
        obs, _, done, _, _ = env2.step(act)
        if done: break
    ppo_acc = accuracy_score(labels, preds) if preds else 0.0
    with open(f'model_accuracy_{ticker}_ppo.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': ticker, 'model_type': 'PPO',
                   'training_accuracy': round(ppo_acc * 100, 2),
                   'validation_accuracy': round(ppo_acc * 100, 2),
                   'backtest_accuracy': round(ppo_acc * 100, 2),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f"  [PPO]     Accuracy: {ppo_acc*100:.2f}%  (buy signals: {sum(preds)}/{len(preds)})")

    winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
    print(f"  Winner:   {winner}  (diff: {abs(xgb_acc - ppo_acc)*100:.2f}%)")
    return {
        'ticker':   ticker,
        'company':  company,
        'xgb_acc':  round(xgb_acc  * 100, 2),
        'ppo_acc':  round(ppo_acc  * 100, 2),
        'winner':   winner,
        'days':     len(df),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("  BATCH TRAINING — US Stocks (XGBoost + PPO)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total:   {len(STOCKS)} stocks")
    print("=" * 60)

    results, failed = [], []

    for ticker, company, start_date in STOCKS:
        try:
            r = train_stock(ticker, company, start_date)
            if r:
                results.append(r)
            else:
                failed.append(ticker)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")
            failed.append(ticker)

    # ── Final comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("  ACCURACY COMPARISON TABLE")
    print("=" * 74)
    print(f"  {'Ticker':<7} {'Company':<28} {'XGBoost':>9} {'PPO':>7} {'Winner':<10}")
    print("  " + "-" * 68)
    for r in sorted(results, key=lambda x: max(x['xgb_acc'], x['ppo_acc']), reverse=True):
        best = max(r['xgb_acc'], r['ppo_acc'])
        star = "🌟" if best >= 70 else "✅" if best >= 60 else "⚠️ "
        print(f"  {r['ticker']:<7} {r['company']:<28} {r['xgb_acc']:>8.2f}% {r['ppo_acc']:>6.2f}% {r['winner']:<10} {star}")
    print("  " + "-" * 68)
    print(f"  Trained: {len(results)} stocks  |  Failed/Skipped: {len(failed)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 74)
