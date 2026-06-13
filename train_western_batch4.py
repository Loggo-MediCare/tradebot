"""
Train XGBoost + PPO for western stocks (Batch 4)
Stocks: JPM, XLE, BRK-A, BRK-B, MCD, LIN, RDDT, KO, DIS, CRM, LULU, XOP, SMH, SHOP, COIN, RSP
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

END_DATE   = '2026-05-30'
TRAIN_STEPS = 60000

STOCKS = [
    # (display_name, yf_ticker,  model_suffix,  train_ppo, train_xgb)
    ('JPM',    'JPM',    'jpm',    True,  True),
    ('XLE',    'XLE',    'xle',    True,  True),
    ('BRK-A',  'BRK-A',  'brk-a',  True,  True),
    ('BRK-B',  'BRK-B',  'brk-b',  True,  True),
    ('MCD',    'MCD',    'mcd',    True,  True),
    ('LIN',    'LIN',    'lin',    True,  True),
    ('RDDT',   'RDDT',   'rddt',   True,  True),   # IPO Apr 2024 — shorter history
    ('KO',     'KO',     'ko',     True,  True),
    ('DIS',    'DIS',    'dis',    True,  True),
    ('CRM',    'CRM',    'crm',    True,  True),
    ('LULU',   'LULU',   'lulu',   True,  True),
    ('XOP',    'XOP',    'xop',    True,  True),
    ('SMH',    'SMH',    'smh',    True,  True),
    ('SHOP',   'SHOP',   'shop',   True,  True),
    ('COIN',   'COIN',   'coin',   True,  False),  # XGB already exists
    ('RSP',    'RSP',    'rsp',    True,  True),
]

FEATURES = [
    'sma_10','sma_30','sma_50','rsi','macd','macd_signal',
    'bb_upper','bb_lower','obv','volume_ratio',
]

# ── Feature Engineering ──────────────────────────────────────────────────────
def add_features(df_raw):
    df = df_raw.copy()
    if hasattr(df.columns, 'levels'):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()

    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + g / (l + 1e-10))

    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    df['bb_mid']   = df['close'].rolling(20).mean()
    df['bb_std']   = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']

    df['obv']          = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['vol_ma20']     = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-10)

    df['future_return'] = df['close'].pct_change(5).shift(-5)
    df['label']         = (df['future_return'] > 0).astype(int)

    return df.bfill().ffill().dropna()


# ── Trading Environment ───────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df, init_balance=100_000):
        super().__init__()
        self.df   = df.reset_index(drop=True)
        self.init = init_balance
        self.action_space      = spaces.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.bal = self.init
        self.shr = 0
        self.pnl = 0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.idx]
        p = float(r.get('close', 1))
        eq = self.bal + self.shr * p
        return np.array([
            self.shr, self.bal, p,
            float(r.get('sma_10',  p)), float(r.get('sma_30', p)),
            float(r.get('sma_50',  p)), float(r.get('rsi',   50)),
            float(r.get('macd',    0)), float(r.get('macd_signal', 0)),
            float(r.get('bb_upper',p)), float(r.get('bb_lower',  p)),
            float(r.get('volume',  0)),
            self.pnl, eq / self.init, self.bal / self.init,
        ], dtype=np.float32)

    def step(self, action):
        r  = self.df.iloc[self.idx]
        p  = float(r.get('close', 1))
        a  = float(action[0])
        if a > 0.3 and self.bal >= p:
            buy       = int(self.bal // p)
            self.shr += buy
            self.bal -= buy * p
        elif a < -0.3 and self.shr > 0:
            self.bal += self.shr * p
            self.shr  = 0
        self.idx += 1
        done     = self.idx >= len(self.df) - 1
        self.pnl = self.bal + self.shr * p - self.init
        return self._obs(), self.pnl / self.init * 0.01, done, False, {}


# ── XGBoost backtest ─────────────────────────────────────────────────────────
def backtest_xgb(model, df_test):
    cap = 100_000.0; shr = 0; trades = 0; wins = 0; bp = 0
    for _, row in df_test.iterrows():
        price = float(row['close'])
        pred  = model.predict(row[FEATURES].values.reshape(1, -1))[0]
        if pred == 1 and shr == 0:
            shr = int(cap // price); cap -= shr * price; bp = price; trades += 1
        elif pred == 0 and shr > 0:
            cap += shr * price
            if price > bp: wins += 1
            shr = 0
    if shr > 0:
        cap += shr * float(df_test.iloc[-1]['close'])
    roi = (cap - 100_000) / 100_000 * 100
    wr  = wins / trades * 100 if trades else 0
    return roi, trades, wr


# ── PPO backtest ──────────────────────────────────────────────────────────────
def backtest_ppo(model, df_test):
    env = TradingEnv(df_test)
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(act)
    final = env.bal + env.shr * float(df_test.iloc[-1]['close'])
    return (final - 100_000) / 100_000 * 100


# ── Main ──────────────────────────────────────────────────────────────────────
results = []
print(f"\n{'='*70}")
print(f"  Western Batch 4 Training  |  {len(STOCKS)} stocks  |  {END_DATE}")
print(f"{'='*70}\n")

for name, ticker, suffix, do_ppo, do_xgb in STOCKS:
    print(f"\n{'─'*60}")
    print(f"  [{name}]  ticker={ticker}  suffix={suffix}")
    print(f"{'─'*60}")

    # Download
    start = '2018-01-01' if name != 'RDDT' else '2024-03-01'
    df_raw = yf.download(ticker, start=start, end=END_DATE,
                         progress=False, auto_adjust=True)
    if df_raw.empty:
        print(f"  ⚠  No data for {ticker} — skip")
        continue

    df = add_features(df_raw)
    if len(df) < 60:
        print(f"  ⚠  Too little data ({len(df)} rows) — skip")
        continue

    split = int(len(df) * 0.8)
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_te = df.iloc[split:].reset_index(drop=True)

    bh_roi = (float(df_te.iloc[-1]['close']) - float(df_te.iloc[0]['close'])) \
             / float(df_te.iloc[0]['close']) * 100

    row = {'name': name, 'ticker': ticker,
           'rows': len(df),
           'test_start': df.index[split].date() if hasattr(df.index[split],'date') else df.index[split],
           'bh_roi': bh_roi,
           'xgb_acc': None, 'xgb_roi': None,
           'ppo_roi': None}

    # XGBoost
    if do_xgb:
        X_tr = df_tr[FEATURES].values;  y_tr = df_tr['label'].values
        X_te = df_te[FEATURES].values;  y_te = df_te['label'].values
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss', verbosity=0,
        )
        xgb.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, xgb.predict(X_te))
        xgb_roi, trades, wr = backtest_xgb(xgb, df_te)
        joblib.dump(xgb, f'xgb_{suffix}_model.pkl')
        row['xgb_acc'] = acc * 100
        row['xgb_roi'] = xgb_roi
        print(f"  XGBoost  acc={acc*100:.1f}%  ROI={xgb_roi:+.1f}%  "
              f"trades={trades}  win={wr:.0f}%")
        print(f"  Saved: xgb_{suffix}_model.pkl")

    # PPO
    if do_ppo:
        env = TradingEnv(df_tr)
        ppo = PPO('MlpPolicy', env, verbose=0,
                  learning_rate=3e-4, n_steps=2048, batch_size=64,
                  n_epochs=10, gamma=0.99, ent_coef=0.01)
        ppo.learn(total_timesteps=TRAIN_STEPS)
        ppo.save(f'ppo_{suffix}_improved')
        ppo_roi = backtest_ppo(ppo, df_te)
        row['ppo_roi'] = ppo_roi
        print(f"  PPO      ROI={ppo_roi:+.1f}%")
        print(f"  Saved: ppo_{suffix}_improved")

    print(f"  Buy&Hold ROI={bh_roi:+.1f}%")
    winner = 'B&H'
    if row['ppo_roi'] and row['bh_roi']:
        if row['ppo_roi'] > row['bh_roi']:
            winner = 'PPO'
    if row['xgb_roi'] and row['bh_roi']:
        if row['xgb_roi'] > row['bh_roi']:
            winner = 'XGB' if winner == 'B&H' else winner

    row['winner'] = winner
    results.append(row)

# ── Summary Table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  {'Stock':<8}  {'Rows':>5}  {'XGB Acc':>8}  {'XGB ROI':>9}  {'PPO ROI':>9}  {'B&H ROI':>9}  {'Winner'}")
print(f"  {'─'*65}")
for r in results:
    xacc = f"{r['xgb_acc']:.1f}%" if r['xgb_acc'] else '  skip'
    xroi = f"{r['xgb_roi']:+.1f}%" if r['xgb_roi'] else '  skip'
    proi = f"{r['ppo_roi']:+.1f}%" if r['ppo_roi'] else '  skip'
    bh   = f"{r['bh_roi']:+.1f}%"
    print(f"  {r['name']:<8}  {r['rows']:>5}  {xacc:>8}  {xroi:>9}  {proi:>9}  {bh:>9}  {r.get('winner','')}")

if results:
    xgb_avgs = [r['xgb_roi'] for r in results if r['xgb_roi'] is not None]
    ppo_avgs = [r['ppo_roi'] for r in results if r['ppo_roi'] is not None]
    bh_avgs  = [r['bh_roi']  for r in results]
    print(f"\n  Average XGB ROI: {sum(xgb_avgs)/len(xgb_avgs):+.2f}%" if xgb_avgs else "")
    print(f"  Average PPO ROI: {sum(ppo_avgs)/len(ppo_avgs):+.2f}%" if ppo_avgs else "")
    print(f"  Average B&H ROI: {sum(bh_avgs)/len(bh_avgs):+.2f}%")
    xwins = sum(1 for r in results if r.get('winner') == 'XGB')
    pwins = sum(1 for r in results if r.get('winner') == 'PPO')
    bwins = sum(1 for r in results if r.get('winner') == 'B&H')
    print(f"\n  XGB wins: {xwins}  PPO wins: {pwins}  B&H wins: {bwins}")

print(f"\nDone at {datetime.now().strftime('%Y-%m-%d %H:%M')}")
