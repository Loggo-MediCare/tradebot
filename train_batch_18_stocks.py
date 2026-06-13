"""
Batch trainer: XGBoost + PPO for 18 Taiwan stocks
Trains both models for each stock and shows accuracy comparison table.
"""
import os, sys, io, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Stocks with known exchange; 6640/6664 will be auto-detected
STOCKS = [
    ('3450', 'TW'),
    ('3711', 'TW'),
    ('4958', 'TW'),
    ('5483', 'TWO'),
    ('6147', 'TWO'),
    ('6196', 'TW'),
    ('6223', 'TWO'),
    ('6239', 'TW'),
    ('6274', 'TWO'),
    ('6438', 'TW'),
    ('6488', 'TWO'),
    ('6640', None),   # auto-detect
    ('6664', None),   # auto-detect
    ('6669', 'TW'),
    ('7769', 'TW'),
    ('8028', 'TW'),
    ('8358', 'TW'),
    ('8996', 'TW'),
]

PPO_TIMESTEPS = 100_000


def detect_exchange(code):
    for suffix in ['TW', 'TWO']:
        df = yf.download(f"{code}.{suffix}", period='3y', progress=False, auto_adjust=True)
        if not df.empty and len(df) > 100:
            return suffix
    return None


def add_indicators(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd']        = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper']    = bb_mid + 2 * bb_std
    df['bb_lower']    = bb_mid - 2 * bb_std
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility']       = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()


FEATURE_COLS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
    'obv', 'obv_ma20', 'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr', 'price_change_5d', 'price_change_10d',
    'price_change_20d', 'ma50_slope'
]


def train_xgboost(df, code, exchange):
    df = df.copy()
    df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
    df = df.dropna(subset=FEATURE_COLS + ['target'])
    X = df[FEATURE_COLS].values
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss', verbosity=0
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) * 100
    fname = f"xgb_{code}_{exchange.lower()}_model.pkl"
    joblib.dump(model, fname)
    print(f"    XGBoost saved: {fname}  accuracy={acc:.1f}%")
    return acc


class TradingEnv(gym.Env):
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
        return self._obs(), {}

    def _obs(self):
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
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1.0, 1.0)
        cp = float(self.df.iloc[self.current_step]['close'])
        if a < -0.1:
            sell = int(self.shares_held * abs(a))
            if sell > 0:
                self.balance += sell * cp
                self.shares_held -= sell
        elif a > 0.1:
            buy = int(self.balance // cp * a)
            if buy > 0:
                self.balance -= buy * cp
                self.shares_held += buy
        self.total_profit = self.balance + self.shares_held * cp - self.initial_balance
        reward = self.total_profit / self.initial_balance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._obs(), float(reward), done, False, {}


def train_ppo(df, code, exchange):
    env = TradingEnv(df)
    model = PPO('MlpPolicy', env, verbose=0, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    fname = f"ppo_{code}_{exchange.lower()}_improved"
    model.save(fname)
    print(f"    PPO saved: {fname}.zip  ({PPO_TIMESTEPS:,} timesteps)")
    return fname


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"Batch Training: XGBoost + PPO for {len(STOCKS)} Taiwan stocks")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

results = []

for code, exchange in STOCKS:
    print(f"\n[{code}] ", end='', flush=True)

    # Auto-detect exchange if needed
    if exchange is None:
        exchange = detect_exchange(code)
        if exchange is None:
            print(f"❌ Cannot find data for {code}.TW or {code}.TWO — skipping")
            results.append({'code': code, 'exchange': '?', 'xgb_acc': None, 'ppo': None, 'status': 'NO DATA'})
            continue
        print(f"(detected {exchange}) ", end='', flush=True)

    ticker = f"{code}.{exchange}"
    print(ticker)

    # Download data
    try:
        raw = yf.download(ticker, period='3y', progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 100:
            print(f"  ❌ Insufficient data ({len(raw)} rows)")
            results.append({'code': code, 'exchange': exchange, 'xgb_acc': None, 'ppo': None, 'status': 'NO DATA'})
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        raw = raw.reset_index()
        df = add_indicators(raw)
        print(f"  Downloaded {len(df)} rows")
    except Exception as e:
        print(f"  ❌ Download failed: {e}")
        results.append({'code': code, 'exchange': exchange, 'xgb_acc': None, 'ppo': None, 'status': f'ERROR: {e}'})
        continue

    # Train XGBoost
    try:
        xgb_acc = train_xgboost(df, code, exchange)
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")
        xgb_acc = None

    # Train PPO
    try:
        ppo_name = train_ppo(df, code, exchange)
    except Exception as e:
        print(f"  ❌ PPO failed: {e}")
        ppo_name = None

    results.append({'code': code, 'exchange': exchange, 'xgb_acc': xgb_acc,
                    'ppo': ppo_name, 'status': 'OK'})

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Code':<6} {'Exchange':<8} {'XGBoost Acc':<14} {'PPO Model':<30} {'Status'}")
print("-" * 70)
for r in results:
    xgb_str = f"{r['xgb_acc']:.1f}%" if r['xgb_acc'] is not None else "—"
    ppo_str  = r['ppo'] if r['ppo'] else "—"
    print(f"{r['code']:<6} {r['exchange']:<8} {xgb_str:<14} {ppo_str:<30} {r['status']}")
print("=" * 70)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
