"""
Retrain XGBoost + PPO for all 234 Taiwan stocks listed in run_all_local_tw_to_excel.py
"""
import os, sys, io, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

warnings.filterwarnings('ignore')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PPO_TIMESTEPS = 100_000

T_STOCKS = {'3449'}

TWO_STOCKS = {
    '3498', '3615', '4533', '4577', '4768', '4908', '4991', '5011',
    '6134', '6187', '6220', '6530', '6877', '7805', '8086', '8908', '8917', '8927',
    '6274', '1785', '4749', '3131', '6683', '3363', '3081', '6510',
    '8069', '6223', '5483', '6163', '7709', '7717',
    '3260', '3491', '5371', '3105', '4971',
    '8064', '3163', '3455',
    '3680', '4772', '6788', '7703', '8147', '8071',
    '8027', '5351', '7734', '7751', '6138',
    '1569', '1595', '4951',
    '6234', '6488', '6207',
    # previously failed as .TW — correct exchange is TWO
    '1815', '3147', '3152', '3236', '3264', '3265', '3577', '3580', '3581', '3587',
    '3609', '3624', '3663', '3691', '5289', '6146', '6147', '6204', '6432', '6538',
    '6643', '6980', '8044', '8291', '8299', '8455',
}

ALL_CODES = [
    '2330', '2317', '6515', '2408', '2308', '2313', '2454', '2485', '2337', '2344',
    '2367', '3481', '2603', '6770', '3665', '3017', '3711', '3037', '2327', '2382',
    '3443', '2383', '6442', '3661', '6669', '6683', '3231', '2303', '2368', '2345',
    '1303', '2360', '2449', '6443', '4989', '6285', '3715', '3563', '3653', '2891',
    '6239', '3533', '8069', '6223', '3363', '3449', '5483', '6163', '7709', '7717',
    '3260', '3491', '5371', '3105', '4971', '6187', '3615', '4577', '4768', '4991',
    '6220', '6877', '8927', '1519', '6805', '6789', '8021', '3006', '6830', '2357',
    '3030', '2409', '2376', '8210', '6446', '1326', '8046', '1605', '1301', '2059',
    '6781', '2884', '6271', '2002', '6526', '3138', '8150', '1101', '2890', '3044',
    '4967', '2451', '8110', '2385', '4938', '3576', '2634', '1514', '4722', '6472',
    '8131', '6230', '2363', '6209', '3135', '6269', '8438', '4564', '4540', '8499',
    '6477', '3004', '4746', '8222', '3022', '6668', '2314', '1314', '8908', '9931',
    '8917', '6505', '9918', '2412', '6274', '8112', '2049', '1785', '6531', '2395',
    '4749', '3131', '3081', '6510', '3535', '8064', '3163', '3455', '2426', '3583',
    '8028', '3680', '4772', '6788', '7703', '8147', '2404', '6196', '6605', '6139',
    '8071', '1560', '6438', '6449', '8027', '5351', '4720', '6176', '3380', '6672',
    '6213', '7734', '7751', '2486', '6138', '8103', '1569', '1595', '6108', '4951',
    '1727', '6234', '6488', '6207', '6937', '3189', '6147', '3624', '8455', '6924',
    '3577', '8374', '2359', '3236', '6204', '3024', '6432', '3609', '8299', '3581',
    '8291', '3265', '3714', '2340', '1773', '5215', '3587', '3691', '3264', '6257',
    '3055', '5289', '7610', '7788', '2481', '3023', '3663', '6538', '3580', '2355',
    '8044', '3147', '6980', '2428', '1597', '2455', '3026', '2851', '6146', '6706',
    '3152', '6643', '6533', '1717', '3008', '3028', '1795', '1815', '2379', '3013',
    '3034', '3413',
]


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
total = len(ALL_CODES)
print("=" * 70)
print(f"Retrain XGBoost + PPO: {total} Taiwan stocks")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

results = []
start_time = time.time()

for i, code in enumerate(ALL_CODES, 1):
    if code in T_STOCKS:
        print(f"\n[{i}/{total}] {code}.T — skipped (Tokyo exchange)")
        results.append({'code': code, 'exchange': 'T', 'xgb_acc': None, 'ppo': None, 'status': 'SKIPPED'})
        continue

    exchange = 'TWO' if code in TWO_STOCKS else 'TW'
    ticker = f"{code}.{exchange}"

    elapsed = time.time() - start_time
    avg_per = elapsed / i if i > 1 else 0
    eta_sec = avg_per * (total - i)
    eta_str = f"ETA {int(eta_sec//60)}m{int(eta_sec%60):02d}s" if avg_per > 0 else "ETA --"
    print(f"\n[{i}/{total}] {ticker}  ({eta_str})")

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

    try:
        xgb_acc = train_xgboost(df, code, exchange)
    except Exception as e:
        print(f"  ❌ XGBoost failed: {e}")
        xgb_acc = None

    try:
        ppo_name = train_ppo(df, code, exchange)
    except Exception as e:
        print(f"  ❌ PPO failed: {e}")
        ppo_name = None

    results.append({'code': code, 'exchange': exchange, 'xgb_acc': xgb_acc,
                    'ppo': ppo_name, 'status': 'OK'})

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Code':<6} {'Exch':<5} {'XGBoost Acc':<14} {'PPO Model':<32} {'Status'}")
print("-" * 70)
for r in results:
    xgb_str = f"{r['xgb_acc']:.1f}%" if r['xgb_acc'] is not None else "—"
    ppo_str  = r['ppo'] if r['ppo'] else "—"
    print(f"{r['code']:<6} {r['exchange']:<5} {xgb_str:<14} {ppo_str:<32} {r['status']}")
print("=" * 70)

ok  = sum(1 for r in results if r['status'] == 'OK')
skp = sum(1 for r in results if r['status'] == 'SKIPPED')
nd  = sum(1 for r in results if r['status'] == 'NO DATA')
err = sum(1 for r in results if r['status'] not in ('OK', 'SKIPPED', 'NO DATA'))
print(f"OK: {ok}  |  No data: {nd}  |  Errors: {err}  |  Skipped: {skp}")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
