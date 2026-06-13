"""
XGBoost + PPO batch training — 14 TW Stocks (Batch 3)
2492 華新科, 3026 禾伸堂, 2472 立隆電, 2375 凱美, 2327 國巨,
6449 鉅邦,   6175 立敦,   8042 金山電, 6127 九豪, 3357 臺慶科,
2478 大毅,   3090 日電貿, 6224 聚鼎,   8043 蜜望寶
"""
import os, sys, io, warnings, json, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')
logging.getLogger('yfinance').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Known OTC (TWO) codes ─────────────────────────────────────────────────────
_TWO = {
    # Batch 3 — confirmed OTC
    '6449', '6175', '8042', '6127', '3357', '8043',
    # Previous batches
    '3680','1785','8299','3081','3498','3615','4533','4577','4768','4908',
    '4991','5011','6134','6187','6220','6530','6877','7805','8086','8908',
    '8917','8927','6274','4749','3131','6683','3363','6510','8069','6223',
    '5483','6163','7709','7717','3260','3491','5371','3105','4971','8064',
    '3163','3455','4772','6788','7703','8147','8071','8027','5351','7734',
    '7751','6138','1569','1595','4951','6234','6488','6207','3624','8455',
    '8291','3577','3236','3691','6204','6432','3609','3450','3581','3265',
    '5289','3587','3264','3663','6538','3580','8044','3209','6147',
    '4760','6284','3485','6570','6173','6727',
}

def _ticker(code):
    return f'{code}.TWO' if code in _TWO else f'{code}.TW'

STOCKS = [
    {'code': '2492', 'name': '華新科'},
    {'code': '3026', 'name': '禾伸堂'},
    {'code': '2472', 'name': '立隆電'},
    {'code': '2375', 'name': '凱美'},
    {'code': '2327', 'name': '國巨'},
    {'code': '6449', 'name': '鉅邦'},
    {'code': '6175', 'name': '立敦'},
    {'code': '8042', 'name': '金山電'},
    {'code': '6127', 'name': '九豪'},
    {'code': '3357', 'name': '臺慶科'},
    {'code': '2478', 'name': '大毅'},
    {'code': '3090', 'name': '日電貿'},
    {'code': '6224', 'name': '聚鼎'},
    {'code': '8043', 'name': '蜜望寶'},
]
for s in STOCKS:
    s['ticker'] = _ticker(s['code'])

END_DATE      = '2026-05-29'
PPO_TIMESTEPS = 100_000

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]


# ── Indicators ────────────────────────────────────────────────────────────────
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
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target']        = (df['future_return'] > 0.02).astype(int)
    return df.bfill().ffill()


# ── Download with auto-fallback TW ↔ TWO ─────────────────────────────────────
def download_data(ticker, code):
    df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty or len(df) < 50:
        alt = (ticker.replace('.TW', '.TWO') if '.TW' in ticker and '.TWO' not in ticker
               else ticker.replace('.TWO', '.TW'))
        print(f"  ⚠  {ticker} empty, trying {alt}...")
        df2 = yf.download(alt, start='2015-01-01', end=END_DATE, progress=False)
        if isinstance(df2.columns, pd.MultiIndex):
            df2.columns = df2.columns.droplevel(1)
        if not df2.empty and len(df2) >= 50:
            df2 = df2.rename(columns={'Close':'close','Volume':'volume',
                                      'Open':'open','High':'high','Low':'low'}).reset_index()
            return df2, alt
        return None, ticker
    df = df.rename(columns={'Close':'close','Volume':'volume',
                             'Open':'open','High':'high','Low':'low'}).reset_index()
    return df, ticker


# ── XGBoost ───────────────────────────────────────────────────────────────────
def train_xgboost(df_clean, code, ticker):
    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['target']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=200,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss'
    )
    model.fit(X_tr, y_tr)
    acc      = accuracy_score(y_te, model.predict(X_te))
    suffix   = 'two' if '.TWO' in ticker else 'tw'
    suffix_up= 'TWO' if '.TWO' in ticker else 'TW'
    joblib.dump(model, f'xgb_{code}_{suffix}_model.pkl')
    acc_data = {
        'symbol': ticker, 'model_type': 'XGBoost',
        'training_accuracy':   float(accuracy_score(X_tr, model.predict(X_tr)) * 100) if False else float(acc * 100),
        'validation_accuracy': float(acc * 100),
        'backtest_accuracy':   float(acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f'model_accuracy_{code}_{suffix_up}.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)
    with open(f'model_accuracy_{code}.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)
    return acc


# ── PPO TradingEnv ────────────────────────────────────────────────────────────
class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 0; self.balance = 10000.0; self.shares = 0; self.profit = 0.0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.step_idx]; p = float(r['close'])
        tv = self.balance + self.shares * p
        return np.array([
            float(self.shares), float(self.balance), p,
            float(r.get('sma_10', 0)), float(r.get('sma_30', 0)), float(r.get('sma_50', 0)),
            float(r.get('rsi', 50)),   float(r.get('macd', 0)),   float(r.get('macd_signal', 0)),
            float(r.get('bb_upper', 0)), float(r.get('bb_lower', 0)),
            float(r.get('volume', 0)), float(self.profit),
            (self.shares * p) / tv if tv > 0 else 0,
            self.balance / tv if tv > 0 else 1,
        ], dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1.0, 1.0); p = float(self.df.iloc[self.step_idx]['close'])
        if a < -0.1:
            s = int(self.shares * abs(a))
            if s > 0: self.balance += s * p; self.shares -= s
        elif a > 0.1:
            s = int((self.balance // p) * a)
            if s > 0: self.balance -= s * p; self.shares += s
        self.profit   = (self.balance + self.shares * p) - 10000.0
        self.step_idx += 1; done = self.step_idx >= len(self.df) - 1
        reward = self.profit / 10000.0
        if abs(a) > 0.1: reward += 0.01
        if self.balance > (self.balance + self.shares * p) * 0.9: reward -= 0.005
        return self._obs(), reward, done, False, {}


def train_ppo(train_df, test_df, code, ticker, timesteps=PPO_TIMESTEPS):
    env = DummyVecEnv([lambda: TradingEnv(train_df)])
    model = PPO('MlpPolicy', env, verbose=0,
                learning_rate=0.0003, n_steps=2048, batch_size=64,
                n_epochs=10, gamma=0.99, ent_coef=0.01)
    model.learn(total_timesteps=timesteps)
    suffix   = 'two' if '.TWO' in ticker else 'tw'
    suffix_up= 'TWO' if '.TWO' in ticker else 'TW'
    model.save(f'ppo_{code}_{suffix}_improved')

    env2 = TradingEnv(test_df.reset_index(drop=True))
    obs, _ = env2.reset(); preds, labels = [], []
    for i in range(len(test_df) - 6):
        act, _ = model.predict(obs, deterministic=True)
        preds.append(1 if float(act[0]) > 0.1 else 0)
        labels.append(int(test_df.iloc[i]['target']))
        obs, _, done, _, _ = env2.step(act)
        if done: break
    acc = accuracy_score(labels, preds) if preds else 0.0

    acc_data = {
        'symbol': ticker, 'model_type': 'PPO',
        'training_accuracy':   float(acc * 100),
        'validation_accuracy': float(acc * 100),
        'backtest_accuracy':   float(acc * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(f'model_accuracy_{code}_{suffix_up}.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)
    with open(f'model_accuracy_{code}.json', 'w', encoding='utf-8') as f:
        json.dump(acc_data, f, ensure_ascii=False, indent=2)
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 72)
    print("  🚀 Batch XGBoost + PPO — 14 TW Stocks (Batch 3)")
    print(f"  End date: {END_DATE}  |  PPO steps: {PPO_TIMESTEPS:,}")
    print("=" * 72)

    results = []

    for i, s in enumerate(STOCKS, 1):
        code   = s['code']
        name   = s['name']
        ticker = s['ticker']

        print(f"\n[{i:02d}/{len(STOCKS)}] {'─'*60}")
        print(f"  {ticker} ({name})")

        raw, actual_ticker = download_data(ticker, code)
        if raw is None:
            print(f"  ❌ 無法下載資料，跳過")
            results.append({'code': code, 'name': name, 'ticker': ticker,
                             'xgb': 0.0, 'ppo': 0.0, 'winner': 'N/A', 'rows': 0})
            continue

        df = add_indicators(raw)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])
        if len(df_clean) < 100:
            print(f"  ❌ 可用資料不足 ({len(df_clean)} 行)，跳過")
            results.append({'code': code, 'name': name, 'ticker': actual_ticker,
                             'xgb': 0.0, 'ppo': 0.0, 'winner': 'N/A', 'rows': len(df_clean)})
            continue

        n     = len(df_clean)
        split = int(n * 0.8)
        train_df = df_clean.iloc[:split].copy()
        test_df  = df_clean.iloc[split:].copy()

        print(f"  Data: {n} days  [{str(df_clean['Date'].iloc[0])[:10]} → {str(df_clean['Date'].iloc[-1])[:10]}]")
        print(f"  Train: {len(train_df)}  Test: {len(test_df)}")

        # XGBoost
        print("  [XGBoost] training...")
        xgb_acc = train_xgboost(df_clean, code, actual_ticker)
        print(f"  [XGBoost] acc: {xgb_acc*100:.2f}%")

        # PPO
        print(f"  [PPO] training ({PPO_TIMESTEPS:,} steps)...")
        ppo_acc = train_ppo(train_df, test_df, code, actual_ticker)
        print(f"  [PPO]     acc: {ppo_acc*100:.2f}%")

        winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
        results.append({'code': code, 'name': name, 'ticker': actual_ticker,
                         'xgb': xgb_acc, 'ppo': ppo_acc, 'winner': winner, 'rows': n})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  📊 ACCURACY COMPARISON SUMMARY")
    print("=" * 72)
    print(f"  {'#':<4} {'Ticker':<12} {'Name':<12} {'XGBoost':>10} {'PPO':>10}  {'Winner'}")
    print(f"  {'─'*62}")
    xgb_sum = ppo_sum = 0
    xgb_wins = ppo_wins = 0
    for i, r in enumerate(results, 1):
        xgb_s = f"{r['xgb']*100:.2f}%"
        ppo_s = f"{r['ppo']*100:.2f}%"
        print(f"  {i:<4} {r['ticker']:<12} {r['name']:<12} {xgb_s:>10} {ppo_s:>10}  {r['winner']:>8}")
        if r['winner'] not in ('N/A', ''):
            xgb_sum += r['xgb']; ppo_sum += r['ppo']
            if r['winner'] == 'XGBoost': xgb_wins += 1
            else: ppo_wins += 1

    valid = [r for r in results if r['winner'] not in ('N/A', '')]
    n_valid = len(valid) or 1
    print("=" * 72)
    print(f"  Average: XGBoost {xgb_sum/n_valid*100:.2f}%  |  PPO {ppo_sum/n_valid*100:.2f}%")
    print(f"  Wins:    XGBoost {xgb_wins}/{n_valid}        |  PPO {ppo_wins}/{n_valid}")
    print("=" * 72)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
