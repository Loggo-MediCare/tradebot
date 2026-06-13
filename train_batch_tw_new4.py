"""Batch train XGBoost + PPO — TW stocks batch 4 (28 stocks)"""
import os, json, warnings, sys, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

TWO_STOCKS = {'3498','3615','4533','4577','4768','4908','4991','5011',
              '6134','6187','6220','6530','6877','7805','8086','8908','8917','8927',
              '6274','1785','4749','3131','6683','3363','3081','6510',
              '8069','6223','5483','6163','7709','7717','3260','3491','5371','3105','4971',
              '8064','3163','3455','3680','4772','6788','7703','8147','8071',
              '8027','5351','7734','7751','6138','1569','1595','4951',
              '6234','6488','6207',
              '3624','8455','8291','6924','3577','3055','8374','7610',
              '5215','2359','3236','3691','8043','7788','6204','3024',
              '3209','6432','3609','6449',
              '6257','3450','4979','2426','3581','3163','3265','3535','3498'}

def get_ticker(code):
    return f"{code}.TWO" if code in TWO_STOCKS else f"{code}.TW"

STOCKS = [
    ('2303', '聯電',       '2015-01-01'),
    ('6138', '茂達',       '2015-01-01'),
    ('3264', '欣銓',       '2015-01-01'),
    ('6147', '頎邦',       '2015-01-01'),
    ('1560', '中砂',       '2015-01-01'),
    ('8150', '南茂',       '2015-01-01'),
    ('6257', '矽格',       '2015-01-01'),
    ('8064', '東捷',       '2015-01-01'),
    ('2454', '聯發科',     '2015-01-01'),
    ('3455', '由田',       '2015-01-01'),
    ('3450', '聯鈞',       '2015-01-01'),
    ('4979', '華星光',     '2015-01-01'),
    ('2426', '鼎元',       '2015-01-01'),
    ('4966', '譜瑞-KY',   '2015-01-01'),
    ('3581', '博磊',       '2015-01-01'),
    ('3163', 'Browave',    '2015-01-01'),
    ('3265', '台星科',     '2015-01-01'),
    ('2330', '台積電',     '2015-01-01'),
    ('3535', '晶彩科',     '2015-01-01'),
    ('3714', '富采',       '2015-01-01'),
    ('2340', '台亞',       '2015-01-01'),
    ('3587', '閎康',       '2015-01-01'),
    ('2344', '華邦電',     '2015-01-01'),
    ('2408', '南亞科',     '2015-01-01'),
    ('8028', '昇陽半導體', '2015-01-01'),
    ('3498', '陽程',       '2015-01-01'),
    ('3680', '家登',       '2015-01-01'),
    ('1773', '勝一',       '2015-01-01'),
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
    df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_m']        = df['close'].rolling(20).mean()
    df['bb_s']        = df['close'].rolling(20).std()
    df['bb_u']        = df['bb_m'] + 2 * df['bb_s']
    df['bb_l']        = df['bb_m'] - 2 * df['bb_s']
    df['bb_position'] = ((df['close'] - df['bb_l']) / (df['bb_u'] - df['bb_l']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K']           = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D']           = df['K'].rolling(3).mean()
    df['obv']         = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20']    = df['obv'].rolling(20).mean()
    df['volatility']  = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr']              = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
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
        return np.array([float(self.sh), float(self.bal), p,
                         float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),
                         float(r.get('rsi',50)),   float(r.get('macd',0)),   float(r.get('macd_signal',0)),
                         float(r.get('bb_u',0)),   float(r.get('bb_l',0)),   float(r.get('volume',0)),
                         float(self.profit),
                         (self.sh * p) / tv if tv > 0 else 0,
                         self.bal / tv if tv > 0 else 1], dtype=np.float32)
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
        return self._obs(), self.profit / 10_000.0 + (0.01 if abs(a) > 0.1 else 0), done, False, {}


def train_stock(code, name, start_date, idx):
    ticker = get_ticker(code)
    sfx    = 'TWO' if ticker.endswith('.TWO') else 'TW'
    print(f"\n{'='*60}")
    print(f"  [{idx}/{len(STOCKS)}] {code} {name} ({ticker})")
    print(f"{'='*60}")
    try:
        df = yf.download(ticker, start=start_date, end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 50:
            print(f"  [SKIP] Only {len(df)} rows"); return None
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                 'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  Data: {len(df)} days")
    except Exception as e:
        print(f"  [ERROR] {e}"); return None

    df = build_features(df)
    dc = df.dropna(subset=FEAT + ['target'])
    if len(dc) < 50:
        print(f"  [SKIP] {len(dc)} clean rows"); return None
    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy(); te = dc.iloc[split:].copy()
    print(f"  Train: {len(tr)} | Test: {len(te)}")

    # XGBoost
    print("  [XGBoost] Training...")
    X = dc[FEAT]; y = dc['target']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
    xm = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
                             min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                             objective='binary:logistic', random_state=42, eval_metric='logloss')
    xm.fit(Xt, yt)
    xgb_acc = accuracy_score(ye, xm.predict(Xe))
    joblib.dump(xm, f'xgb_{code}_{sfx.lower()}_model.pkl')
    with open(f'model_accuracy_{code}_{sfx}.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': code, 'ticker': ticker, 'model_type': 'XGBoost',
                   'training_accuracy': round(accuracy_score(yt, xm.predict(Xt))*100, 2),
                   'validation_accuracy': round(xgb_acc*100, 2),
                   'backtest_accuracy':   round(xgb_acc*100, 2),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f"  [XGBoost] {xgb_acc*100:.2f}%")

    # PPO
    print(f"  [PPO] Training ({TOTAL_TIMESTEPS//1000}k steps)...")
    env = DummyVecEnv([lambda: TradingEnv(tr)])
    pm = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003,
             n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    pm.learn(total_timesteps=TOTAL_TIMESTEPS)
    pm.save(f'ppo_{code}_{sfx.lower()}_improved')
    env2 = TradingEnv(te.reset_index(drop=True)); obs, _ = env2.reset()
    preds, labels = [], []
    for i in range(len(te) - 6):
        act, _ = pm.predict(obs, deterministic=True)
        preds.append(1 if float(act[0]) > 0.1 else 0)
        labels.append(int(te.iloc[i]['target']))
        obs, _, done, _, _ = env2.step(act)
        if done: break
    ppo_acc = accuracy_score(labels, preds) if preds else 0.0
    with open(f'model_accuracy_{code}_{sfx}_ppo.json', 'w', encoding='utf-8') as f:
        json.dump({'symbol': code, 'ticker': ticker, 'model_type': 'PPO',
                   'validation_accuracy': round(ppo_acc*100, 2),
                   'backtest_accuracy':   round(ppo_acc*100, 2),
                   'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
    print(f"  [PPO]     {ppo_acc*100:.2f}%  (buys: {sum(preds)}/{len(preds)})")

    winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
    return {'code': code, 'name': name, 'ticker': ticker,
            'xgb_acc': round(xgb_acc*100,2), 'ppo_acc': round(ppo_acc*100,2), 'winner': winner}


if __name__ == '__main__':
    print("=" * 60)
    print(f"  BATCH 4 — {len(STOCKS)} TW Stocks (XGBoost + PPO)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    results, failed = [], []
    for idx, (code, name, start) in enumerate(STOCKS, 1):
        try:
            r = train_stock(code, name, start, idx)
            if r: results.append(r)
            else: failed.append(f"{code} {name}")
        except Exception as e:
            print(f"  [ERROR] {code}: {e}"); failed.append(f"{code} {name}")

    print("\n" + "=" * 66)
    print("  ACCURACY COMPARISON TABLE — Batch 4")
    print("=" * 66)
    print(f"  {'Code':<6} {'Name':<10} {'Ticker':<12} {'XGBoost':>9} {'PPO':>7} {'Winner'}")
    print("  " + "-" * 60)
    for r in sorted(results, key=lambda x: max(x['xgb_acc'], x['ppo_acc']), reverse=True):
        best = max(r['xgb_acc'], r['ppo_acc'])
        star = "🌟" if best >= 70 else "✅" if best >= 60 else "⚠️ "
        print(f"  {r['code']:<6} {r['name']:<10} {r['ticker']:<12} "
              f"{r['xgb_acc']:>8.2f}% {r['ppo_acc']:>6.2f}% {r['winner']} {star}")
    print("  " + "-" * 60)
    print(f"  Trained: {len(results)}  |  Failed: {len(failed)}")
    if failed: print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)
