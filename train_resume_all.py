"""Resume all interrupted training — skips stocks with existing XGBoost+PPO models"""
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

# All stocks needing training (41 total after crash)
STOCKS = [
    ('1773', '勝一',       'TW'),
    ('2308', '台達電',     'TW'),
    ('2330', '台積電',     'TW'),
    ('2340', '台亞',       'TW'),
    ('2359', '所羅門',     'TW'),
    ('2368', '金像電',     'TW'),
    ('2383', '台光電',     'TW'),
    ('2454', '聯發科',     'TW'),
    ('3024', '憶聲',       'TW'),
    ('3055', '蔚華科',     'TWO'),
    ('3209', '全科',       'TWO'),
    ('3236', '千如',       'TWO'),
    ('3264', '欣銓',       'TW'),
    ('3265', '台星科',     'TWO'),
    ('3450', '聯鈞',       'TWO'),
    ('3498', '陽程',       'TWO'),
    ('3535', '晶彩科',     'TWO'),
    ('3577', '泓格',       'TWO'),
    ('3581', '博磊',       'TWO'),
    ('3587', '閎康',       'TW'),
    ('3609', '三一東林',   'TWO'),
    ('3691', '碩禾',       'TW'),
    ('3714', '富采',       'TW'),
    ('4966', '譜瑞-KY',   'TW'),
    ('4979', '華星光',     'TW'),
    ('5215', '科嘉-KY',   'TWO'),
    ('5289', '宜鼎',       'TW'),
    ('6147', '頎邦',       'TWO'),
    ('6204', '艾華',       'TWO'),
    ('6257', '矽格',       'TWO'),
    ('6432', '今展科',     'TWO'),
    ('6449', '鈺邦',       'TWO'),
    ('6924', '榮惠-KY',   'TW'),
    ('7610', '聯友金屬',   'TWO'),
    ('7788', '兆川精密',   'TWO'),
    ('8028', '昇陽半導體', 'TWO'),
    ('8043', '蜜望實',     'TWO'),
    ('8150', '南茂',       'TW'),
    ('8299', '群聯',       'TW'),
    ('8374', '羅昇',       'TWO'),
    ('8455', '大拓-KY',   'TW'),
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


def train_stock(code, name, sfx, idx, total):
    ticker   = f"{code}.{sfx}"
    sfx_low  = sfx.lower()
    xgb_file = f'xgb_{code}_{sfx_low}_model.pkl'
    ppo_file  = f'ppo_{code}_{sfx_low}_improved.zip'
    need_xgb  = not os.path.exists(xgb_file)
    need_ppo  = not os.path.exists(ppo_file)

    if not need_xgb and not need_ppo:
        print(f"  [{idx}/{total}] {code} {name} — already complete, skipping")
        return None

    print(f"\n{'='*60}")
    print(f"  [{idx}/{total}] {code} {name} ({ticker})")
    print(f"  Need: {'XGBoost ' if need_xgb else ''}{'PPO' if need_ppo else ''}")
    print(f"{'='*60}")

    # Download data (always needed)
    try:
        df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 50:
            print(f"  [SKIP] Only {len(df)} rows"); return None
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                 'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  Data: {len(df)} days")
    except Exception as e:
        print(f"  [ERROR] download: {e}"); return None

    df = build_features(df)
    dc = df.dropna(subset=FEAT + ['target'])
    if len(dc) < 50:
        print(f"  [SKIP] {len(dc)} clean rows"); return None
    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy(); te = dc.iloc[split:].copy()
    print(f"  Train: {len(tr)} | Test: {len(te)}")

    xgb_acc = ppo_acc = None

    # ── XGBoost (if missing) ─────────────────────────────────────────────────
    if need_xgb:
        print("  [XGBoost] Training...")
        try:
            X = dc[FEAT]; y = dc['target']
            Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
            xm = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
                                     min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                     objective='binary:logistic', random_state=42, eval_metric='logloss')
            xm.fit(Xt, yt)
            xgb_acc = accuracy_score(ye, xm.predict(Xe))
            joblib.dump(xm, xgb_file)
            with open(f'model_accuracy_{code}_{sfx}.json', 'w', encoding='utf-8') as f:
                json.dump({'symbol': code, 'ticker': ticker, 'model_type': 'XGBoost',
                           'training_accuracy': round(accuracy_score(yt, xm.predict(Xt))*100, 2),
                           'validation_accuracy': round(xgb_acc*100, 2),
                           'backtest_accuracy':   round(xgb_acc*100, 2),
                           'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2)
            print(f"  [XGBoost] {xgb_acc*100:.2f}%")
        except Exception as e:
            print(f"  [XGBoost ERROR] {e}")
    else:
        print(f"  [XGBoost] already exists — skipped")
        try:
            d = json.load(open(f'model_accuracy_{code}_{sfx}.json', encoding='utf-8'))
            xgb_acc = d.get('backtest_accuracy', 0) / 100
        except: xgb_acc = 0

    # ── PPO (if missing) ─────────────────────────────────────────────────────
    if need_ppo:
        print(f"  [PPO] Training ({TOTAL_TIMESTEPS//1000}k steps)...")
        try:
            env = DummyVecEnv([lambda: TradingEnv(tr)])
            pm = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003,
                     n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
            pm.learn(total_timesteps=TOTAL_TIMESTEPS)
            pm.save(f'ppo_{code}_{sfx_low}_improved')
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
        except Exception as e:
            print(f"  [PPO ERROR] {e}")
    else:
        print(f"  [PPO] already exists — skipped")
        try:
            d = json.load(open(f'model_accuracy_{code}_{sfx}_ppo.json', encoding='utf-8'))
            ppo_acc = d.get('backtest_accuracy', 0) / 100
        except: ppo_acc = 0

    if xgb_acc is None: xgb_acc = 0
    if ppo_acc is None: ppo_acc = 0
    winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
    return {'code': code, 'name': name, 'ticker': ticker,
            'xgb_acc': round(xgb_acc*100,2), 'ppo_acc': round(ppo_acc*100,2), 'winner': winner}


if __name__ == '__main__':
    total = len(STOCKS)
    print("=" * 60)
    print(f"  RESUME TRAINING — {total} remaining stocks")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  (Already complete stocks will be skipped)")
    print("=" * 60)

    results, failed = [], []
    for idx, (code, name, sfx) in enumerate(STOCKS, 1):
        try:
            r = train_stock(code, name, sfx, idx, total)
            if r: results.append(r)
        except Exception as e:
            print(f"  [ERROR] {code}: {e}"); failed.append(f"{code} {name}")

    if results:
        print("\n" + "=" * 68)
        print("  ACCURACY COMPARISON TABLE — Resumed")
        print("=" * 68)
        print(f"  {'Code':<6} {'Name':<10} {'Ticker':<12} {'XGBoost':>9} {'PPO':>7} {'Winner'}")
        print("  " + "-" * 62)
        for r in sorted(results, key=lambda x: max(x['xgb_acc'], x['ppo_acc']), reverse=True):
            best = max(r['xgb_acc'], r['ppo_acc'])
            star = "🌟" if best >= 70 else "✅" if best >= 60 else "⚠️ "
            print(f"  {r['code']:<6} {r['name']:<10} {r['ticker']:<12} "
                  f"{r['xgb_acc']:>8.2f}% {r['ppo_acc']:>6.2f}% {r['winner']} {star}")
        print("  " + "-" * 62)
        print(f"  Trained: {len(results)}  |  Failed: {len(failed)}")
        if failed: print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 68)
