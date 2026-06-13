"""Batch train XGBoost + PPO for 16 Taiwan stocks from image_d15f27.png"""
import os, json, warnings, sys, io, joblib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

STOCKS = [
    ('1326', '1326.TW', '台化'),
    ('2303', '2303.TW', '聯電'),
    ('2308', '2308.TW', '台達電'),
    ('2337', '2337.TW', '旺宏'),
    ('2344', '2344.TW', '華邦電'),
    ('2368', '2368.TW', '金像電'),
    ('2449', '2449.TW', '京元電子'),
    ('2609', '2609.TW', '陽明'),
    ('2882', '2882.TW', '國泰金'),
    ('2890', '2890.TW', '永豐金'),
    ('2892', '2892.TW', '第一金'),
    ('3034', '3034.TW', '聯詠'),
    ('3036', '3036.TW', '文曄'),
    ('3189', '3189.TW', '景碩'),
    ('3231', '3231.TW', '緯創'),
    ('3443', '3443.TW', '創意'),
]

FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']


def build_features(df):
    df = df.copy()
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
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
    df['target']           = (df['close'].shift(-5) / df['close'] - 1 > 0.02).astype(int)
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


def train_one(code, ticker, name):
    print(f"\n{'='*60}")
    print(f"  {code} {name} ({ticker})")
    print(f"{'='*60}")

    # Download
    df_raw = yf.download(ticker, start='2015-01-01', end='2026-06-01', progress=False)
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = df_raw.columns.droplevel(1)
    if df_raw.empty or len(df_raw) < 100:
        print(f"  [SKIP] only {len(df_raw)} rows"); return None
    df_raw = df_raw.rename(columns={'Close':'close','Volume':'volume',
                                    'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"  Data: {len(df_raw)} days")

    df = build_features(df_raw)
    dc = df.dropna(subset=FEAT + ['target'])
    if len(dc) < 80:
        print(f"  [SKIP] only {len(dc)} clean rows"); return None

    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy(); te = dc.iloc[split:].copy()

    # XGBoost
    X = dc[FEAT]; y = dc['target']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
    xm = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
                             min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                             objective='binary:logistic', random_state=42, eval_metric='logloss')
    xm.fit(Xt, yt)
    xgb_acc = accuracy_score(ye, xm.predict(Xe))
    joblib.dump(xm, f'xgb_{code}_tw_model.pkl')

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    acc_xgb = {'symbol': code, 'ticker': ticker, 'model_type': 'XGBoost',
               'training_accuracy':   round(accuracy_score(yt, xm.predict(Xt)) * 100, 2),
               'validation_accuracy': round(xgb_acc * 100, 2),
               'backtest_accuracy':   round(xgb_acc * 100, 2),
               'last_updated': now}
    json.dump(acc_xgb, open(f'model_accuracy_{code}_TW.json',  'w', encoding='utf-8'), indent=2)
    json.dump(acc_xgb, open(f'model_accuracy_{code}.json', 'w', encoding='utf-8'), indent=2)
    print(f"  [XGBoost] {xgb_acc*100:.2f}%")

    # PPO
    env = DummyVecEnv([lambda: TradingEnv(tr)])
    pm = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003,
             n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    pm.learn(total_timesteps=100_000)
    pm.save(f'ppo_{code}_tw_improved')

    env2 = TradingEnv(te.reset_index(drop=True)); obs, _ = env2.reset()
    preds, labels = [], []
    for i in range(len(te) - 6):
        act, _ = pm.predict(obs, deterministic=True)
        preds.append(1 if float(act[0]) > 0.1 else 0)
        labels.append(int(te.iloc[i]['target']))
        obs, _, done, _, _ = env2.step(act)
        if done: break
    ppo_acc = accuracy_score(labels, preds) if preds else 0.0
    json.dump({'symbol': code, 'ticker': ticker, 'model_type': 'PPO',
               'validation_accuracy': round(ppo_acc * 100, 2),
               'backtest_accuracy':   round(ppo_acc * 100, 2),
               'last_updated': now},
              open(f'model_accuracy_{code}_TW_ppo.json', 'w', encoding='utf-8'), indent=2)
    print(f"  [PPO]     {ppo_acc*100:.2f}%")

    winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
    return {'code': code, 'name': name, 'xgb': xgb_acc*100, 'ppo': ppo_acc*100, 'winner': winner}


# ── Main ─────────────────────────────────────────────────────────────────────
results = []
start_time = datetime.now()
print(f"Starting batch training: {len(STOCKS)} stocks  ({start_time.strftime('%Y-%m-%d %H:%M')})")

for code, ticker, name in STOCKS:
    r = train_one(code, ticker, name)
    if r:
        results.append(r)

# ── Accuracy comparison table ─────────────────────────────────────────────────
elapsed = (datetime.now() - start_time).seconds // 60
print(f"\n{'='*60}")
print(f"  ACCURACY COMPARISON TABLE  ({len(results)}/{len(STOCKS)} completed, {elapsed} min)")
print(f"{'='*60}")
print(f"  {'Code':<8} {'Name':<10} {'XGBoost':>9} {'PPO':>8} {'Winner':<10}")
print(f"  {'-'*7} {'-'*10} {'-'*9} {'-'*8} {'-'*10}")
for r in results:
    flag = '🏆' if r['winner'] == 'XGBoost' else '  '
    flag2 = '🏆' if r['winner'] == 'PPO' else '  '
    print(f"  {r['code']:<8} {r['name']:<10} {r['xgb']:>8.2f}%{flag} {r['ppo']:>7.2f}%{flag2} {r['winner']}")

if results:
    avg_xgb = sum(r['xgb'] for r in results) / len(results)
    avg_ppo = sum(r['ppo'] for r in results) / len(results)
    xgb_wins = sum(1 for r in results if r['winner'] == 'XGBoost')
    ppo_wins = len(results) - xgb_wins
    print(f"  {'-'*7} {'-'*10} {'-'*9} {'-'*8} {'-'*10}")
    print(f"  {'AVERAGE':<8} {'':10} {avg_xgb:>8.2f}%   {avg_ppo:>7.2f}%")
    print(f"  Wins:                    XGBoost={xgb_wins}   PPO={ppo_wins}")
print(f"{'='*60}")
