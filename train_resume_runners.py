"""Resume training — fills missing XGBoost/PPO for all stocks in both runners"""
import os, json, re, warnings, sys, io, shutil
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

# ── Exchange mapping ──────────────────────────────────────────────────────────
TWO = {'3498','3615','4533','4577','4768','4908','4991','5011','6134','6187',
       '6220','6530','6877','7805','8086','8908','8917','8927','6274','1785',
       '4749','3131','6683','3363','3081','6510','8069','6223','5483','6163',
       '7709','7717','3260','3491','5371','3105','4971','8064','3163','3455',
       '3680','4772','6788','7703','8147','8071','8027','5351','7734','7751',
       '6138','1569','1595','4951','6234','6488','6207','3624','8455','8291',
       '3577','3236','3691','6204','6432','3609','3450','3581','3265',
       '5289','3587','3264','3663','6538','3580','8044','8299','3209','6147'}
T = {'3449'}

def get_ticker(code):
    if code in T:   return f'{code}.T',   'T'
    if code in TWO: return f'{code}.TWO', 'TWO'
    # US stocks (uppercase, no suffix)
    if code.upper() == code or code.lower() not in [c.lower() for c in TWO]:
        try:
            import yfinance as yf
            df = yf.download(code, period='5d', progress=False)
            if not df.empty: return code, 'US'
        except: pass
    return f'{code}.TW', 'TW'

# ── Build list from runners ───────────────────────────────────────────────────
tw_runner = open('run_all_local_tw_to_excel.py',  encoding='utf-8').read()
us_runner = open('run_all_western_to_excel.py',   encoding='utf-8').read()
tw_codes  = set(re.findall(r"get_trading_signal_([^']+)\.py'", tw_runner))
us_codes  = set(re.findall(r"get_trading_signal_([^']+)\.py'", us_runner))

STOCKS = []
for code in sorted(tw_codes | us_codes):
    has_xgb = any(f.startswith(f'xgb_{code}_') and f.endswith('.pkl') for f in os.listdir('.'))
    has_ppo = any(f.startswith(f'ppo_{code}_') and f.endswith('.zip') for f in os.listdir('.'))
    if not has_xgb or not has_ppo:
        is_us = code.upper() in {c.upper() for c in us_codes}
        STOCKS.append((code, not has_xgb, not has_ppo, is_us))

print(f"Stocks needing training: {len(STOCKS)}")
print(f"  Need XGBoost: {sum(1 for _,x,_,_ in STOCKS if x)}")
print(f"  Need PPO:     {sum(1 for _,_,p,_ in STOCKS if p)}")

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


def train(code, need_xgb, need_ppo, is_us, idx, total):
    # Determine ticker
    if is_us:
        ticker = code.upper(); sfx = 'US'
        # HK stocks
        if code in ('01810','02202'): ticker = f'{code}.HK'; sfx = 'HK'
    else:
        if code in T:   ticker = f'{code}.T';   sfx = 'T'
        elif code in TWO: ticker = f'{code}.TWO'; sfx = 'TWO'
        else:           ticker = f'{code}.TW';  sfx = 'TW'

    sfx_low = sfx.lower()
    xgb_file = f'xgb_{code}_{sfx_low}_model.pkl' if sfx != 'US' else f'xgb_{code.lower()}_model.pkl'
    ppo_file  = f'ppo_{code}_{sfx_low}_improved'  if sfx != 'US' else f'ppo_{code.lower()}_improved'

    print(f"\n[{idx}/{total}] {code} ({ticker})  need={'XGB ' if need_xgb else ''}{'PPO' if need_ppo else ''}")

    try:
        df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if df.empty or len(df) < 50:
            print(f"  [SKIP] {len(df)} rows"); return None
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                 'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  Data: {len(df)} days")
    except Exception as e:
        print(f"  [ERROR] {e}"); return None

    df = build_features(df)
    dc = df.dropna(subset=FEAT + ['target'])
    if len(dc) < 50: print(f"  [SKIP] {len(dc)} clean rows"); return None
    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy(); te = dc.iloc[split:].copy()

    xgb_acc = ppo_acc = 0.0

    if need_xgb:
        X = dc[FEAT]; y = dc['target']
        Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
        xm = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
                                 min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                 objective='binary:logistic', random_state=42, eval_metric='logloss')
        xm.fit(Xt, yt)
        xgb_acc = accuracy_score(ye, xm.predict(Xe))
        joblib.dump(xm, xgb_file)
        acc_data = {'symbol': code, 'ticker': ticker, 'model_type': 'XGBoost',
                    'training_accuracy': round(accuracy_score(yt, xm.predict(Xt))*100,2),
                    'validation_accuracy': round(xgb_acc*100,2),
                    'backtest_accuracy': round(xgb_acc*100,2),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        json_name = f'model_accuracy_{code}_{sfx}.json' if sfx != 'US' else f'model_accuracy_{code.upper()}.json'
        json.dump(acc_data, open(json_name,'w',encoding='utf-8'), indent=2)
        # copy to plain name for display
        plain = f'model_accuracy_{code}.json'
        if not os.path.exists(plain): shutil.copy(json_name, plain)
        print(f"  [XGBoost] {xgb_acc*100:.2f}%")

    if need_ppo:
        env = DummyVecEnv([lambda: TradingEnv(tr)])
        pm = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003,
                 n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
        pm.learn(total_timesteps=TOTAL_TIMESTEPS)
        pm.save(ppo_file)
        env2 = TradingEnv(te.reset_index(drop=True)); obs,_=env2.reset()
        preds,labels=[],[]
        for i in range(len(te)-6):
            act,_=pm.predict(obs,deterministic=True)
            preds.append(1 if float(act[0])>0.1 else 0)
            labels.append(int(te.iloc[i]['target']))
            obs,_,done,_,_=env2.step(act)
            if done: break
        ppo_acc = accuracy_score(labels,preds) if preds else 0.0
        json_ppo = f'model_accuracy_{code}_{sfx}_ppo.json' if sfx != 'US' else f'model_accuracy_{code.upper()}_ppo.json'
        json.dump({'symbol':code,'ticker':ticker,'model_type':'PPO',
                   'validation_accuracy':round(ppo_acc*100,2),'backtest_accuracy':round(ppo_acc*100,2),
                   'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                  open(json_ppo,'w',encoding='utf-8'),indent=2)
        print(f"  [PPO]     {ppo_acc*100:.2f}%")

    winner = 'XGBoost' if xgb_acc >= ppo_acc else 'PPO'
    return {'code':code,'ticker':ticker,'xgb':round(xgb_acc*100,2),'ppo':round(ppo_acc*100,2),'winner':winner}


if __name__ == '__main__':
    print("="*60)
    print(f"  RESUME TRAINING — {len(STOCKS)} stocks")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    results, failed = [], []
    total = len(STOCKS)
    for idx, (code, need_xgb, need_ppo, is_us) in enumerate(STOCKS, 1):
        try:
            r = train(code, need_xgb, need_ppo, is_us, idx, total)
            if r: results.append(r)
            else: failed.append(code)
        except Exception as e:
            print(f"  [ERROR] {code}: {e}"); failed.append(code)

    print("\n"+"="*62)
    print("  RESULTS")
    print("="*62)
    print(f"  {'Code':<10} {'Ticker':<12} {'XGBoost':>9} {'PPO':>7} {'W'}")
    print("  "+"-"*56)
    for r in sorted(results, key=lambda x:max(x['xgb'],x['ppo']), reverse=True):
        best=max(r['xgb'],r['ppo'])
        star="🌟" if best>=70 else ("✅" if best>=60 else "⚠️")
        print(f"  {r['code']:<10} {r['ticker']:<12} {r['xgb']:>8.2f}% {r['ppo']:>6.2f}% {r['winner']} {star}")
    print(f"\n  Done: {len(results)}  Failed: {len(failed)}")
    if failed: print(f"  Failed: {', '.join(failed[:20])}")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*62)
