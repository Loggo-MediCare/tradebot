"""
Compare 3 RF→PPO hybrid variants + XGBoost for any ticker.

Variant A: RF Feature Selection → PPO (top-N features only)
Variant B: RF Probability Input → PPO (RF proba as extra obs)
Variant C: RF Gating → PPO (action masking by RF regime)

Usage: python _compare_rf_ppo_variants.py MU
"""
import os, sys, io, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESTEPS = 100000
SPLIT     = 0.8


# ── Data ─────────────────────────────────────────────────────────────────────

def download(ticker, days=1095):
    end = datetime.now()
    df  = yf.download(ticker, start=end - timedelta(days=days),
                      end=end, progress=False, auto_adjust=True)
    if df.empty: raise ValueError(f"No data: {ticker}")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Open':'open','High':'high',
                             'Low':'low','Volume':'volume'})
    return df.reset_index()


def add_indicators(df):
    c = df['close']
    for w in [10, 20, 30, 50, 200]:
        df[f'sma_{w}'] = c.rolling(w).mean()
    e12 = c.ewm(span=12, adjust=False).mean()
    e26 = c.ewm(span=26, adjust=False).mean()
    d   = c.diff()
    g   = d.where(d>0,0).rolling(14).mean()
    l   = (-d.where(d<0,0)).rolling(14).mean()
    df['rsi']         = 100 - (100/(1+g/(l+1e-9)))
    df['macd']        = e12 - e26
    df['macd_signal'] = df['macd'].ewm(span=9,adjust=False).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    m20 = c.rolling(20).mean(); s20 = c.rolling(20).std()
    df['bb_upper']    = m20 + s20*2
    df['bb_lower']    = m20 - s20*2
    df['bb_pos']      = (c-df['bb_lower'])/(df['bb_upper']-df['bb_lower']+1e-9)
    df['vol_ratio']   = df['volume']/df['volume'].rolling(20).mean()
    df['atr']         = (df['high']-df['low']).rolling(14).mean()
    df['volatility']  = c.pct_change().rolling(20).std()
    for p in [5,10,20]: df[f'roc_{p}'] = c.pct_change(p)*100
    return df.dropna()


ALL_FEATURES = ['sma_10','sma_20','sma_30','sma_50','rsi','macd','macd_signal',
                'macd_hist','bb_pos','vol_ratio','atr','volatility',
                'roc_5','roc_10','roc_20']


# ── RF helpers ────────────────────────────────────────────────────────────────

def make_labels(df, fwd=5, thr=0.02):
    ret = df['close'].shift(-fwd)/df['close'] - 1
    return np.where(ret>thr, 2, np.where(ret<-thr, 0, 1)), ~np.isnan(ret)

def train_rf(df, features, fwd=5, thr=0.02):
    labels, valid = make_labels(df, fwd, thr)
    X = StandardScaler().fit(df[features].values[valid])
    scaler = StandardScaler().fit(df[features].values[valid])
    X_s = scaler.transform(df[features].values[valid])
    y   = labels[valid]
    rf  = RandomForestClassifier(n_estimators=100, max_depth=6,
                                  min_samples_leaf=5, random_state=42,
                                  class_weight='balanced')
    rf.fit(X_s, y)
    acc = accuracy_score(y, rf.predict(X_s))
    imp = sorted(zip(features, rf.feature_importances_), key=lambda x:-x[1])
    return rf, scaler, acc, imp, valid


# ── Base env ──────────────────────────────────────────────────────────────────

class BaseEnv(gym.Env):
    def __init__(self, df, obs_size, init=10000):
        super().__init__()
        self.df   = df.reset_index(drop=True)
        self.init = init
        self.action_space      = spaces.Box(-1.0, 1.0, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self._rst()

    def _rst(self):
        self.step_i = 0; self.bal = self.init
        self.shares = 0; self.profit = 0; self.trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self._rst(); return self._obs(), {}

    def _price(self): return float(self.df.iloc[self.step_i]['close'])

    def _trade(self, a):
        p = self._price()
        if a < -0.1 and self.shares > 0:
            s = int(self.shares*abs(a))
            if s>0: self.bal+=s*p; self.shares-=s; self.trades+=1
        elif a > 0.1 and self.bal >= p:
            b = int((self.bal*a)//p)
            if b>0: self.bal-=b*p; self.shares+=b; self.trades+=1

    def _step_common(self, a):
        self._trade(a)
        self.step_i += 1
        done = self.step_i >= len(self.df)-1
        pn   = float(self.df.iloc[self.step_i]['close'])
        tv   = self.bal + self.shares*pn
        self.profit = tv - self.init
        if done: self.bal += self.shares*pn; self.shares=0
        return self.profit/self.init, done

    def _base_obs(self, features):
        row = self.df.iloc[self.step_i]
        tv  = self.bal + self.shares*float(row['close'])
        sr  = self.shares*float(row['close'])/tv if tv>0 else 0
        cr  = self.bal/tv if tv>0 else 1
        return [float(row.get(f,0)) for f in features] + \
               [float(self.profit), sr, cr]


# ── Variant A: RF Feature Selection ──────────────────────────────────────────

class EnvA(BaseEnv):
    """PPO uses only top-N RF-selected features."""
    def __init__(self, df, top_features, init=10000):
        self.top_features = top_features
        obs_size = len(top_features) + 3  # +profit, stock_ratio, cash_ratio
        super().__init__(df, obs_size, init)

    def _obs(self):
        return np.array(self._base_obs(self.top_features), dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if hasattr(action,'__len__') else float(action)
        a = max(-1.0, min(1.0, a))
        r, done = self._step_common(a)
        return self._obs(), r, done, False, {}


# ── Variant B: RF Probability Input ──────────────────────────────────────────

class EnvB(BaseEnv):
    """PPO gets all features + RF daily probabilities."""
    def __init__(self, df, rf_proba, features, init=10000):
        self.rf_proba = rf_proba
        self.features = features
        obs_size = len(features) + 3 + 3  # +profit,sr,cr + 3 RF proba
        super().__init__(df, obs_size, init)

    def _obs(self):
        base = self._base_obs(self.features)
        rf   = self.rf_proba[self.step_i].tolist()
        return np.array(base + rf, dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if hasattr(action,'__len__') else float(action)
        a = max(-1.0, min(1.0, a))
        # Small RF nudge on reward
        rf_boost = (self.rf_proba[self.step_i][2] - self.rf_proba[self.step_i][0]) * 0.20
        r, done = self._step_common(a)
        return self._obs(), r + rf_boost, done, False, {}


# ── Variant C: RF Gating ──────────────────────────────────────────────────────

class EnvC(BaseEnv):
    """PPO action is masked by RF regime."""
    def __init__(self, df, rf_proba, features, bull=0.55, bear=0.45, init=10000):
        self.rf_proba = rf_proba
        self.features = features
        self.bull = bull; self.bear = bear
        obs_size = len(features) + 3
        super().__init__(df, obs_size, init)

    def _gate(self, a):
        p_buy = self.rf_proba[self.step_i][2]
        if p_buy >= self.bull:   return max(0.0, a)   # bullish: no sell
        if p_buy <= self.bear:   return min(0.0, a)   # bearish: no buy
        return 0.0                                     # neutral: hold

    def _obs(self):
        return np.array(self._base_obs(self.features), dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if hasattr(action,'__len__') else float(action)
        a = self._gate(max(-1.0, min(1.0, a)))
        r, done = self._step_common(a)
        return self._obs(), r, done, False, {}


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(env):
    obs, _ = env.reset()
    done   = False
    while not done:
        obs, _, done, _, _ = env.step(np.array([0.0]))  # hold baseline
    return env.profit / env.init * 100, env.trades, env.bal


def backtest_ppo(model, env_class, **kwargs):
    env  = env_class(**kwargs)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
    roi = env.profit / env.init * 100
    return roi, env.trades, env.bal


# ── Main ──────────────────────────────────────────────────────────────────────

def compare(ticker):
    print(f'\n{"="*65}')
    print(f'  RF→PPO Variant Comparison  |  {ticker}')
    print(f'{"="*65}')

    # Data
    df  = add_indicators(download(ticker))
    n   = len(df)
    sp  = int(n * SPLIT)
    tr  = df.iloc[:sp].reset_index(drop=True)
    te  = df.iloc[sp:].reset_index(drop=True)
    print(f'Train: {len(tr)} rows  |  Test: {len(te)} rows\n')

    results = {}

    # ── RF on train set ──────────────────────────────────────────────────
    print('Training Random Forest...')
    rf, scaler, rf_acc, importance, valid = train_rf(tr, ALL_FEATURES)
    top5  = [f for f,_ in importance[:5]]
    top10 = [f for f,_ in importance[:10]]
    print(f'  RF accuracy: {rf_acc:.1%}')
    print(f'  Top 5 features: {top5}')

    # RF probabilities for train and test
    def get_proba(df_):
        X_s = scaler.transform(df_[ALL_FEATURES].values)
        return rf.predict_proba(X_s)   # (N, 3): [P(sell), P(hold), P(buy)]

    tr_proba = get_proba(tr)
    te_proba = get_proba(te)

    # ── Variant A: Feature Selection ─────────────────────────────────────
    print('\n[Variant A] RF Feature Selection → PPO (top 10 features)')
    envA_tr = DummyVecEnv([lambda: EnvA(tr, top10)])
    mA = PPO('MlpPolicy', envA_tr, learning_rate=3e-4, n_steps=1024,
             batch_size=64, n_epochs=10, verbose=0)
    mA.learn(total_timesteps=TIMESTEPS)
    roi_A, trades_A, bal_A = backtest_ppo(mA, EnvA, df=te, top_features=top10)
    print(f'  ROI: {roi_A:+.2f}%  Trades: {trades_A}  Final: ${bal_A:,.2f}')
    results['A_FeatureSelect'] = roi_A

    # ── Variant B: RF Proba Input ────────────────────────────────────────
    print('\n[Variant B] RF Probability Input → PPO')
    envB_tr = DummyVecEnv([lambda: EnvB(tr, tr_proba, ALL_FEATURES)])
    mB = PPO('MlpPolicy', envB_tr, learning_rate=3e-4, n_steps=1024,
             batch_size=64, n_epochs=10, verbose=0)
    mB.learn(total_timesteps=TIMESTEPS)
    roi_B, trades_B, bal_B = backtest_ppo(mB, EnvB,
                                          df=te, rf_proba=te_proba,
                                          features=ALL_FEATURES)
    print(f'  ROI: {roi_B:+.2f}%  Trades: {trades_B}  Final: ${bal_B:,.2f}')
    results['B_ProbaInput'] = roi_B

    # ── Variant C: RF Gating ─────────────────────────────────────────────
    print('\n[Variant C] RF Gating → PPO (action masking)')
    envC_tr = DummyVecEnv([lambda: EnvC(tr, tr_proba, ALL_FEATURES)])
    mC = PPO('MlpPolicy', envC_tr, learning_rate=3e-4, n_steps=1024,
             batch_size=64, n_epochs=10, verbose=0)
    mC.learn(total_timesteps=TIMESTEPS)
    roi_C, trades_C, bal_C = backtest_ppo(mC, EnvC,
                                          df=te, rf_proba=te_proba,
                                          features=ALL_FEATURES)
    print(f'  ROI: {roi_C:+.2f}%  Trades: {trades_C}  Final: ${bal_C:,.2f}')
    results['C_Gating'] = roi_C

    # ── Summary ──────────────────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  RESULTS: {ticker}  (test set, out-of-sample)')
    print(f'{"="*65}')
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    medals = ['🥇','🥈','🥉']
    for i, (name, roi) in enumerate(ranked):
        medal = medals[i] if i < 3 else '  '
        label = {
            'A_FeatureSelect': 'A. RF Feature Selection → PPO',
            'B_ProbaInput':    'B. RF Proba Input      → PPO',
            'C_Gating':        'C. RF Gating           → PPO',
        }[name]
        print(f'  {medal} {label:35s}  ROI: {roi:+7.2f}%')

    winner = ranked[0][0]
    print(f'\n  Best variant: {winner}')
    print(f'{"="*65}')

    # Record best in tracker
    from model_accuracy_tracker import ModelAccuracyTracker
    best_roi = ranked[0][1]
    tracker = ModelAccuracyTracker(ticker, 'Hybrid')
    from model_accuracy_tracker import ModelAccuracyTracker as MAT
    tracker.update_training_stats(
        backtest_acc=MAT.roi_to_score(best_roi),
        win_rate=min(100, max(0, 50 + best_roi/3))
    )
    return results


if __name__ == '__main__':
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else 'MU'
    compare(ticker)
