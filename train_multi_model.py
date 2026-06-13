"""
Multi-Model Training Framework with ROI Backtesting
====================================================
Trains 4 models and compares BOTH accuracy and actual ROI:
  Model 1: PPO           (reinforcement learning, continuous action)
  Model 2: DQN           (reinforcement learning, discrete action)
  Model 3: XGBoost       (gradient boosting classifier)
  Model 4: Hybrid RF→PPO (Random Forest features + PPO)

Usage:
  python train_multi_model.py TICKER [EXCHANGE]
  python train_multi_model.py MU
  python train_multi_model.py 2330 TW
"""
import os, sys, io, json, warnings, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

# ── Feature columns ───────────────────────────────────────────────────────────
FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','sma_200','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']

INITIAL_BALANCE = 10_000.0
PPO_STEPS       = 100_000
DQN_STEPS       = 100_000


# ── Data preparation ──────────────────────────────────────────────────────────
def download_and_prepare(ticker, start='2015-01-01', end='2026-05-01'):
    print(f"  Downloading {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    if df.empty or len(df) < 100:
        raise ValueError(f"Insufficient data: {len(df)} rows")
    df = df.rename(columns={'Close':'close','Volume':'volume',
                             'Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"  {len(df)} trading days loaded")
    return df

def add_features(df):
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


# ── Environments ──────────────────────────────────────────────────────────────
class ContinuousEnv(gym.Env):
    """For PPO / Hybrid — continuous action [-1, 1]"""
    def __init__(self, df, obs_size=15):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.obs_size = obs_size
        self.action_space      = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0; self.bal = INITIAL_BALANCE; self.sh = 0; self.profit = 0.0
        return self._obs(), {}

    def _obs(self):
        r = self.df.iloc[self.i]; p = float(r['close']); tv = self.bal + self.sh * p
        base = [float(self.sh), float(self.bal), p,
                float(r.get('sma_10',0)), float(r.get('sma_30',0)), float(r.get('sma_50',0)),
                float(r.get('rsi',50)),   float(r.get('macd',0)),   float(r.get('macd_signal',0)),
                float(r.get('bb_u',0)),   float(r.get('bb_l',0)),   float(r.get('volume',0)),
                float(self.profit),
                (self.sh * p) / tv if tv > 0 else 0,
                self.bal / tv if tv > 0 else 1]
        # pad or add RF signal if obs_size > 15
        if self.obs_size > 15:
            base.append(float(r.get('rf_signal', 0.5)))
        return np.array(base[:self.obs_size], dtype=np.float32)

    def step(self, action):
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        a = np.clip(a, -1, 1); p = float(self.df.iloc[self.i]['close'])
        if a < -0.1:
            s = int(self.sh * abs(a))
            if s > 0: self.bal += s * p; self.sh -= s
        elif a > 0.1:
            s = int((self.bal // p) * a)
            if s > 0: self.bal -= s * p; self.sh += s
        self.profit = (self.bal + self.sh * p) - INITIAL_BALANCE
        self.i += 1; done = self.i >= len(self.df) - 1
        return self._obs(), self.profit / INITIAL_BALANCE + (0.01 if abs(a) > 0.1 else 0), done, False, {}

    def portfolio_value(self):
        return self.bal + self.sh * float(self.df.iloc[self.i]['close'])


class DiscreteEnv(gym.Env):
    """For DQN — discrete actions: 0=HOLD, 1=BUY, 2=SELL"""
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0; self.bal = INITIAL_BALANCE; self.sh = 0; self.profit = 0.0
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
        p = float(self.df.iloc[self.i]['close'])
        if action == 1 and self.bal >= p:        # BUY all-in
            s = int(self.bal // p); self.bal -= s * p; self.sh += s
        elif action == 2 and self.sh > 0:         # SELL all
            self.bal += self.sh * p; self.sh = 0
        self.profit = (self.bal + self.sh * p) - INITIAL_BALANCE
        self.i += 1; done = self.i >= len(self.df) - 1
        rew = self.profit / INITIAL_BALANCE
        return self._obs(), rew, done, False, {}

    def portfolio_value(self):
        return self.bal + self.sh * float(self.df.iloc[self.i]['close'])


# ── ROI simulators ────────────────────────────────────────────────────────────
def roi_from_rl(model, env_class, te, **kwargs):
    """Run RL model on test data and return (roi, accuracy)"""
    env = env_class(te.reset_index(drop=True), **kwargs)
    obs, _ = env.reset()
    preds, labels = [], []
    while True:
        act, _ = model.predict(obs, deterministic=True)
        if isinstance(act, np.ndarray): act_val = int(act[0] > 0.1) if act.shape else int(act)
        else: act_val = int(float(act) > 0.1)
        preds.append(act_val)
        labels.append(int(te.iloc[env.i if hasattr(env,'i') else env.current_step]['target']))
        obs, _, done, _, _ = env.step(act)
        if done: break
    final_val = env.portfolio_value()
    roi = (final_val - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    acc = accuracy_score(labels, preds) * 100 if preds else 0
    return round(roi, 2), round(acc, 2)

def roi_from_xgb(model, te):
    """Simulate XGBoost trading: buy on signal=1, sell on signal=0"""
    bal = INITIAL_BALANCE; sh = 0
    for i in range(len(te) - 1):
        X   = te[FEAT].iloc[[i]]
        pred = model.predict(X)[0]
        price = float(te['close'].iloc[i])
        if pred == 1 and sh == 0:          # BUY all-in
            sh = bal // price; bal -= sh * price
        elif pred == 0 and sh > 0:          # SELL all
            bal += sh * price; sh = 0
    if sh > 0: bal += sh * float(te['close'].iloc[-1])
    roi = (bal - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    # accuracy on test set
    X_all = te[FEAT]; y_all = te['target']
    Xt, Xe, yt, ye = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)
    acc = accuracy_score(ye, model.predict(Xe)) * 100
    return round(roi, 2), round(acc, 2)


# ── Main training function ─────────────────────────────────────────────────────
def train_all_models(ticker, start='2015-01-01'):
    print(f"\n{'='*65}")
    print(f"  MULTI-MODEL TRAINING: {ticker}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

    # Download & split data
    df = download_and_prepare(ticker, start)
    df = add_features(df)
    dc = df.dropna(subset=FEAT + ['target'])
    split = int(len(dc) * 0.8)
    tr = dc.iloc[:split].copy()
    te = dc.iloc[split:].copy()
    print(f"  Train: {len(tr)} | Test: {len(te)}")

    results = {}
    sl = ticker.lower().replace('.', '_')

    # ── Model 1: PPO ─────────────────────────────────────────────────────────
    print(f"\n  ├── Model 1: PPO  ({PPO_STEPS//1000}k steps)...")
    env = DummyVecEnv([lambda: ContinuousEnv(tr)])
    ppo = PPO('MlpPolicy', env, verbose=0, learning_rate=0.0003,
              n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    ppo.learn(total_timesteps=PPO_STEPS)
    ppo.save(f'ppo_{sl}_improved')
    ppo_roi, ppo_acc = roi_from_rl(ppo, ContinuousEnv, te)
    results['PPO'] = {'roi': ppo_roi, 'acc': ppo_acc}
    print(f"  │   Accuracy: {ppo_acc:.2f}%  ROI: {ppo_roi:+.2f}%")

    # ── Model 2: DQN ─────────────────────────────────────────────────────────
    print(f"\n  ├── Model 2: DQN  ({DQN_STEPS//1000}k steps)...")
    denv = DummyVecEnv([lambda: DiscreteEnv(tr)])
    dqn = DQN('MlpPolicy', denv, verbose=0, learning_rate=0.0003,
              batch_size=64, gamma=0.99, exploration_fraction=0.3,
              exploration_final_eps=0.05, target_update_interval=500)
    dqn.learn(total_timesteps=DQN_STEPS)
    dqn.save(f'dqn_{sl}_model')
    dqn_roi, dqn_acc = roi_from_rl(dqn, DiscreteEnv, te)
    results['DQN'] = {'roi': dqn_roi, 'acc': dqn_acc}
    print(f"  │   Accuracy: {dqn_acc:.2f}%  ROI: {dqn_roi:+.2f}%")

    # ── Model 3: XGBoost ─────────────────────────────────────────────────────
    print(f"\n  ├── Model 3: XGBoost...")
    X = dc[FEAT]; y = dc['target']
    Xt, Xe, yt, ye = train_test_split(X, y, test_size=0.2, shuffle=False)
    xgb_m = xgb.XGBClassifier(max_depth=5, learning_rate=0.05, n_estimators=200,
                                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                objective='binary:logistic', random_state=42, eval_metric='logloss')
    xgb_m.fit(Xt, yt)
    joblib.dump(xgb_m, f'xgb_{sl}_model.pkl')
    xgb_roi, xgb_acc = roi_from_xgb(xgb_m, te)
    results['XGBoost'] = {'roi': xgb_roi, 'acc': xgb_acc}
    print(f"  │   Accuracy: {xgb_acc:.2f}%  ROI: {xgb_roi:+.2f}%")

    # ── Model 4: Hybrid RF → PPO ─────────────────────────────────────────────
    print(f"\n  └── Model 4: Hybrid RF→PPO...")
    # Step A: train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(Xt, yt)
    joblib.dump(rf, f'rf_{sl}_model.pkl')
    # Step B: add RF buy-probability as extra feature in observations
    tr2 = tr.copy()
    te2 = te.copy()
    tr2['rf_signal'] = rf.predict_proba(tr2[FEAT])[:, 1]
    te2['rf_signal'] = rf.predict_proba(te2[FEAT])[:, 1]
    # Step C: train PPO with 16-dim obs (15 standard + rf_signal)
    henv = DummyVecEnv([lambda: ContinuousEnv(tr2, obs_size=16)])
    h_ppo = PPO('MlpPolicy', henv, verbose=0, learning_rate=0.0003,
                n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, ent_coef=0.01)
    h_ppo.learn(total_timesteps=PPO_STEPS)
    h_ppo.save(f'hybrid_rf_ppo_{sl}_model')
    hybrid_roi, hybrid_acc = roi_from_rl(h_ppo, ContinuousEnv, te2, obs_size=16)
    results['Hybrid RF→PPO'] = {'roi': hybrid_roi, 'acc': hybrid_acc}
    print(f"      Accuracy: {hybrid_acc:.2f}%  ROI: {hybrid_roi:+.2f}%")

    # ── Save comparison JSON ──────────────────────────────────────────────────
    best_roi    = max(results, key=lambda m: results[m]['roi'])
    best_acc    = max(results, key=lambda m: results[m]['acc'])
    summary = {
        'ticker': ticker, 'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': results, 'best_roi': best_roi, 'best_accuracy': best_acc
    }
    json.dump(summary, open(f'model_comparison_{sl}.json','w',encoding='utf-8'), indent=2, ensure_ascii=False)

    # Also update model_accuracy file with XGBoost (most reliable for display)
    json.dump({'symbol': ticker, 'model_type': 'XGBoost',
               'validation_accuracy': xgb_acc, 'backtest_accuracy': xgb_acc,
               'roi': xgb_roi, 'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
              open(f'model_accuracy_{sl.upper().replace("_","")}.json','w',encoding='utf-8'), indent=2)

    # ── Final comparison table ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  📊 {ticker} — MODEL COMPARISON")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'ROI':>12}  {'Status'}")
    print(f"  {'-'*58}")

    roi_vals = [v['roi'] for v in results.values()]
    max_roi = max(roi_vals)
    for i, (name, data) in enumerate(results.items()):
        prefix = '├──' if i < len(results)-1 else '└──'
        roi_flag = ' ✅' if data['roi'] == max_roi else ''
        roi_str = f"{data['roi']:+.2f}%{roi_flag}"
        print(f"  {prefix} {name:<17} {data['acc']:>9.2f}%  {roi_str:>14}")

    print(f"\n  🏆 Best ROI:      {best_roi} ({results[best_roi]['roi']:+.2f}%)")
    print(f"  🎯 Best Accuracy: {best_acc} ({results[best_acc]['acc']:.2f}%)")
    print(f"\n  Saved: model_comparison_{sl}.json")
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-model training with ROI comparison')
    parser.add_argument('ticker', help='Stock ticker (e.g. MU, 2330, 2330.TW)')
    parser.add_argument('--start', default='2015-01-01', help='Start date')
    args = parser.parse_args()
    ticker = args.ticker.upper()
    train_all_models(ticker, start=args.start)
