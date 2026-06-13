"""
Calculate Sharpe ratio for all 4 MU models and update accuracy tracker.
Sharpe = (mean daily return * 252 - rf) / (std daily return * sqrt(252))
"""
import os, sys, io, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import yfinance as yf

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RF_RATE   = 0.05   # 5% risk-free rate

TICKER    = sys.argv[1].upper() if len(sys.argv) > 1 else 'MU'


def download_data(ticker, days=1095):
    end = datetime.now()
    df  = yf.download(ticker, start=end-timedelta(days=days),
                      end=end, progress=False, auto_adjust=True)
    if df.empty: raise ValueError(f"No data: {ticker}")
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Open':'open','High':'high',
                             'Low':'low','Volume':'volume'})
    return df.reset_index()


def add_indicators(df):
    c = df['close']
    for w in [10,20,30,50]: df[f'sma_{w}'] = c.rolling(w).mean()
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    d=c.diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']         = 100-(100/(1+g/(l+1e-9)))
    df['macd']        = e12-e26
    df['macd_signal'] = df['macd'].ewm(span=9,adjust=False).mean()
    df['macd_hist']   = df['macd']-df['macd_signal']
    m=c.rolling(20).mean(); s=c.rolling(20).std()
    df['bb_upper']    = m+s*2; df['bb_lower']=m-s*2
    df['bb_pos']      = (c-df['bb_lower'])/(df['bb_upper']-df['bb_lower']+1e-9)
    df['vol_ratio']   = df['volume']/df['volume'].rolling(20).mean()
    df['atr']         = (df['high']-df['low']).rolling(14).mean()
    df['volatility']  = c.pct_change().rolling(20).std()
    for p in [5,10,20]: df[f'roc_{p}']=c.pct_change(p)*100
    return df.bfill().ffill()


HYBRID_FEATURES = ['sma_10','sma_20','sma_30','sma_50','rsi','macd','macd_signal',
                   'macd_hist','bb_pos','vol_ratio','atr','volatility','roc_5','roc_10','roc_20']


def calc_sharpe(portfolio_values, rf=RF_RATE):
    """Annualized Sharpe from list of daily portfolio values."""
    vals  = np.array(portfolio_values)
    if len(vals) < 2: return 0.0
    daily_ret = np.diff(vals) / vals[:-1]
    mean_ret  = np.mean(daily_ret) * 252
    std_ret   = np.std(daily_ret) * np.sqrt(252)
    if std_ret == 0: return 0.0
    return round((mean_ret - rf) / std_ret, 4)


# ── PPO backtest ──────────────────────────────────────────────────────────────
def backtest_ppo(df):
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces

    model_path = os.path.join(BASE_DIR, f'ppo_{TICKER.lower()}_improved')
    if not os.path.exists(model_path + '.zip'):
        print(f'  PPO model not found: {model_path}.zip')
        return None

    class Env(gym.Env):
        def __init__(self, df):
            super().__init__()
            self.df=df.reset_index(drop=True)
            self.action_space=spaces.Box(-1.0,1.0,(1,),np.float32)
            self.observation_space=spaces.Box(-np.inf,np.inf,(15,),np.float32)
            self.reset()
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            self.i=0;self.bal=10000;self.shares=0;self.profit=0
            self.portfolio_hist=[10000.0]
            return self._obs(),{}
        def _obs(self):
            row=self.df.iloc[self.i]; p=float(row.get('close',0))
            tv=self.bal+self.shares*p
            return np.array([self.shares,self.bal,p,
                float(row.get('sma_10',0)),float(row.get('sma_30',0)),float(row.get('sma_50',0)),
                float(row.get('rsi',50)),float(row.get('macd',0)),float(row.get('macd_signal',0)),
                float(row.get('bb_upper',0)),float(row.get('bb_lower',0)),float(row.get('volume',0)),
                self.profit, self.shares*p/tv if tv>0 else 0, self.bal/tv if tv>0 else 1
            ],dtype=np.float32)
        def step(self,action):
            a=float(action[0]) if hasattr(action,'__len__') else float(action)
            a=max(-1.0,min(1.0,a)); p=float(self.df.iloc[self.i]['close'])
            if a<-0.1 and self.shares>0:
                s=int(self.shares*abs(a))
                if s>0: self.bal+=s*p; self.shares-=s
            elif a>0.1 and self.bal>=p:
                b=int(self.bal*a//p)
                if b>0: self.bal-=b*p; self.shares+=b
            self.i+=1; done=self.i>=len(self.df)-1
            pn=float(self.df.iloc[self.i]['close']); tv=self.bal+self.shares*pn
            self.profit=tv-10000
            self.portfolio_hist.append(tv)
            if done: self.bal+=self.shares*pn; self.shares=0
            return self._obs(), self.profit/10000, done, False, {}

    model = PPO.load(model_path)
    env   = Env(df)
    obs,_ = env.reset()
    done  = False
    while not done:
        action,_ = model.predict(obs, deterministic=True)
        obs,_,done,_,_ = env.step(action)
    roi    = env.profit/10000*100
    sharpe = calc_sharpe(env.portfolio_hist)
    return roi, sharpe, env.portfolio_hist


# ── XGBoost backtest ──────────────────────────────────────────────────────────
def backtest_xgb(df):
    import xgboost as xgb
    model_path = os.path.join(BASE_DIR, f'xgb_{TICKER.lower()}_model.json')
    if not os.path.exists(model_path):
        print(f'  XGBoost model not found')
        return None
    model   = xgb.XGBClassifier(); model.load_model(model_path)
    scaler_data = joblib.load(model_path.replace('_model.json','_scaler.pkl'))
    if isinstance(scaler_data, dict):
        scaler = scaler_data['scaler']; feats = scaler_data['features']
    else:
        scaler = scaler_data
        feats  = joblib.load(model_path.replace('_model.json','_features.pkl'))

    df_x    = add_indicators(df.copy()).dropna().reset_index(drop=True)
    fc      = [f for f in feats if f in df_x.columns]
    # Only use features that match the scaler's expected count
    n_expected = scaler.n_features_in_ if hasattr(scaler,'n_features_in_') else len(fc)
    fc = fc[:n_expected]
    X       = scaler.transform(df_x[fc].values)
    preds   = model.predict(X)   # 0=SELL, 1=HOLD, 2=BUY

    bal=10000; shares=0; portfolio=[10000.0]
    for i,pred in enumerate(preds):
        p=float(df_x['close'].iloc[i])
        if pred==2 and bal>=p:
            b=int(bal*0.5//p)
            if b>0: bal-=b*p; shares+=b
        elif pred==0 and shares>0:
            s=int(shares*0.5)
            if s>0: bal+=s*p; shares-=s
        pn=float(df_x['close'].iloc[min(i+1,len(df_x)-1)])
        portfolio.append(bal+shares*pn)

    bal += shares*float(df_x['close'].iloc[-1])
    roi    = (bal-10000)/10000*100
    sharpe = calc_sharpe(portfolio)
    return roi, sharpe, portfolio


# ── Hybrid backtest ───────────────────────────────────────────────────────────
def backtest_hybrid(df):
    from stable_baselines3 import PPO
    import gymnasium as gym
    from gymnasium import spaces

    ppo_path = os.path.join(BASE_DIR, f'hybrid_{TICKER.lower()}_ppo')
    rf_path  = os.path.join(BASE_DIR, f'hybrid_{TICKER.lower()}_rf.pkl')
    if not os.path.exists(ppo_path+'.zip') or not os.path.exists(rf_path):
        print(f'  Hybrid model not found')
        return None

    rf_data  = joblib.load(rf_path)
    rf, rf_sc= rf_data['rf'], rf_data['scaler']
    model    = PPO.load(ppo_path)

    df_h = add_indicators(df.copy()).dropna().reset_index(drop=True)
    fc   = [f for f in HYBRID_FEATURES if f in df_h.columns]
    n_expected = rf_sc.n_features_in_ if hasattr(rf_sc,'n_features_in_') else len(fc)
    fc   = fc[:n_expected]
    X_s  = rf_sc.transform(df_h[fc].values)
    proba= rf.predict_proba(X_s)

    class HybridEnv(gym.Env):
        def __init__(self, df, proba):
            super().__init__()
            self.df=df.reset_index(drop=True); self.proba=proba
            self.action_space=spaces.Box(-1.0,1.0,(1,),np.float32)
            self.observation_space=spaces.Box(-np.inf,np.inf,(17,),np.float32)
            self.reset()
        def reset(self,seed=None,options=None):
            super().reset(seed=seed)
            self.i=0;self.bal=10000;self.shares=0;self.profit=0
            self.portfolio_hist=[10000.0]
            return self._obs(),{}
        def _obs(self):
            row=self.df.iloc[self.i]; p=float(row.get('close',0))
            tv=self.bal+self.shares*p; rf=self.proba[self.i]
            return np.array([self.shares,self.bal,p,
                float(row.get('sma_10',0)),float(row.get('sma_30',0)),
                float(row.get('rsi',50)),float(row.get('macd',0)),float(row.get('macd_signal',0)),
                float(row.get('bb_pos',0))*1000,float(row.get('bb_pos',0))*500,
                float(row.get('volume',0)),self.profit,
                self.shares*p/tv if tv>0 else 0, self.bal/tv if tv>0 else 1,
                float(rf[0]),float(rf[1]),float(rf[2])
            ],dtype=np.float32)
        def step(self,action):
            a=float(action[0]) if hasattr(action,'__len__') else float(action)
            a=max(-1.0,min(1.0,a)); p=float(self.df.iloc[self.i]['close'])
            if a<-0.1 and self.shares>0:
                s=int(self.shares*abs(a))
                if s>0: self.bal+=s*p; self.shares-=s
            elif a>0.1 and self.bal>=p:
                b=int(self.bal*a//p)
                if b>0: self.bal-=b*p; self.shares+=b
            self.i+=1; done=self.i>=len(self.df)-1
            pn=float(self.df.iloc[self.i]['close']); tv=self.bal+self.shares*pn
            self.profit=tv-10000
            self.portfolio_hist.append(tv)
            if done: self.bal+=self.shares*pn; self.shares=0
            return self._obs(),self.profit/10000,done,False,{}

    env=HybridEnv(df_h,proba); obs,_=env.reset()
    done=False
    while not done:
        action,_=model.predict(obs,deterministic=True)
        obs,_,done,_,_=env.step(action)
    roi    = env.profit/10000*100
    sharpe = calc_sharpe(env.portfolio_hist)
    return roi, sharpe, env.portfolio_hist


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    sys.path.insert(0, BASE_DIR)
    from model_accuracy_tracker import ModelAccuracyTracker

    print(f'\n{"="*60}')
    print(f'  Sharpe Ratio Calculation  |  {TICKER}')
    print(f'{"="*60}')

    print('\nDownloading data...')
    df = add_indicators(download_data(TICKER))
    sp = int(len(df) * 0.8)
    test_df = df.iloc[sp:].reset_index(drop=True)
    print(f'Test set: {len(test_df)} rows\n')

    results = {}

    # PPO
    print('[ PPO ]')
    try:
        roi, sharpe, _ = backtest_ppo(test_df)
        print(f'  ROI: {roi:+.2f}%  Sharpe: {sharpe:.3f}')
        t = ModelAccuracyTracker(TICKER, 'PPO')
        t.update_training_stats(sharpe_ratio=sharpe)
        results['PPO'] = (roi, sharpe)
    except Exception as e: print(f'  ERROR: {e}')

    # XGBoost
    print('[ XGBoost ]')
    try:
        roi, sharpe, _ = backtest_xgb(test_df)
        print(f'  ROI: {roi:+.2f}%  Sharpe: {sharpe:.3f}')
        t = ModelAccuracyTracker(TICKER, 'XGBoost')
        t.update_training_stats(sharpe_ratio=sharpe,
                                backtest_acc=ModelAccuracyTracker.roi_to_score(roi))
        results['XGBoost'] = (roi, sharpe)
    except Exception as e: print(f'  ERROR: {e}')

    # Hybrid
    print('[ Hybrid RF→PPO ]')
    try:
        roi, sharpe, _ = backtest_hybrid(test_df)
        print(f'  ROI: {roi:+.2f}%  Sharpe: {sharpe:.3f}')
        t = ModelAccuracyTracker(TICKER, 'Hybrid')
        t.update_training_stats(sharpe_ratio=sharpe,
                                backtest_acc=ModelAccuracyTracker.roi_to_score(roi))
        results['Hybrid'] = (roi, sharpe)
    except Exception as e: print(f'  ERROR: {e}')

    # Summary
    print(f'\n{"="*60}')
    print(f'  FINAL COMPARISON  |  {TICKER}  (test set)')
    print(f'{"="*60}')
    print(f'  {"Model":<20} {"ROI":>10}  {"Sharpe":>8}')
    print(f'  {"-"*42}')
    for m, (roi, sh) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f'  {m:<20} {roi:>+9.2f}%  {sh:>8.3f}')

    from model_accuracy_tracker import get_model_accuracy_display, get_best_model_type
    print(f'\n{get_model_accuracy_display(TICKER)}')
    w,p,d,x,h = get_best_model_type(TICKER)
    print(f'PPO:{p}  DQN:{d}  XGB:{x}  HYB:{h}')
