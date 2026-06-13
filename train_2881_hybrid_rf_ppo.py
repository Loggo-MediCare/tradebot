"""
2881 富邦金 — Hybrid RF→PPO Pipeline
Step 1: Train Random Forest → rank feature importance
Step 2: Select top-K features
Step 3: Build PPO TradingEnv using portfolio state + top-K features as obs
Step 4: Train PPO on enriched observation space
Step 5: Save model + selected feature list for signal script
"""
import os,sys,io,json,warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

import numpy as np,pandas as pd,yfinance as yf,gymnasium as gym,joblib
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from datetime import datetime

CODE='2881';TICKER='2881.TW';NAME='富邦金';TOP_K=10

ALL_FEAT=['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
          'sma_10','sma_30','sma_50','volatility','atr',
          'price_change_5d','price_change_10d','price_change_20d','ma50_slope']

print(f"=== {CODE} {NAME} Hybrid RF→PPO Pipeline ===")

# ── Data ──────────────────────────────────────────────────────────────────────
df=yf.download(TICKER,start='2020-01-01',end='2026-06-09',progress=False)
if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
df=df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
print(f"Data: {len(df)} days  ({df['Date'].iloc[0].date()} ~ {df['Date'].iloc[-1].date()})")

def build_features(df):
    df=df.copy()
    df['sma_10']=df['close'].rolling(10).mean(); df['sma_30']=df['close'].rolling(30).mean()
    df['sma_50']=df['close'].rolling(50).mean(); df['ema_12']=df['close'].ewm(span=12).mean()
    df['ema_26']=df['close'].ewm(span=26).mean()
    d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/(l+1e-10))); df['macd']=df['ema_12']-df['ema_26']
    df['macd_signal']=df['macd'].ewm(span=9).mean(); df['macd_hist']=df['macd']-df['macd_signal']
    df['bb_m']=df['close'].rolling(20).mean(); df['bb_s']=df['close'].rolling(20).std()
    df['bb_u']=df['bb_m']+2*df['bb_s']; df['bb_l']=df['bb_m']-2*df['bb_s']
    df['bb_position']=((df['close']-df['bb_l'])/(df['bb_u']-df['bb_l'])*100).fillna(50)
    lo14=df['low'].rolling(14).min(); hi14=df['high'].rolling(14).max()
    df['K']=((df['close']-lo14)/(hi14-lo14)*100).fillna(50); df['D']=df['K'].rolling(3).mean()
    df['obv']=(np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum()
    df['obv_ma20']=df['obv'].rolling(20).mean()
    df['volatility']=df['close'].rolling(20).std()/df['close'].rolling(20).mean()
    hl=df['high']-df['low']; hc=np.abs(df['high']-df['close'].shift()); lc=np.abs(df['low']-df['close'].shift())
    df['atr']=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']=df['close'].pct_change(5)*100
    df['price_change_10d']=df['close'].pct_change(10)*100
    df['price_change_20d']=df['close'].pct_change(20)*100
    df['ma50_slope']=df['sma_50'].diff(5)/df['sma_50'].shift(5)*100
    df['future_return']=df['close'].shift(-5)/df['close']-1
    df['target']=(df['future_return']>0.02).astype(int)
    return df.bfill().ffill()

df=build_features(df)
dc=df.dropna(subset=ALL_FEAT+['target'])
split=int(len(dc)*0.8)
tr=dc.iloc[:split].copy(); te=dc.iloc[split:].copy()
print(f"Train:{len(tr)} | Test:{len(te)}")

# ── Step 1: Random Forest → Feature Importance ────────────────────────────────
print(f"\n[Step 1] Random Forest — feature importance...")
X_tr=tr[ALL_FEAT]; y_tr=tr['target']
X_te=te[ALL_FEAT];  y_te=te['target']

rf=RandomForestClassifier(n_estimators=300,max_depth=6,min_samples_leaf=5,
                           random_state=42,n_jobs=-1)
rf.fit(X_tr,y_tr)
rf_acc=accuracy_score(y_te,rf.predict(X_te))
print(f"  RF validation accuracy: {rf_acc*100:.2f}%")

importance=pd.Series(rf.feature_importances_,index=ALL_FEAT).sort_values(ascending=False)
print(f"\n  Feature Importance (all {len(ALL_FEAT)}):")
for feat,imp in importance.items():
    bar='█'*int(imp*200)
    print(f"    {feat:<20} {imp:.4f}  {bar}")

# ── Step 2: Select Top-K Features ────────────────────────────────────────────
top_features=importance.head(TOP_K).index.tolist()
print(f"\n[Step 2] Top-{TOP_K} features selected:")
for i,f in enumerate(top_features,1):
    print(f"    {i:2d}. {f}  ({importance[f]:.4f})")

# save feature list for signal script
joblib.dump(top_features,'ppo_2881_hybrid_top_features.pkl')
print(f"  Saved: ppo_2881_hybrid_top_features.pkl")

# obs_dim = portfolio state (5) + top-K features
OBS_DIM=5+TOP_K
print(f"\n  PPO obs_dim: {OBS_DIM}  (5 portfolio + {TOP_K} RF-selected features)")

# ── Step 3 & 4: PPO with enriched obs space ───────────────────────────────────
class HybridTradingEnv(gym.Env):
    def __init__(self,df,top_feats):
        super().__init__()
        self.df=df.reset_index(drop=True)
        self.top_feats=top_feats
        self.obs_dim=5+len(top_feats)
        self.action_space=spaces.Box(low=-1.,high=1.,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,
                                          shape=(self.obs_dim,),dtype=np.float32)
        self.reset()
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.i=0;self.bal=10000.;self.sh=0;self.profit=0.
        return self._obs(),{}
    def _obs(self):
        r=self.df.iloc[self.i]; p=float(r['close']); tv=self.bal+self.sh*p
        # portfolio state
        port=[float(self.sh),float(self.bal),p,float(self.profit),
              (self.sh*p)/tv if tv>0 else 0]
        # RF-selected technical features (normalised roughly)
        tech=[float(r.get(f,0)) for f in self.top_feats]
        return np.array(port+tech,dtype=np.float32)
    def step(self,action):
        a=float(action[0]) if isinstance(action,np.ndarray) else float(action)
        a=np.clip(a,-1,1); p=float(self.df.iloc[self.i]['close'])
        if a<-0.1:
            s=int(self.sh*abs(a))
            if s>0: self.bal+=s*p; self.sh-=s
        elif a>0.1:
            s=int((self.bal//p)*a)
            if s>0: self.bal-=s*p; self.sh+=s
        self.profit=(self.bal+self.sh*p)-10000.
        self.i+=1; done=self.i>=len(self.df)-1
        reward=self.profit/10000.+(0.01 if abs(a)>0.1 else 0)
        return self._obs(),reward,done,False,{}

print(f"\n[Step 3] Training PPO (200k steps, obs={OBS_DIM})...")
env=DummyVecEnv([lambda: HybridTradingEnv(tr.reset_index(drop=True),top_features)])
pm=PPO('MlpPolicy',env,verbose=0,learning_rate=0.0003,
       n_steps=2048,batch_size=64,n_epochs=10,gamma=0.99,ent_coef=0.01)
pm.learn(total_timesteps=200_000)
pm.save('ppo_2881_hybrid_rf')
print("  Saved: ppo_2881_hybrid_rf")

# ── Step 5: Evaluate hybrid PPO on test set ───────────────────────────────────
print(f"\n[Step 4] Evaluating on test set...")
env2=HybridTradingEnv(te.reset_index(drop=True),top_features)
obs,_=env2.reset()
preds,labels=[],[]
for i in range(len(te)-6):
    act,_=pm.predict(obs,deterministic=True)
    preds.append(1 if float(act[0])>0.1 else 0)
    labels.append(int(te.iloc[i]['target']))
    obs,_,done,_,_=env2.step(act)
    if done: break

ppo_hybrid_acc=accuracy_score(labels,preds) if preds else 0.
print(f"  Hybrid RF→PPO accuracy: {ppo_hybrid_acc*100:.2f}%  (buys:{sum(preds)}/{len(preds)})")

# save accuracy JSON
acc_data={
    'symbol':CODE,'ticker':TICKER,'model_type':'Hybrid_RF_PPO',
    'rf_accuracy':round(rf_acc*100,2),
    'validation_accuracy':round(ppo_hybrid_acc*100,2),
    'backtest_accuracy':round(ppo_hybrid_acc*100,2),
    'top_features':top_features,
    'obs_dim':OBS_DIM,
    'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
json.dump(acc_data,open('model_accuracy_2881_hybrid.json','w',encoding='utf-8'),indent=2)

print(f"""
======================================
  2881 {NAME}  RF→PPO RESULTS
======================================
  Random Forest  : {rf_acc*100:.2f}%
  Hybrid RF→PPO  : {ppo_hybrid_acc*100:.2f}%
  (prev pure PPO : 26.80%)
  Top-{TOP_K} features: {', '.join(top_features[:5])}...
======================================""")
