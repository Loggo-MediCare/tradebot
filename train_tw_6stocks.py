"""Train XGBoost + PPO for 3443,3037,3189,8046,3017,3665"""
import os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib, gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','sma_200','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']

STOCKS = [
    ('3443','3443.TW','創意','tw'),
    ('3037','3037.TW','欣興','tw'),
    ('3189','3189.TW','景碩','tw'),
    ('8046','8046.TW','南電','tw'),
    ('3017','3017.TW','奇鋐','tw'),
    ('3665','3665.TW','貿聯KY','tw'),
]

def prep(ticker):
    df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
    df['sma_10']=df['close'].rolling(10).mean(); df['sma_30']=df['close'].rolling(30).mean()
    df['sma_50']=df['close'].rolling(50).mean(); df['sma_200']=df['close'].rolling(200).mean()
    df['ema_12']=df['close'].ewm(span=12).mean(); df['ema_26']=df['close'].ewm(span=26).mean()
    d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/(l+1e-10))); df['macd']=df['ema_12']-df['ema_26']
    df['macd_signal']=df['macd'].ewm(span=9).mean(); df['macd_hist']=df['macd']-df['macd_signal']
    df['bb_m']=df['close'].rolling(20).mean(); df['bb_s']=df['close'].rolling(20).std()
    df['bb_u']=df['bb_m']+2*df['bb_s']; df['bb_l']=df['bb_m']-2*df['bb_s']
    df['bb_position']=((df['close']-df['bb_l'])/(df['bb_u']-df['bb_l'])*100).fillna(50)
    lo14=df['low'].rolling(14).min(); hi14=df['high'].rolling(14).max()
    df['K']=((df['close']-lo14)/(hi14-lo14)*100).fillna(50); df['D']=df['K'].rolling(3).mean()
    df['obv']=(np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum(); df['obv_ma20']=df['obv'].rolling(20).mean()
    df['volatility']=df['close'].rolling(20).std()/df['close'].rolling(20).mean()
    hl=df['high']-df['low']; hc=np.abs(df['high']-df['close'].shift()); lc=np.abs(df['low']-df['close'].shift())
    df['atr']=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']=df['close'].pct_change(5)*100; df['price_change_10d']=df['close'].pct_change(10)*100
    df['price_change_20d']=df['close'].pct_change(20)*100; df['ma50_slope']=df['sma_50'].diff(5)/df['sma_50'].shift(5)*100
    df['future_return']=df['close'].shift(-5)/df['close']-1; df['target']=(df['future_return']>0.02).astype(int)
    return df.bfill().ffill()

class TradingEnv(gym.Env):
    def __init__(self, df):
        super().__init__(); self.df=df.reset_index(drop=True)
        self.action_space=spaces.Box(-1.0,1.0,(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(15,),dtype=np.float32); self.reset()
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.i=0; self.bal=10000.0; self.sh=0; self.profit=0.0; return self._o(),{}
    def _o(self):
        r=self.df.iloc[self.i]; p=float(r['close']); tv=self.bal+self.sh*p
        return np.array([float(self.sh),float(self.bal),p,float(r.get('sma_10',0)),float(r.get('sma_30',0)),
                         float(r.get('sma_50',0)),float(r.get('rsi',50)),float(r.get('macd',0)),
                         float(r.get('macd_signal',0)),float(r.get('bb_u',0)),float(r.get('bb_l',0)),
                         float(r.get('volume',0)),float(self.profit),
                         (self.sh*p)/tv if tv>0 else 0,self.bal/tv if tv>0 else 1],dtype=np.float32)
    def step(self,action):
        a=np.clip(float(action[0]) if isinstance(action,np.ndarray) else float(action),-1,1)
        p=float(self.df.iloc[self.i]['close'])
        if a<-0.1:
            s=int(self.sh*abs(a))
            if s>0: self.bal+=s*p; self.sh-=s
        elif a>0.1:
            s=int((self.bal//p)*a)
            if s>0: self.bal-=s*p; self.sh+=s
        self.profit=(self.bal+self.sh*p)-10000.0; self.i+=1; done=self.i>=len(self.df)-1
        rew=self.profit/10000.0
        if abs(a)>0.1: rew+=0.01
        return self._o(),rew,done,False,{}

results=[]
for code,ticker,name,suffix in STOCKS:
    print(f"\n{'='*55}\n  {ticker} ({name})\n{'='*55}")
    df=prep(ticker); dc=df.dropna(subset=FEAT+['target'])
    print(f"  Data: {len(dc)} rows")
    split=int(len(dc)*0.8); tr=dc.iloc[:split].copy(); te=dc.iloc[split:].copy()

    X=dc[FEAT]; y=dc['target']
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,shuffle=False)
    m=xgb.XGBClassifier(max_depth=5,learning_rate=0.05,n_estimators=200,min_child_weight=3,
                         subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',
                         random_state=42,eval_metric='logloss')
    m.fit(Xt,yt); xgb_acc=accuracy_score(ye,m.predict(Xe))
    joblib.dump(m,f'xgb_{code}_{suffix}_model.pkl')
    with open(f'model_accuracy_{code}_{suffix.upper()}.json','w',encoding='utf-8') as f:
        json.dump({'symbol':ticker,'model_type':'XGBoost',
                   'training_accuracy':float(accuracy_score(yt,m.predict(Xt))*100),
                   'validation_accuracy':float(xgb_acc*100),'backtest_accuracy':float(xgb_acc*100),
                   'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},f,indent=2)
    print(f"  XGBoost: {xgb_acc*100:.2f}%")

    env=DummyVecEnv([lambda tr=tr: TradingEnv(tr)])
    ppo=PPO('MlpPolicy',env,verbose=0,learning_rate=0.0003,n_steps=2048,batch_size=64,n_epochs=10,gamma=0.99,ent_coef=0.01)
    ppo.learn(total_timesteps=100000); ppo.save(f'ppo_{code}_{suffix}_improved')
    env2=TradingEnv(te.reset_index(drop=True)); obs,_=env2.reset()
    preds,labels=[],[]
    for i in range(len(te)-6):
        act,_=ppo.predict(obs,deterministic=True)
        preds.append(1 if float(act[0])>0.1 else 0); labels.append(int(te.iloc[i]['target']))
        obs,_,done,_,_=env2.step(act)
        if done: break
    ppo_acc=accuracy_score(labels,preds) if preds else 0
    with open(f'model_accuracy_{code}_{suffix.upper()}_ppo.json','w',encoding='utf-8') as f:
        json.dump({'symbol':ticker,'model_type':'PPO','training_accuracy':float(ppo_acc*100),
                   'validation_accuracy':float(ppo_acc*100),'backtest_accuracy':float(ppo_acc*100),
                   'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')},f,indent=2)
    print(f"  PPO:     {ppo_acc*100:.2f}% (buy={sum(preds)}/{len(preds)})")
    results.append((ticker,name,xgb_acc,ppo_acc))

print(f"\n{'='*55}\n  COMPARISON SUMMARY\n{'='*55}")
print(f"  {'Ticker':<12} {'Name':<8} {'XGBoost':>9} {'PPO':>9} {'Winner':>9}")
for t,n,xa,pa in results:
    w='XGBoost' if xa>=pa else 'PPO'
    print(f"  {t:<12} {n:<8} {xa*100:>8.2f}% {pa*100:>8.2f}% {w:>9}")
print(f"{'='*55}")
