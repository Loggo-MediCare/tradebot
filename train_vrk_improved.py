"""
改进版 VRK 股票交易 AI 训练
=====================================
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, gymnasium as gym
from gymnasium import spaces
import matplotlib; matplotlib.use('Agg')
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings; warnings.filterwarnings('ignore')

class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True); self.initial_balance = initial_balance; self.current_step = 0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); self.current_step=0; self.balance=self.initial_balance
        self.shares_held=0; self.total_profit=0; self.total_trades=0; self.last_action=0
        return self._get_observation(), {}
    def _get_observation(self):
        row=self.df.iloc[self.current_step]; cp=float(row['close']); tv=self.balance+self.shares_held*cp
        sr=(self.shares_held*cp)/tv if tv>0 else 0; cr=self.balance/tv if tv>0 else 1
        return np.array([float(self.shares_held),float(self.balance),cp,
            float(row.get('sma_10',0)),float(row.get('sma_30',0)),float(row.get('sma_50',0)),
            float(row.get('rsi',50)),float(row.get('macd',0)),float(row.get('macd_signal',0)),
            float(row.get('bb_upper',0)),float(row.get('bb_lower',0)),float(row.get('volume',0)),
            float(self.total_profit),sr,cr],dtype=np.float32)
    def step(self, action):
        action=np.clip(float(action[0]) if isinstance(action,np.ndarray) else float(action),-1.0,1.0)
        cp=float(self.df.iloc[self.current_step]['close']); otv=self.balance+self.shares_held*cp
        if action<-0.1:
            s=int(self.shares_held*abs(action))
            if s>0: self.balance+=s*cp; self.shares_held-=s; self.total_trades+=1
        elif action>0.1:
            s=int((self.balance//cp)*action)
            if s>0: self.balance-=s*cp; self.shares_held+=s; self.total_trades+=1
        ntv=self.balance+self.shares_held*cp; self.total_profit=ntv-self.initial_balance
        reward=self.total_profit/self.initial_balance+(0.01 if abs(action)>0.1 else 0)+(-0.005 if self.balance>otv*0.9 else 0)
        self.current_step+=1
        return self._get_observation(),float(reward),self.current_step>=len(self.df)-1,False,{}

def download_and_prepare_data(ticker, start_date='2015-01-01', end_date='2025-07-31'):
    import yfinance as yf
    df=yf.download(ticker,start=start_date,end=end_date,progress=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
    if df.empty: print(f"❌ {ticker}"); return None
    df=df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"✅ {ticker}: {len(df)} 天"); return df

def add_technical_indicators(df):
    df['sma_10']=df['close'].rolling(10).mean(); df['sma_30']=df['close'].rolling(30).mean(); df['sma_50']=df['close'].rolling(50).mean()
    df['ema_12']=df['close'].ewm(span=12).mean(); df['ema_26']=df['close'].ewm(span=26).mean()
    d=df['close'].diff(); g=d.where(d>0,0).rolling(14).mean(); l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/(l+1e-10))); df['macd']=df['ema_12']-df['ema_26']
    df['macd_signal']=df['macd'].ewm(span=9).mean(); df['macd_hist']=df['macd']-df['macd_signal']
    df['bb_middle']=df['close'].rolling(20).mean(); df['bb_std']=df['close'].rolling(20).std()
    df['bb_upper']=df['bb_middle']+df['bb_std']*2; df['bb_lower']=df['bb_middle']-df['bb_std']*2
    df['bb_position']=((df['close']-df['bb_lower'])/(df['bb_upper']-df['bb_lower'])*100).fillna(50)
    l14=df['low'].rolling(14).min(); h14=df['high'].rolling(14).max()
    df['K']=((df['close']-l14)/(h14-l14)*100).fillna(50); df['D']=df['K'].rolling(3).mean()
    df['OBV']=(np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum(); df['OBV_MA']=df['OBV'].rolling(20).mean()
    df['MA_20']=df['close'].rolling(20).mean(); df['MA_50']=df['close'].rolling(50).mean(); df['MA_200']=df['close'].rolling(200).mean()
    df['volatility']=df['close'].rolling(20).std()/df['close'].rolling(20).mean()
    tr=pd.concat([df['high']-df['low'],(df['high']-df['close'].shift()).abs(),(df['low']-df['close'].shift()).abs()],axis=1).max(axis=1)
    df['ATR']=tr.rolling(14).mean(); df['price_change_5d']=df['close'].pct_change(5)*100; df['price_change_20d']=df['close'].pct_change(20)*100
    df['MA50_slope']=df['MA_50'].diff(5)/df['MA_50'].shift(5)*100; df['future_direction']=(df['close'].shift(-5)>df['close']).astype(int)
    return df.fillna(method='bfill').fillna(method='ffill')

def analyze_feature_importance(df, ticker):
    from datetime import datetime; import json
    features=['rsi','macd','macd_signal','macd_hist','bb_position','K','D','OBV','OBV_MA','MA_20','MA_50','MA_200','volatility','ATR','price_change_5d','price_change_20d','MA50_slope']
    ml=df.dropna(subset=features+['future_direction'])
    if len(ml)==0: return
    Xs=StandardScaler().fit_transform(ml[features]); Xt,Xv,yt,yv=train_test_split(Xs,ml['future_direction'],test_size=0.2,shuffle=False)
    rf=RandomForestClassifier(n_estimators=100,random_state=42,class_weight='balanced'); rf.fit(Xt,yt)
    acc=accuracy_score(yv,rf.predict(Xv)); print(f"準確率: {acc:.4f}")
    fi=pd.DataFrame({'Feature':features,'Importance':rf.feature_importances_}).sort_values('Importance',ascending=False)
    with open(f'{ticker}_feature_importance.json','w',encoding='utf-8') as f:
        json.dump({'ticker':ticker,'analysis_date':datetime.now().strftime('%Y-%m-%d'),'model_accuracy':float(acc),'feature_importance':{r['Feature']:float(r['Importance']) for _,r in fi.iterrows()}},f,indent=2)
    print(f"✅ {ticker}_feature_importance.json")

def train_model(df, ticker, total_timesteps=100000):
    env=DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model=PPO('MlpPolicy',env,verbose=1,learning_rate=0.0003,n_steps=2048,batch_size=64,n_epochs=10,gamma=0.99,ent_coef=0.01)
    model.learn(total_timesteps=total_timesteps); model.save(f"ppo_{ticker.lower()}_improved")
    print(f"✅ ppo_{ticker.lower()}_improved.zip"); return model

if __name__ == "__main__":
    TICKER = 'VRK'
    df = download_and_prepare_data(TICKER)
    if df is None: exit(1)
    df = add_technical_indicators(df)
    train_model(df.iloc[:int(len(df)*0.8)].copy(), TICKER)
    analyze_feature_importance(df, TICKER)
    print(f"\n✅ 训练完成! ppo_{TICKER.lower()}_improved.zip")
