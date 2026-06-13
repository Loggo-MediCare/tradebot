"""Add XGBoost for 3491 昇達科 + 2313 華通 (PPO already exists)"""
import os, json, warnings, sys, io, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
warnings.filterwarnings('ignore')

STOCKS = [('3491', '昇達科', 'TW'), ('2313', '華通', 'TW')]
END_DATE = '2026-05-01'
FEAT = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
        'sma_10','sma_30','sma_50','sma_200','volatility','atr',
        'price_change_5d','price_change_10d','price_change_20d','ma50_slope']

def build_features(df):
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
    df['obv']=(np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum()
    df['obv_ma20']=df['obv'].rolling(20).mean()
    df['volatility']=df['close'].rolling(20).std()/df['close'].rolling(20).mean()
    hl=df['high']-df['low']; hc=np.abs(df['high']-df['close'].shift()); lc=np.abs(df['low']-df['close'].shift())
    df['atr']=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']=df['close'].pct_change(5)*100; df['price_change_10d']=df['close'].pct_change(10)*100
    df['price_change_20d']=df['close'].pct_change(20)*100
    df['ma50_slope']=df['sma_50'].diff(5)/df['sma_50'].shift(5)*100
    df['future_return']=df['close'].shift(-5)/df['close']-1
    df['target']=(df['future_return']>0.02).astype(int)
    return df.bfill().ffill()

results = []
for code, name, sfx in STOCKS:
    ticker = f"{code}.{sfx}"; sl = sfx.lower()
    print(f"\n{'='*50}\n  {code} {name} ({ticker}) — XGBoost only\n{'='*50}")
    df = yf.download(ticker, start='2015-01-01', end=END_DATE, progress=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
    if df.empty or len(df) < 50: print(f"  [SKIP] {len(df)} rows"); continue
    df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"  Data: {len(df)} days")
    df = build_features(df)
    dc = df.dropna(subset=FEAT+['target'])
    X=dc[FEAT]; y=dc['target']
    Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.2,shuffle=False)
    xm=xgb.XGBClassifier(max_depth=5,learning_rate=0.05,n_estimators=200,min_child_weight=3,
                          subsample=0.8,colsample_bytree=0.8,objective='binary:logistic',random_state=42,eval_metric='logloss')
    xm.fit(Xt,yt)
    xgb_acc=accuracy_score(ye,xm.predict(Xe))
    joblib.dump(xm,f'xgb_{code}_{sl}_model.pkl')
    d={'symbol':code,'ticker':ticker,'model_type':'XGBoost',
       'training_accuracy':round(accuracy_score(yt,xm.predict(Xt))*100,2),
       'validation_accuracy':round(xgb_acc*100,2),'backtest_accuracy':round(xgb_acc*100,2),
       'last_updated':datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(d,open(f'model_accuracy_{code}_{sfx}.json','w',encoding='utf-8'),indent=2)
    json.dump(d,open(f'model_accuracy_{code}.json','w',encoding='utf-8'),indent=2)
    print(f"  [XGBoost] {xgb_acc*100:.2f}%")
    results.append((code, name, xgb_acc))

print(f"\n{'='*50}\n  DONE\n{'='*50}")
for code, name, acc in results:
    print(f"  {code} {name}: XGBoost {acc*100:.2f}%")
print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
