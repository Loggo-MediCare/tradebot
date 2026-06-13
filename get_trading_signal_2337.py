"""台股 2337 (旺宏) — XGBoost 59.28% / PPO 63.57%"""
import os,sys,io,warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8'); warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,yfinance as yf,joblib,gymnasium as gym
from gymnasium import spaces; from stable_baselines3 import PPO; from datetime import datetime
from model_accuracy_tracker import get_model_accuracy_display,should_mute_ai_signal
from ppo_backtest_cache import format_ppo_roi_line
from tw_news_tracker import print_tavily_news_tw
CODE='2337';TICKER='2337.TW';NAME='旺宏';XGB_ACC=59.28;PPO_ACC=63.57
XGB_MODEL=f'xgb_{CODE}_tw_model.pkl';PPO_MODEL=f'ppo_{CODE}_tw_improved'
FEAT=['rsi','macd','macd_signal','macd_hist','bb_position','K','D','obv','obv_ma20',
      'sma_10','sma_30','sma_50','volatility','atr','price_change_5d','price_change_10d','price_change_20d','ma50_slope']
class TradingEnv(gym.Env):
    def __init__(self,df):
        super().__init__();self.df=df.reset_index(drop=True)
        self.action_space=spaces.Box(low=-1.,high=1.,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(low=-np.inf,high=np.inf,shape=(15,),dtype=np.float32);self.reset()
    def reset(self,seed=None,options=None):
        super().reset(seed=seed);self.i=0;self.bal=10000.;self.sh=0;self.profit=0.;return self._obs(),{}
    def _obs(self):
        r=self.df.iloc[self.i];p=float(r['close']);tv=self.bal+self.sh*p
        return np.array([float(self.sh),float(self.bal),p,float(r.get('sma_10',0)),float(r.get('sma_30',0)),
            float(r.get('sma_50',0)),float(r.get('rsi',50)),float(r.get('macd',0)),float(r.get('macd_signal',0)),
            float(r.get('bb_u',0)),float(r.get('bb_l',0)),float(r.get('volume',0)),float(self.profit),
            (self.sh*p)/tv if tv>0 else 0,self.bal/tv if tv>0 else 1],dtype=np.float32)
    def step(self,action):
        a=float(action[0]) if isinstance(action,np.ndarray) else float(action);a=np.clip(a,-1,1)
        p=float(self.df.iloc[self.i]['close'])
        if a<-0.1:
            s=int(self.sh*abs(a))
            if s>0:self.bal+=s*p;self.sh-=s
        elif a>0.1:
            s=int((self.bal//p)*a)
            if s>0:self.bal-=s*p;self.sh+=s
        self.profit=(self.bal+self.sh*p)-10000.;self.i+=1;done=self.i>=len(self.df)-1
        return self._obs(),self.profit/10000.+(0.01 if abs(a)>0.1 else 0),done,False,{}
def build_features(df):
    df=df.copy();df['sma_10']=df['close'].rolling(10).mean();df['sma_30']=df['close'].rolling(30).mean()
    df['sma_50']=df['close'].rolling(50).mean();df['ema_12']=df['close'].ewm(span=12).mean();df['ema_26']=df['close'].ewm(span=26).mean()
    d=df['close'].diff();g=d.where(d>0,0).rolling(14).mean();l=(-d.where(d<0,0)).rolling(14).mean()
    df['rsi']=100-(100/(1+g/(l+1e-10)));df['macd']=df['ema_12']-df['ema_26']
    df['macd_signal']=df['macd'].ewm(span=9).mean();df['macd_hist']=df['macd']-df['macd_signal']
    df['bb_m']=df['close'].rolling(20).mean();df['bb_s']=df['close'].rolling(20).std()
    df['bb_u']=df['bb_m']+2*df['bb_s'];df['bb_l']=df['bb_m']-2*df['bb_s']
    df['bb_position']=((df['close']-df['bb_l'])/(df['bb_u']-df['bb_l'])*100).fillna(50)
    lo14=df['low'].rolling(14).min();hi14=df['high'].rolling(14).max()
    df['K']=((df['close']-lo14)/(hi14-lo14)*100).fillna(50);df['D']=df['K'].rolling(3).mean()
    df['obv']=(np.sign(df['close'].diff())*df['volume']).fillna(0).cumsum();df['obv_ma20']=df['obv'].rolling(20).mean()
    df['volatility']=df['close'].rolling(20).std()/df['close'].rolling(20).mean()
    hl=df['high']-df['low'];hc=np.abs(df['high']-df['close'].shift());lc=np.abs(df['low']-df['close'].shift())
    df['atr']=pd.concat([hl,hc,lc],axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']=df['close'].pct_change(5)*100;df['price_change_10d']=df['close'].pct_change(10)*100
    df['price_change_20d']=df['close'].pct_change(20)*100;df['ma50_slope']=df['sma_50'].diff(5)/df['sma_50'].shift(5)*100
    return df.bfill().ffill()
def get_trading_signal():
    print(f"🤖 {CODE}({NAME}) | {get_model_accuracy_display(CODE)} | {datetime.now().strftime('%Y-%m-%d %H:%M')}");print("="*80)
    df_raw=yf.download(TICKER,period='300d',progress=False,auto_adjust=True)
    if df_raw.empty:print("❌ 無法取得數據");return None
    if isinstance(df_raw.columns,pd.MultiIndex):df_raw.columns=df_raw.columns.droplevel(1)
    df_raw=df_raw.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
    print(f"✅ {len(df_raw)}天數據")
    target_price=None
    try:
        info=yf.Ticker(TICKER).info;target_price=info.get('targetMeanPrice');target_high=info.get('targetHighPrice')
        rec_mean=info.get('recommendationMean');rec_key=info.get('recommendationKey','');n=info.get('numberOfAnalystOpinions',0)
        if target_price and n>0:print(f"   📊 目標價 NT${target_price:.2f}(平均)/NT${target_high:.2f}(最高)  評級:{rec_key}({rec_mean:.1f}/5,{n}位)")
    except:pass
    df=build_features(df_raw);row=df.iloc[-1]
    try:latest_date=str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except:latest_date=datetime.now().strftime('%Y-%m-%d')
    cp=float(row['close']);prev=float(df['close'].iloc[-2]) if len(df)>1 else cp
    chg=(cp-prev)/prev*100 if prev>0 else 0.
    rsi=float(row['rsi']);rsi_p=float(df['rsi'].iloc[-2]) if len(df)>1 else rsi
    macd=float(row['macd']);msig=float(row['macd_signal'])
    s10=float(row['sma_10']);s30=float(row['sma_30']);s50=float(row['sma_50'])
    bbu=float(row['bb_u']);bbl=float(row['bb_l']);bbp=float(row['bb_position'])
    vol=float(row['volume']);avgvol=float(df['volume'].tail(20).mean());vr=vol/avgvol if avgvol>0 else 1.
    obv=float(row['obv']);obvm=float(row['obv_ma20']);slope=float(row['ma50_slope'])
    print(f"\n日期:{latest_date}  現價:NT${cp:.2f}({chg:+.2f}%)  量比:{vr:.2f}x")
    print("\n"+"="*80+"\n📊 技術指標\n"+"="*80)
    print(f"RSI(14):{rsi:.2f}{'[超買]'if rsi>75 else'[超賣]'if rsi<30 else'[中性]'}  {rsi_p:.2f}→{rsi:.2f}({rsi-rsi_p:+.2f})")
    print(f"MACD:{macd:.4f}{'[金叉]'if macd>msig else'[死叉]'}  Signal:{msig:.4f}")
    ma_tr='🟢多頭'if s10>s30>s50 else('🔴空頭'if s10<s30<s50 else'🟡混合')
    print(f"SMA 10/30/50:NT${s10:.2f}/{s30:.2f}/{s50:.2f}  {ma_tr}  MA50斜率:{slope:+.4f}%")
    print(f"布林帶:{bbp:.1f}%  上軌NT${bbu:.2f} 下軌NT${bbl:.2f}  OBV:{'多頭'if obv>obvm else'空頭'}")
    if target_price:print(f"分析師目標:NT${target_price:.2f}  空間:{(target_price-cp)/cp*100:+.1f}%")
    print("\n"+"="*80+f"\n🧠XGBoost({XGB_ACC:.2f}%) + 🤖PPO({PPO_ACC:.2f}%)\n"+"="*80)
    xgb_prob=None
    try:
        xm=joblib.load(XGB_MODEL);fr=row[FEAT].values.reshape(1,-1)
        xgb_prob=float(xm.predict_proba(fr)[0][1]);xgb_pred=int(xm.predict(fr)[0])
        print(f"今日買入機率 P(buy): {xgb_prob*100:.1f}%（P(not buy): {(1-xgb_prob)*100:.1f}%）  {'看多📈'if xgb_pred==1 else'看空📉'}")
    except Exception as e:print(f"⚠️XGBoost:{e}")
    ppo_action=0.
    try:
        pm=PPO.load(PPO_MODEL);env=TradingEnv(df);env.i=len(df)-1
        act,_=pm.predict(env._obs(),deterministic=True)
        ppo_action=float(act[0]) if isinstance(act,np.ndarray) else float(act)
        if should_mute_ai_signal(TICKER,threshold=52):ppo_action=0.;print("⚠️PPO靜音")
        else:print(format_ppo_roi_line(CODE,TICKER,PPO_MODEL,df,ppo_action))
    except Exception as e:print(f"⚠️PPO:{e}")
    score=50.;reasons=[];warns=[]
    if xgb_prob is not None:
        score+=(xgb_prob-0.5)*60
        if xgb_prob>0.6:reasons.append(f"XGBoost{xgb_prob*100:.1f}%")
        elif xgb_prob<0.4:warns.append(f"XGBoost看空{xgb_prob*100:.1f}%")
    if ppo_action>0.1:score+=ppo_action*15;reasons.append(f"PPO買入{ppo_action:+.2f}")
    elif ppo_action<-0.1:score+=ppo_action*15;warns.append(f"PPO賣出{ppo_action:+.2f}")
    if macd>msig:score+=5;reasons.append("MACD金叉")
    else:score-=5;warns.append("MACD死叉")
    if s10>s30:score+=5;reasons.append("均線多頭")
    else:score-=5;warns.append("均線空頭")
    if rsi<35:score+=8;reasons.append(f"RSI超賣{rsi:.1f}")
    elif rsi>75:score-=8;warns.append(f"RSI超買{rsi:.1f}")
    if slope>0.05:score+=4;reasons.append("MA50上升")
    elif slope<-0.05:score-=4;warns.append("MA50下降")
    if obv>obvm:score+=3;reasons.append("OBV多頭")
    else:score-=3
    if target_price:
        up=(target_price-cp)/cp*100
        if up>15:score+=5;reasons.append(f"目標價空間{up:.1f}%")
        elif up<-5:score-=5;warns.append(f"超越目標價{up:.1f}%")
    if bbp<20:score+=5;reasons.append(f"接近下軌{bbp:.0f}%")
    elif bbp>85:score-=5;warns.append(f"接近上軌{bbp:.0f}%")
    score=max(0,min(100,score))
    if score>=65:signal="買入(BUY)";em="🟢"
    elif score<=35:signal="賣出(SELL)";em="🔴"
    else:signal="持有(HOLD)";em="🟡"
    print(f"\n{em} {signal}  評分:{score:.0f}/100")
    if warns:print("  ⚠️ "+", ".join(warns))
    if reasons:print("  📌 "+", ".join(reasons))
    if score>=65:print(f"  💡 買入NT${cp*0.995:.2f}~{cp:.2f}  止損NT${cp*0.95:.2f}")
    elif score<=35:print(f"  💡 支撐NT${bbl:.2f}")
    else:print(f"  💡 觀望  支撐NT${bbl:.2f} 壓力NT${bbu:.2f}")
    print(f"\n📱 {TICKER}({NAME}) {latest_date} NT${cp:.2f}({chg:+.2f}%) {signal} {score:.0f}/100  {get_model_accuracy_display(CODE)}")
    return{'date':latest_date,'symbol':TICKER,'name':NAME,'current_price':cp,'price_change_pct':chg,
           'signal':signal,'score':score,'xgb_prob':xgb_prob,'ppo_action':ppo_action,'rsi':rsi,'macd':macd}
if __name__=='__main__':
    get_trading_signal()
    print('\n'+'='*80+f'\n🌐 {CODE} {NAME} 即時新聞\n'+'='*80)
    print_tavily_news_tw(CODE,NAME,max_results=5)
