"""


7734.TWO (印能科技) 多策略整合信號生成器


Ensemble: ARIMA + LSTM + ElasticNet + LASSO + LR  (62.50% accuracy)


Random Forest excluded (37.5% — below chance)


"""


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import sys, io, json, warnings


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


warnings.filterwarnings('ignore')


import numpy as np


import pandas as pd


import yfinance as yf


import joblib


from sklearn.preprocessing import MinMaxScaler


from statsmodels.tsa.arima.model import ARIMA


from datetime import datetime
from tw_news_tracker import print_tavily_news_tw


TICKER = '7734.TWO'


def add_indicators(df):


    df['sma_10'] = df['close'].rolling(10).mean()


    df['sma_30'] = df['close'].rolling(30).mean()


    df['ema_12'] = df['close'].ewm(span=12).mean()


    df['ema_26'] = df['close'].ewm(span=26).mean()


    d = df['close'].diff()


    g = d.where(d > 0, 0).rolling(14).mean()


    l = (-d.where(d < 0, 0)).rolling(14).mean()


    df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))


    df['macd']        = df['ema_12'] - df['ema_26']


    df['macd_signal'] = df['macd'].ewm(span=9).mean()


    df['macd_hist']   = df['macd'] - df['macd_signal']


    df['bb_mid']  = df['close'].rolling(20).mean()


    df['bb_std']  = df['close'].rolling(20).std()


    df['bb_pos']  = ((df['close'] - (df['bb_mid'] - 2*df['bb_std'])) /


                     (4*df['bb_std'] + 1e-10) * 100).fillna(50)


    lo14 = df['low'].rolling(14).min()


    hi14 = df['high'].rolling(14).max()


    df['K']         = ((df['close'] - lo14) / (hi14 - lo14 + 1e-10) * 100).fillna(50)


    df['D']         = df['K'].rolling(3).mean()


    df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)


    df['ret_5d']    = df['close'].pct_change(5) * 100


    df['ret_10d']   = df['close'].pct_change(10) * 100


    df['ret_20d']   = df['close'].pct_change(20) * 100


    return df.bfill().ffill()


def get_trading_signal():


    print(f"🤖 {TICKER} (印能科技) | Ensemble 62.5% | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


    print("   Models: ARIMA + LSTM + ElasticNet + LASSO + LR")


    print("⚠️  Limited history — treat signals as low-confidence")


    print("=" * 80)


    # Load config & models


    try:


        with open('ensemble_7734_config.json', encoding='utf-8') as f:


            cfg = json.load(f)


        scaler = joblib.load('ensemble_7734_scaler.pkl')


        lr     = joblib.load('ensemble_7734_lr.pkl')


        lasso  = joblib.load('ensemble_7734_lasso.pkl')


        en     = joblib.load('ensemble_7734_en.pkl')


        print("✅ Sklearn models loaded")


    except Exception as e:


        print(f"❌ Model load failed: {e}"); return None


    try:


        import tensorflow as tf


        lstm_model = tf.keras.models.load_model('lstm_7734_two_model.keras')


        print("✅ LSTM model loaded")


    except Exception as e:


        print(f"⚠️  LSTM unavailable: {e}")


        lstm_model = None


    # Download data


    print(f"\n📊 下載 {TICKER} 最新數據...")


    try:


        df = yf.download(TICKER, period='1y', progress=False)


        if df.empty:


            print("❌ 無法獲取數據"); return None


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        df = df.rename(columns={'Close':'close','Volume':'volume',


                                'Open':'open','High':'high','Low':'low'}).reset_index()


        print(f"✅ 成功下載 {len(df)} 天數據")


    except Exception as e:


        print(f"❌ 數據下載失敗: {e}"); return None


    df = add_indicators(df)


    FEATURES  = cfg['features']


    LSTM_FEAT = cfg['lstm_features']


    LOOKBACK  = cfg['lookback']


    THRESHOLD = cfg['threshold']     # ensemble vote threshold


    LSTM_THR  = cfg['lstm_threshold']


    WEIGHTS   = cfg['weights']


    df_clean = df.dropna(subset=FEATURES)


    if len(df_clean) < LOOKBACK + 1:


        print("❌ 數據不足"); return None


    latest        = df_clean.iloc[-1]


    prev_close    = float(df_clean['close'].iloc[-2])


    current_price = float(latest['close'])


    price_change_pct = (current_price - prev_close) / prev_close * 100


    X_latest = scaler.transform(latest[FEATURES].values.reshape(1, -1))


    # ── Individual model votes ──


    votes = {}


    # LR


    lr_prob = float(np.clip(lr.predict(X_latest)[0], 0, 1))


    votes['LR'] = (1 if lr_prob >= 0.5 else 0, lr_prob)


    # LASSO


    lasso_prob = float(np.clip(lasso.predict(X_latest)[0], 0, 1))


    votes['LASSO'] = (1 if lasso_prob >= 0.5 else 0, lasso_prob)


    # EN


    en_prob = float(np.clip(en.predict(X_latest)[0], 0, 1))


    votes['EN'] = (1 if en_prob >= 0.5 else 0, en_prob)


    # ARIMA (walk-forward on last 30 days)


    arima_vote = 0


    try:


        log_close = np.log(df_clean['close'].values[-60:])


        am = ARIMA(log_close, order=(1, 1, 1)).fit()


        fc = am.forecast(steps=5)[-1]


        arima_vote = 1 if fc > log_close[-1] else 0


        votes['ARIMA'] = (arima_vote, float(arima_vote))


    except Exception:


        votes['ARIMA'] = (0, 0.5)


    # LSTM


    if lstm_model is not None:


        mm = MinMaxScaler()


        scaled = mm.fit_transform(df_clean[LSTM_FEAT].values)


        seq = scaled[-LOOKBACK:].reshape(1, LOOKBACK, len(LSTM_FEAT))


        lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])


        votes['LSTM'] = (1 if lstm_prob >= LSTM_THR else 0, lstm_prob)


    else:


        votes['LSTM'] = (0, 0.5)


    # ── Weighted vote ──


    vote_sum = sum(WEIGHTS.get(k, 0) * v[0] for k, v in votes.items())


    signal   = 'BUY' if vote_sum >= THRESHOLD else 'HOLD/WAIT'


    # Technical indicators


    rsi       = float(latest['rsi'])


    macd      = float(latest['macd'])


    ms        = float(latest['macd_signal'])


    s10       = float(latest['sma_10'])


    s30       = float(latest['sma_30'])


    vol_ratio = float(latest['vol_ratio'])


    avg_vol   = float(df_clean['volume'].tail(20).mean())


    candle_dir = 'up' if current_price > prev_close else 'down' if current_price < prev_close else 'flat'


    print("\n" + "=" * 80)


    print("📊 技術指標")


    print("=" * 80)


    print(f"當前價格:        NT${current_price:.2f}")


    print(f"今日漲跌幅:      {price_change_pct:+.2f}%")


    print(f"RSI (14):        {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")


    print(f"MACD:            {macd:.4f}  {'[金叉]' if macd > ms else '[死叉]'}")


    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  {'[多頭]' if s10 > s30 else '[空頭]'}")


    print(f"成交量:          {int(latest['volume']):,}")


    print(f"量比:            {vol_ratio:.2f}x  {'[放量]' if vol_ratio > 1.5 else '[縮量]' if vol_ratio < 0.7 else '[正常]'}")


    print(f"量價方向:        {'價漲量增' if candle_dir == 'up' and vol_ratio >= 1.2 else '價跌量增' if candle_dir == 'down' and vol_ratio >= 1.2 else '中性'}")


    print("\n" + "=" * 80)


    print("🤖 各模型投票結果")


    print("=" * 80)


    for name, (vote, prob) in votes.items():


        w = WEIGHTS.get(name, 0)


        bar = '🟢 BUY ' if vote == 1 else '🔴 HOLD'


        print(f"  {name:<8} {bar}  prob={prob*100:>5.1f}%  weight={w*100:.1f}%")


    print(f"  {'─'*44}")


    print(f"  加權票數: {vote_sum:.3f} / {THRESHOLD:.2f} (門檻)")


    print("\n" + "=" * 80)


    print("🎯 整合信號")


    print("=" * 80)


    if signal == 'BUY':


        print("🟢 買入信號 (BUY)")


        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")


        print(f"   止損參考:    NT${current_price*0.95:.2f} (-5%)")


    else:


        print("🔴 不建議買入 (HOLD/WAIT)")


        print(f"   加權票數不足: {vote_sum:.3f} < {THRESHOLD:.2f}")


    print("=" * 80)


    print("⚠️  印能科技歷史數據有限 (~484天)，信號僅供參考")


    return {
            'vote_sum': vote_sum, 'rsi': rsi, 'macd': macd}


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 7734 印能科技 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('7734', '印能科技', max_results=5)