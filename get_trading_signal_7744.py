"""


7744.TWO (致伸) XGBoost 交易信號生成器


"""


TICKER = '7744.TWO'


MODEL_FILE = 'xgb_7744_two_model.pkl'


NAME = '致伸'


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


import sys, io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np, pandas as pd, yfinance as yf, joblib


from datetime import datetime
from tw_news_tracker import print_tavily_news_tw


try:


    from model_accuracy_tracker import get_model_accuracy_display


    _acc = get_model_accuracy_display(TICKER)


except Exception:


    _acc = 'N/A'


FEATURE_COLUMNS = [


    'rsi','macd','macd_signal','macd_hist','bb_position','K','D',


    'obv','obv_ma20','sma_10','sma_30','sma_50','sma_200',


    'volatility','atr','price_change_5d','price_change_10d','price_change_20d','ma50_slope'


]


def add_technical_indicators(df):


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


    df['bb_middle']   = df['close'].rolling(20).mean()


    df['bb_std']      = df['close'].rolling(20).std()


    df['bb_upper']    = df['bb_middle'] + 2 * df['bb_std']


    df['bb_lower']    = df['bb_middle'] - 2 * df['bb_std']


    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)


    lo14 = df['low'].rolling(14).min()


    hi14 = df['high'].rolling(14).max()


    df['K']          = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)


    df['D']          = df['K'].rolling(3).mean()


    df['obv']        = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()


    df['obv_ma20']   = df['obv'].rolling(20).mean()


    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()


    hl = df['high'] - df['low']


    hc = np.abs(df['high'] - df['close'].shift())


    lc = np.abs(df['low']  - df['close'].shift())


    df['atr']              = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()


    df['price_change_5d']  = df['close'].pct_change(5)  * 100


    df['price_change_10d'] = df['close'].pct_change(10) * 100


    df['price_change_20d'] = df['close'].pct_change(20) * 100


    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100


    return df.bfill().ffill()


def get_trading_signal():


    print("=" * 80)


    print(f"🤖 {TICKER} ({NAME}) AI 交易信號生成器")


    print(f"模型準確度: {_acc}")


    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    print("=" * 80)


    try:


        model = joblib.load(MODEL_FILE)


        print(f"✅ 模型加載成功: {MODEL_FILE}")


    except Exception as e:


        print(f"❌ 模型加載失敗: {e}"); return None


    print(f"\n📊 下載 {TICKER} 最新數據...")


    try:


        df = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)


        if df.empty: print("❌ 無法獲取數據"); return None


        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)


        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()


        print(f"✅ 成功下載 {len(df)} 天數據")


    except Exception as e:


        print(f"❌ 數據下載失敗: {e}"); return None


    df = add_technical_indicators(df)


    latest = df.iloc[-1]


    current_price = float(latest['close'])


    rsi  = float(latest['rsi'])


    macd = float(latest['macd'])


    ms   = float(latest['macd_signal'])


    s10  = float(latest['sma_10'])


    s30  = float(latest['sma_30'])


    try:


        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())


    except Exception:


        latest_date = datetime.now().strftime('%Y-%m-%d')


    X     = latest[FEATURE_COLUMNS].values.reshape(1, -1)


    proba = model.predict_proba(X)[0]


    pred  = model.predict(X)[0]


    buy_prob = proba[1] * 100


    print("\n" + "=" * 80)


    print("📊 技術指標分析")


    print("=" * 80)


    print(f"當前價格: NT${current_price:.2f}")


    print(f"RSI: {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")


    print(f"MACD: {macd:.4f}  {'[金叉]' if macd > ms else '[死叉]'}")


    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  {'[多頭]' if s10 > s30 else '[空頭]'}")


    print(f"布林帶位置: {float(latest['bb_position']):.1f}%")


    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")


    print("\n" + "=" * 80)


    print("🎯 AI 交易信號")


    print("=" * 80)


    print(f"今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")


    if pred == 1 and buy_prob >= 60:


        print("🟢 買入信號 (BUY)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")


        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")


        print(f"   止損位: NT${current_price*0.95:.2f} (-5%)")


    elif pred == 1 and buy_prob >= 52:


        print("🟡 弱買入 (HOLD - WEAK BUY)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（謹慎操作）")


    elif pred == 0 and buy_prob <= 35:


        print("🔴 賣出觀望 (SELL)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心偏低）")


    else:


        print("🟡 持有觀望 (HOLD)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心不足）")


    print("=" * 80)


    print(f"\n📱 快速摘要:")


    print(f"   股票: {TICKER} ({NAME})")


    print(f"   日期: {latest_date}")


    print(f"   價格: NT${current_price:.2f}")


    if pred == 1 and buy_prob >= 60:


        print("   信號: 🟢 買入信號 (BUY)")


    elif pred == 1 and buy_prob >= 52:


        print("   信號: 🟡 持有 (HOLD)")


    elif pred == 0 and buy_prob <= 35:


        print("   信號: 🔴 賣出 (SELL)")


    else:


        print("   信號: 🟡 持有 (HOLD)")


    print(f"   {_acc}")


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 7744 致伸 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('7744', '致伸', max_results=5)