"""


6994.TW (環科) XGBoost 交易信號生成器


"""


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import sys


import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np


import pandas as pd


import yfinance as yf


import joblib


from datetime import datetime


from model_accuracy_tracker import get_model_accuracy_display
from tw_news_tracker import print_tavily_news_tw


TICKER = '6994.TW'


MODEL_FILE = 'xgb_6994_tw_model.pkl'


def add_technical_indicators(df):


    """添加技術指標"""


    df['sma_10'] = df['close'].rolling(10).mean()


    df['sma_30'] = df['close'].rolling(30).mean()


    df['sma_50'] = df['close'].rolling(50).mean()


    df['sma_200'] = df['close'].rolling(200).mean()


    df['ema_12'] = df['close'].ewm(span=12).mean()


    df['ema_26'] = df['close'].ewm(span=26).mean()


    delta = df['close'].diff()


    gain = delta.where(delta > 0, 0).rolling(14).mean()


    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()


    rs = gain / (loss + 1e-10)


    df['rsi'] = 100 - (100 / (1 + rs))


    df['macd'] = df['ema_12'] - df['ema_26']


    df['macd_signal'] = df['macd'].ewm(span=9).mean()


    df['macd_hist'] = df['macd'] - df['macd_signal']


    df['bb_middle'] = df['close'].rolling(20).mean()


    df['bb_std'] = df['close'].rolling(20).std()


    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)


    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)


    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)


    low_14 = df['low'].rolling(14).min()


    high_14 = df['high'].rolling(14).max()


    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)


    df['D'] = df['K'].rolling(3).mean()


    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()


    df['obv_ma20'] = df['obv'].rolling(20).mean()


    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()


    high_low = df['high'] - df['low']


    high_close = np.abs(df['high'] - df['close'].shift())


    low_close = np.abs(df['low'] - df['close'].shift())


    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)


    df['atr'] = true_range.rolling(14).mean()


    df['price_change_5d'] = df['close'].pct_change(5) * 100


    df['price_change_10d'] = df['close'].pct_change(10) * 100


    df['price_change_20d'] = df['close'].pct_change(20) * 100


    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100


    return df.bfill().ffill()


def get_trading_signal():


    """生成今日交易信號"""


    # 壓縮標題區塊


    accuracy_display = get_model_accuracy_display(TICKER)


    print(f"🤖 {TICKER} (環科) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


    print("=" * 80)


    # 1. 加載 XGBoost 模型


    try:


        model = joblib.load(MODEL_FILE)


        print(f"✅ 模型加載成功: {MODEL_FILE}")


    except Exception as e:


        print(f"❌ 模型加載失敗: {e}")


        return None


    # 2. 下載最新數據


    print(f"\n📊 下載 {TICKER} 最新數據...")


    try:


        df = yf.download(TICKER, period='1y', progress=False)


        if df.empty:


            print("❌ 無法獲取數據")


            return None


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        df = df.rename(columns={


            'Close': 'close', 'Volume': 'volume',


            'Open': 'open', 'High': 'high', 'Low': 'low'


        }).reset_index()


        print(f"✅ 成功下載 {len(df)} 天數據")


    except Exception as e:


        print(f"❌ 數據下載失敗: {e}")


        return None


    # 3. 添加技術指標


    df = add_technical_indicators(df)


    # 4. 獲取最新數據


    latest = df.iloc[-1]


    current_price = float(latest['close'])


    feature_columns = [


        'rsi', 'macd', 'macd_signal', 'macd_hist',


        'bb_position', 'K', 'D', 'obv', 'obv_ma20',


        'sma_10', 'sma_30', 'sma_50', 'sma_200',


        'volatility', 'atr',


        'price_change_5d', 'price_change_10d', 'price_change_20d',


        'ma50_slope'


    ]


    X = latest[feature_columns].values.reshape(1, -1)


    # 5. 模型預測


    print("\n🧠 AI 模型分析中...")


    prediction_proba = model.predict_proba(X)[0]


    prediction = model.predict(X)[0]


    buy_prob = prediction_proba[1] * 100


    sell_prob = prediction_proba[0] * 100


    print(f"預測結果: {'買入機率' if prediction == 1 else '不買入'} — 今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {sell_prob:.1f}%）")


    # 6. 技術指標分析


    print("\n" + "=" * 80)


    print("📊 技術指標")


    print("=" * 80)


    print(f"當前價格: NT${current_price:.2f}")


    print(f"RSI: {float(latest['rsi']):.2f} {'[超買]' if float(latest['rsi']) > 70 else '[超賣]' if float(latest['rsi']) < 30 else '[中性]'}")


    print(f"MACD: {float(latest['macd']):.4f} {'[金叉]' if float(latest['macd']) > float(latest['macd_signal']) else '[死叉]'}")


    print(f"均線: SMA10={float(latest['sma_10']):.2f}, SMA30={float(latest['sma_30']):.2f} {'[多頭]' if float(latest['sma_10']) > float(latest['sma_30']) else '[空頭]'}")


    print(f"布林帶位置: {float(latest['bb_position']):.1f}%")


    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")


    # 7. 交易建議


    print("\n" + "=" * 80)


    print("🎯 交易信號")


    print("=" * 80)


    if prediction == 1 and buy_prob >= 60:


        print("🟢 買入信號 (BUY)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")


        print(f"   建議買入價格: NT${current_price * 0.995:.2f} - NT${current_price * 1.005:.2f}")


    elif prediction == 1 and buy_prob >= 52:


        print("🟡 弱買入信號 (WEAK BUY)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）")


        print(f"   建議謹慎買入或觀望")


    else:


        print("🔴 不建議買入 (HOLD/WAIT)")


        print(f"   今日買入機率 P(buy): {buy_prob:.1f}%（P(not buy): {100-buy_prob:.1f}%）（信心不足）")


    print("=" * 80)


    return {
        'ticker': TICKER,


        'price': current_price,


        'prediction': prediction,


        'buy_probability': buy_prob,


        'rsi': float(latest['rsi']),


        'macd': float(latest['macd'])


    }


if __name__ == "__main__":


    get_trading_signal()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 6994 環科 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('6994', '環科', max_results=5)