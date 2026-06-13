"""


8917.TWO (欣泰) 交易信號生成器


使用 XGBoost 模型預測交易信號


"""


import sys


import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


import pandas as pd


import yfinance as yf


import numpy as np


import joblib


from datetime import datetime


from model_accuracy_tracker import get_model_accuracy_display
from tw_news_tracker import print_tavily_news_tw


TICKER = '8917.TWO'


MODEL_FILE = 'xgb_8917_model.pkl'


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


def main():


    try:


        accuracy_display = get_model_accuracy_display(TICKER)


        print(f"🤖 {TICKER} (欣泰) | 準確度: {accuracy_display} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")


        print("=" * 100)


        # 載入模型


        try:


            model = joblib.load(MODEL_FILE)


        except FileNotFoundError:


            print(f"❌ 找不到模型文件: {MODEL_FILE}")


            return


        print(f"\n📊 下載 {TICKER} 最新數據...")


        try:


            df = yf.download(TICKER, period='1y', progress=False)


            if df.empty:


                print(f"❌ 無法下載 {TICKER} 數據")


                return


            if isinstance(df.columns, pd.MultiIndex):


                df.columns = df.columns.droplevel(1)


            df = df.rename(columns={'Close': 'close', 'Volume': 'volume',


                                    'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()


            df = add_technical_indicators(df)


            feature_columns = [


                'rsi', 'macd', 'macd_signal', 'macd_hist',


                'bb_position', 'K', 'D', 'obv', 'obv_ma20',


                'sma_10', 'sma_30', 'sma_50', 'sma_200',


                'volatility', 'atr',


                'price_change_5d', 'price_change_10d', 'price_change_20d',


                'ma50_slope'


            ]


            df_clean = df.dropna(subset=feature_columns)


            if len(df_clean) == 0:


                print("❌ 數據不足，無法生成信號")


                return


            latest_data = df_clean.iloc[-1]


            X = latest_data[feature_columns].values.reshape(1, -1)


            prediction_proba = model.predict_proba(X)[0]


            prediction = model.predict(X)[0]


            print(f"\n📈 最新股價: ${latest_data['close']:.2f}")


            print(f"📅 數據日期: {latest_data['Date']}")


            print(f"\n🎯 模型預測:")


            print(f"   上漲概率: {prediction_proba[1]*100:.2f}%")


            print(f"   下跌概率: {prediction_proba[0]*100:.2f}%")


            if prediction == 1:


                signal = "🟢 買入信號"


                confidence = prediction_proba[1]


            else:


                signal = "🔴 賣出信號"


                confidence = prediction_proba[0]


            print(f"\n💡 交易信號: {signal}")


            print(f"   信心指數: {confidence*100:.2f}%")


            print("\n" + "=" * 100)


            print("✅ 信號生成完成")


        except Exception as e:


            print(f"❌ 數據處理錯誤: {e}")


            import traceback


            traceback.print_exc()


    except Exception as e:


        print(f"❌ 錯誤: {e}")


        import traceback


        traceback.print_exc()


if __name__ == "__main__":


    main()



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 8917 欣泰 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('8917', '欣泰', max_results=5)