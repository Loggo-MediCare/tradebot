"""Generate XGBoost signal scripts for all new stocks."""
import os

STOCKS = [
    ('3189', '3189.TW',  '景碩',    'tw'),
    ('6234', '6234.TWO', '高僑',    'two'),
    ('6488', '6488.TWO', '環球晶',  'two'),
    ('6207', '6207.TWO', '雷科',    'two'),
    ('6937', '6937.TW',  '天虹',    'tw'),
    ('8103', '8103.TW',  '瀚荃',    'tw'),
    ('1569', '1569.TWO', '濱川',    'two'),
    ('1595', '1595.TWO', '川寶',    'two'),
    ('6108', '6108.TW',  '競國',    'tw'),
    ('4951', '4951.TWO', '精拓科',  'two'),
    ('1727', '1727.TW',  '中華化',  'tw'),
    ('2486', '2486.TW',  '一詮',    'tw'),
    ('6138', '6138.TWO', '茂達',    'two'),
    ('8027', '8027.TWO', '鈦昇',    'two'),
    ('5351', '5351.TWO', '鈺創',    'two'),
    ('4720', '4720.TW',  '德淵',    'tw'),
    ('6176', '6176.TW',  '瑞儀',    'tw'),
    ('3380', '3380.TW',  '明泰',    'tw'),
    ('6672', '6672.TW',  '騰輝電子KY','tw'),
    ('6213', '6213.TW',  '聯茂',    'tw'),
    ('7734', '7734.TWO', '印能科技', 'two'),
    ('7751', '7751.TWO', '竑騰',    'two'),
    ('3583', '3583.TW',  '辛耘',    'tw'),
    ('8028', '8028.TW',  '昇陽半導體','tw'),
    ('3680', '3680.TWO', '家登',    'two'),
    ('4772', '4772.TWO', '台特化',  'two'),
    ('6788', '6788.TWO', '華景電',  'two'),
    ('7703', '7703.TWO', '銳澤',    'two'),
    ('8147', '8147.TWO', '大綜',    'two'),
    ('2404', '2404.TW',  '漢唐',    'tw'),
    ('6196', '6196.TW',  '帆宣',    'tw'),
    ('6605', '6605.TW',  '信紘科',  'tw'),
    ('6139', '6139.TW',  '亞翔',    'tw'),
    ('8071', '8071.TWO', '竹陸科技','two'),
    ('1560', '1560.TW',  '中砂',    'tw'),
    ('6438', '6438.TW',  '迅得',    'tw'),
    ('6449', '6449.TW',  '倍利科',  'tw'),
]

BODY = """\
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, yfinance as yf, joblib
from datetime import datetime
from model_accuracy_tracker import get_model_accuracy_display
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'K', 'D',
    'obv', 'obv_ma20', 'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr', 'price_change_5d', 'price_change_10d', 'price_change_20d', 'ma50_slope'
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
    df['rsi'] = 100 - (100 / (1 + g / (l + 1e-10)))
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
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()

def get_trading_signal():
    accuracy_display = get_model_accuracy_display(TICKER)
    print("🤖 " + TICKER + " (" + NAME + ") | 準確度: " + accuracy_display +
          " | " + datetime.now().strftime('%Y-%m-%d %H:%M'))
    print("=" * 80)
    try:
        model = joblib.load(MODEL_FILE)
        print("✅ 模型加載成功: " + MODEL_FILE)
    except Exception as e:
        print("❌ 模型加載失敗: " + str(e)); return None
    print("\\n📊 下載 " + TICKER + " 最新數據...")
    try:
        df = yf.download(TICKER, period='1y', progress=False)
        if df.empty: print("❌ 無法獲取數據"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print("✅ 成功下載 " + str(len(df)) + " 天數據")
    except Exception as e:
        print("❌ 數據下載失敗: " + str(e)); return None
    df = add_technical_indicators(df)
    latest = df.iloc[-1]
    current_price = float(latest['close'])
    X = latest[FEATURE_COLUMNS].values.reshape(1, -1)
    print("\\n🧠 AI 模型分析中...")
    proba = model.predict_proba(X)[0]
    pred  = model.predict(X)[0]
    buy_prob = proba[1] * 100
    print("預測結果: " + ("買入機率" if pred == 1 else "不買入") +
          " (買入: " + f"{buy_prob:.1f}%" + ", 不買入: " + f"{proba[0]*100:.1f}%)")
    print("\\n" + "=" * 80 + "\\n📊 技術指標\\n" + "=" * 80)
    rsi = float(latest['rsi']); macd = float(latest['macd']); ms = float(latest['macd_signal'])
    s10 = float(latest['sma_10']); s30 = float(latest['sma_30'])
    print(f"當前價格: NT${current_price:.2f}")
    print(f"RSI: {rsi:.2f}  " + ("[超買]" if rsi > 70 else "[超賣]" if rsi < 30 else "[中性]"))
    print(f"MACD: {macd:.4f}  " + ("[金叉]" if macd > ms else "[死叉]"))
    print(f"均線: SMA10={s10:.2f}, SMA30={s30:.2f}  " + ("[多頭]" if s10 > s30 else "[空頭]"))
    print(f"布林帶位置: {float(latest['bb_position']):.1f}%")
    print(f"KD: K={float(latest['K']):.1f}, D={float(latest['D']):.1f}")
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        print("\\n型態評分調整: " + f"{get_pattern_score_adjustment(patterns):+.1f}" + " 分")
    except Exception: pass
    print("\\n" + "=" * 80 + "\\n🎯 交易信號\\n" + "=" * 80)
    if pred == 1 and buy_prob >= 60:
        print("🟢 買入信號 (BUY)")
        print(f"   信心度: {buy_prob:.1f}%")
        print(f"   建議買入價格: NT${current_price*0.995:.2f} - NT${current_price*1.005:.2f}")
    elif pred == 1 and buy_prob >= 52:
        print("🟡 弱買入信號 (WEAK BUY)")
        print(f"   信心度: {buy_prob:.1f}%")
    else:
        print("🔴 不建議買入 (HOLD/WAIT)")
        print(f"   買入信心度不足: {buy_prob:.1f}%")
    print("=" * 80)
    return {'ticker': TICKER, 'price': current_price, 'prediction': pred,
            'buy_probability': buy_prob, 'rsi': rsi, 'macd': macd}

if __name__ == "__main__":
    get_trading_signal()
"""

for code, ticker, name, suffix in STOCKS:
    fname = f'get_trading_signal_{code}.py'
    if os.path.exists(fname):
        print(f'  SKIP  {fname}')
        continue
    model_file = f'xgb_{code}_{suffix}_model.pkl'
    header = f'"""\n{ticker} ({name}) XGBoost 交易信號生成器\n"""\n'
    consts = f"TICKER = '{ticker}'\nMODEL_FILE = '{model_file}'\nNAME = '{name}'\n"
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(header + consts + BODY)
    print(f'  CREATED {fname}')

print('Done.')
