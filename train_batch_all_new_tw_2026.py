"""
批量訓練台股新增股票 + 自動生成信號腳本 (2026)
- 59 stocks to train, 43 signal scripts to generate
- Auto-detects .TW vs .TWO exchange suffix
- Skips stocks that already have a model file
"""
import os, sys, io, json, time
import numpy as np, pandas as pd, yfinance as yf
import xgboost as xgb, joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TWO_STOCKS = {
    '3498','3615','4533','4577','4768','4908','4991','5011',
    '6134','6187','6220','6530','6877','7805','8086','8908','8917','8927',
    '6274','1785','4749','3131','6510','6683','3363','3081',
    '8069','6223','5483','6163','7709','7717',
    '3260','3491','5371','3105','4971',
    '8064','3163','3455',
    '3680','4772','6788','7703','8147','8071',
    '8027','5351','7734','7751','6138',
    '1569','1595','4951',
    '6234','6488','6207',
    '5386','6166','6263','6344','6426','6574','6691','7769',
}

# (code, chinese_name)
STOCKS = [
    # --- 16: Need Train Only (signal script already exists) ---
    ('1605','華新'),('2059','川湖'),('2303','聯電'),('2327','國巨'),
    ('2395','研華'),('2409','友達'),('2412','中華電信'),('2884','玉山金'),
    ('3481','群創'),('4938','和碩'),('6163','華電網'),('6443','元晶'),
    ('6446','藥華藥'),('7709','榮田'),('7769','震陞'),('8112','至上'),
    # --- 43: Need Both (train + generate signal script) ---
    ('1582','信錦'),('2356','英業達'),('2417','圓剛'),('2492','貝斯特'),
    ('2645','漢翔'),('2810','大成鋼'),('2880','華南金'),('2886','兆豐金'),
    ('2892','第一金'),('2912','統一超商'),('3090','日成'),('3167','渼洋'),
    ('3209','全科'),('3234','光環'),('3402','聯德'),('4142','國光生技'),
    ('4542','達方'),('4900','富爾特'),('4919','新唐科技'),('4958','臻鼎-KY'),
    ('4966','譜瑞-KY'),('4979','華星光通'),('5386','雷凌科技'),('5475','德英電子'),
    ('6166','增你強'),('6217','中探針'),('6263','普萊德'),('6344','萬年清'),
    ('6426','統新'),('6457','醣聯'),('6574','霖揚'),('6584','申豐'),
    ('6592','和潤企業'),('6658','迪智'),('6691','長天科技'),('6727','瑞傳'),
    ('6831','騰雲'),('6834','新纖維'),('6903','信強'),('6944','譜力'),
    ('7744','致伸'),('8043','金穎生技'),('9933','中鼎'),
]

FEATURE_COLUMNS = [
    'rsi','macd','macd_signal','macd_hist','bb_position','K','D',
    'obv','obv_ma20','sma_10','sma_30','sma_50','sma_200',
    'volatility','atr','price_change_5d','price_change_10d','price_change_20d','ma50_slope'
]


def get_ticker(code):
    if code in TWO_STOCKS:
        return f"{code}.TWO"
    try:
        df = yf.download(f"{code}.TW", period="5y", progress=False)
        if not df.empty and len(df) >= 200:
            return f"{code}.TW"
    except Exception:
        pass
    try:
        df = yf.download(f"{code}.TWO", period="5y", progress=False)
        if not df.empty and len(df) >= 100:
            return f"{code}.TWO"
    except Exception:
        pass
    return f"{code}.TW"


def add_features(df):
    df['sma_10']   = df['close'].rolling(10).mean()
    df['sma_30']   = df['close'].rolling(30).mean()
    df['sma_50']   = df['close'].rolling(50).mean()
    df['sma_200']  = df['close'].rolling(200).mean()
    df['ema_12']   = df['close'].ewm(span=12).mean()
    df['ema_26']   = df['close'].ewm(span=26).mean()
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


def train_stock(code, name):
    ticker = get_ticker(code)
    suffix = 'two' if ticker.endswith('.TWO') else 'tw'
    model_file = f"xgb_{code}_{suffix}_model.pkl"
    model_path = os.path.join(SCRIPT_DIR, model_file)

    if os.path.exists(model_path):
        print(f"  ⏭️  模型已存在 ({model_file})，跳過")
        return ticker, suffix, model_file, None

    print(f"  📡 下載 {ticker}...", end=' ', flush=True)
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-01-01', progress=False)
        if df.empty:
            print(f"❌ 無數據"); return ticker, suffix, model_file, False
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"{len(df)} 天", end=' ', flush=True)
    except Exception as e:
        print(f"❌ {e}"); return ticker, suffix, model_file, False

    df = add_features(df)
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)
    df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])

    if len(df_clean) < 100:
        print(f"❌ 數據不足 ({len(df_clean)})"); return ticker, suffix, model_file, False

    X = df_clean[FEATURE_COLUMNS]
    y = df_clean['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.05, n_estimators=200,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        objective='binary:logistic', random_state=42, eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, model_path)
    json.dump({
        'symbol': ticker, 'model_type': 'XGBoost',
        'training_accuracy': round(accuracy_score(y_train, model.predict(X_train)) * 100, 2),
        'validation_accuracy': round(test_acc * 100, 2),
        'backtest_accuracy': round(test_acc * 100, 2),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }, open(os.path.join(SCRIPT_DIR, f"model_accuracy_{code}_{suffix.upper()}.json"), 'w', encoding='utf-8'),
        ensure_ascii=False, indent=2)

    star = '🌟' if test_acc >= 0.70 else '✅' if test_acc >= 0.50 else '⚠️ '
    print(f"→ {star} {test_acc*100:.2f}%  💾 {model_file}")
    return ticker, suffix, model_file, test_acc


def make_signal_script(code, name, ticker, model_file):
    path = os.path.join(SCRIPT_DIR, f"get_trading_signal_{code}.py")
    if os.path.exists(path):
        return
    script = f'''"""
{ticker} ({name}) XGBoost 交易信號生成器
"""
TICKER = '{ticker}'
MODEL_FILE = '{model_file}'
NAME = '{name}'
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, yfinance as yf, joblib
from datetime import datetime
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
    print(f"🤖 {{TICKER}} ({{NAME}}) AI 交易信號生成器")
    print(f"模型準確度: {{_acc}}")
    print(f"生成時間: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print("=" * 80)
    try:
        model = joblib.load(MODEL_FILE)
        print(f"✅ 模型加載成功: {{MODEL_FILE}}")
    except Exception as e:
        print(f"❌ 模型加載失敗: {{e}}"); return None
    print(f"\\n📊 下載 {{TICKER}} 最新數據...")
    try:
        df = yf.download(TICKER, period='300d', progress=False, auto_adjust=True)
        if df.empty: print("❌ 無法獲取數據"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={{'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}}).reset_index()
        print(f"✅ 成功下載 {{len(df)}} 天數據")
    except Exception as e:
        print(f"❌ 數據下載失敗: {{e}}"); return None
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
    print("\\n" + "=" * 80)
    print("📊 技術指標分析")
    print("=" * 80)
    print(f"當前價格: NT${{current_price:.2f}}")
    print(f"RSI: {{rsi:.2f}}  {{\'[超買]\' if rsi > 70 else \'[超賣]\' if rsi < 30 else \'[中性]\'}}")
    print(f"MACD: {{macd:.4f}}  {{\'[金叉]\' if macd > ms else \'[死叉]\'}}")
    print(f"均線: SMA10={{s10:.2f}}, SMA30={{s30:.2f}}  {{\'[多頭]\' if s10 > s30 else \'[空頭]\'}}")
    print(f"布林帶位置: {{float(latest[\'bb_position\']):.1f}}%")
    print(f"KD: K={{float(latest[\'K\']):.1f}}, D={{float(latest[\'D\']):.1f}}")
    print("\\n" + "=" * 80)
    print("🎯 AI 交易信號")
    print("=" * 80)
    print(f"買入概率: {{buy_prob:.1f}}%")
    if pred == 1 and buy_prob >= 60:
        print("🟢 買入信號 (BUY)")
        print(f"   信心度: {{buy_prob:.1f}}%")
        print(f"   建議買入價格: NT${{current_price*0.995:.2f}} - NT${{current_price*1.005:.2f}}")
        print(f"   止損位: NT${{current_price*0.95:.2f}} (-5%)")
    elif pred == 1 and buy_prob >= 52:
        print("🟡 弱買入 (HOLD - WEAK BUY)")
        print(f"   信心度: {{buy_prob:.1f}}%，謹慎操作")
    elif pred == 0 and buy_prob <= 35:
        print("🔴 賣出觀望 (SELL)")
        print(f"   買入信心度低: {{buy_prob:.1f}}%")
    else:
        print("🟡 持有觀望 (HOLD)")
        print(f"   買入信心度不足: {{buy_prob:.1f}}%")
    print("=" * 80)
    print(f"\\n📱 快速摘要:")
    print(f"   股票: {{TICKER}} ({{NAME}})")
    print(f"   日期: {{latest_date}}")
    print(f"   價格: NT${{current_price:.2f}}")
    if pred == 1 and buy_prob >= 60:
        print("   信號: 🟢 買入信號 (BUY)")
    elif pred == 1 and buy_prob >= 52:
        print("   信號: 🟡 持有 (HOLD)")
    elif pred == 0 and buy_prob <= 35:
        print("   信號: 🔴 賣出 (SELL)")
    else:
        print("   信號: 🟡 持有 (HOLD)")
    print(f"   {{_acc}}")
    return {{'ticker': TICKER, 'price': current_price, 'prediction': int(pred), 'buy_probability': buy_prob}}

if __name__ == "__main__":
    get_trading_signal()
'''
    with open(path, 'w', encoding='utf-8') as f:
        f.write(script)
    print(f"  📝 信號腳本: get_trading_signal_{code}.py")


if __name__ == "__main__":
    print("=" * 80)
    print(f"批量訓練台股 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"共 {len(STOCKS)} 個股票 (32個已完成將跳過)")
    print("=" * 80)

    results = []
    for i, (code, name) in enumerate(STOCKS, 1):
        print(f"\n[{i}/{len(STOCKS)}] {code} {name}")
        try:
            ticker, suffix, model_file, acc = train_stock(code, name)
            make_signal_script(code, name, ticker, model_file)
            results.append({'code': code, 'name': name, 'acc': acc})
        except Exception as e:
            print(f"  ❌ 失敗: {e}")
            results.append({'code': code, 'name': name, 'acc': False})
        time.sleep(0.5)

    print("\n\n" + "=" * 80)
    print("完成!")
    trained = [r for r in results if isinstance(r['acc'], float)]
    skipped = [r for r in results if r['acc'] is None]
    failed  = [r for r in results if r['acc'] is False]
    print(f"  新訓練: {len(trained)}  跳過: {len(skipped)}  失敗: {len(failed)}")
    if trained:
        print("\n準確度排行 (新訓練):")
        for r in sorted(trained, key=lambda x: x['acc'], reverse=True):
            s = '🌟' if r['acc'] >= 0.70 else '✅' if r['acc'] >= 0.50 else '⚠️ '
            print(f"  {s} {r['code']} {r['name']:12s} {r['acc']*100:.2f}%")
    if failed:
        print(f"\n失敗: {[r['code'] for r in failed]}")
