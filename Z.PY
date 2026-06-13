# Uninstall conflicting versions and install compatible ones


import yfinance as yf
import pandas as pd
import numpy as np
import json
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Create ta module with technical analysis functions
class ta:
    @staticmethod
    def sma(data, length=20):
        """Simple Moving Average"""
        return data.rolling(window=length).mean()
    
    @staticmethod
    def rsi(data, length=14):
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """MACD Indicator"""
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'MACD_12_26_9': macd_line,
            'MACDs_12_26_9': signal_line,
            'MACDh_12_26_9': histogram
        })
    
    @staticmethod
    def stoch(high, low, close, k_length=14, d_length=3):
        """Stochastic Oscillator"""
        low_min = low.rolling(window=k_length).min()
        high_max = high.rolling(window=k_length).max()
        k_percent = 100 * ((close - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_length).mean()
        return pd.DataFrame({
            'STOCHk_14_3_3': k_percent,
            'STOCHd_14_3_3': d_percent
        })
    
    @staticmethod
    def bbands(data, length=20, std_dev=2):
        """Bollinger Bands"""
        sma = data.rolling(window=length).mean()
        std = data.rolling(window=length).std()
        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)
        return pd.DataFrame({
            'BBU_20_2': bb_upper,
            'BBM_20_2': sma,
            'BBL_20_2': bb_lower
        })
    
    @staticmethod
    def atr(high, low, close, length=14):
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=length).mean()
        return atr
    
    @staticmethod
    def obv(close, volume):
        """On Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

# Fix console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# 定義要分析的股票清單 (AI伺服器與半導體供應鏈相關股票) train for 1504.tw 3532.tw. 2312.tw. 1519.tw 2419.tw 6488.tw
tickers = [
    '2408.TW',  # 南亞科 - DRAM製造商
    '8021.TW',  # 尖點 - 電子工具/PCB鑽針
    '2344.TW',  # 華邦電 - 利基型記憶體
    '8110.TW',  # 華東 - 封測廠
    '4722.TW',  # 國精化 - 半導體化學材料
    '8210.TW',  # 勤誠 - 伺服器機殼
    '8499.TW',  # 鼎炫-KY - 電子測試設備
    '2368.TW',  # 金像電 - PCB製造商
    '4989.TW',  # 榮科 - 電子零組件
    '2383.TW',  # 台光電 - 銅箔基板
    '1504.TW',  # 台光電 - 銅箔基板
    '3532.TW',  # 台光電 - 銅箔基板
    '2312.TW',  # 台光電 - 銅箔基板
    '1519.TW',  # 台光電 - 銅箔基板
    '2419.TW',  # 台光電 - 銅箔基板
    '6488.TW',  # 台光電 - 銅箔基板
    '2308.TW',  # 台光電 - 銅箔基板
    '6443.TW',  # 台光電 - 銅箔基板
    'HTGC',
    'NAT',
    'NXPI',
    'AMKR',

]

def get_stock_feature_importance(ticker):
    print(f"⚡ 正在運算 ML 模型: {ticker} ...")

    # 1. 抓取數據 (抓取過去 2 年以確保有足夠樣本訓練)
    df = yf.download(ticker, period="2y", progress=False)

    if df.empty:
        return None

    # 處理 MultiIndex Column 問題 (yfinance 新版修正)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. 特徵工程 (Feature Engineering) - 建立您指定的特徵
    # 移動平均線
    df['MA_20'] = ta.sma(df['Close'], length=20)
    df['MA_50'] = ta.sma(df['Close'], length=50)
    df['MA_200'] = ta.sma(df['Close'], length=200)

    # 計算 MA50 斜率 (Slope): 取 5 天前的變化率作為斜率代理
    df['MA50_slope'] = (df['MA_50'] - df['MA_50'].shift(5)) / 5

    # 動能指標 (Momentum)
    df['rsi'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'])
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_hist'] = macd['MACDh_12_26_9']

    # KD 指標 (Stochastic)
    stoch = ta.stoch(df['High'], df['Low'], df['Close'])
    df['K'] = stoch['STOCHk_14_3_3']
    df['D'] = stoch['STOCHd_14_3_3']

    # 布林通道位置 (BB Position / %B)
    bb = ta.bbands(df['Close'], length=20)
    # %B = (Price - Lower) / (Upper - Lower)
    # Check column names (pandas_ta version differences)
    bb_lower_col = [col for col in bb.columns if 'BBL' in col][0]
    bb_upper_col = [col for col in bb.columns if 'BBU' in col][0]
    df['bb_position'] = (df['Close'] - bb[bb_lower_col]) / (bb[bb_upper_col] - bb[bb_lower_col])

    # 波動率 (Volatility) & ATR
    df['volatility'] = df['Close'].pct_change().rolling(20).std()
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

    # 成交量指標 (Volume)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_MA'] = ta.sma(df['OBV'], length=20) # OBV 的移動平均

    # 價格變化 (Price Change)
    df['price_change_5d'] = df['Close'].pct_change(5)
    df['price_change_20d'] = df['Close'].pct_change(20)
    df['price_change_1d'] = df['Close'].pct_change(1)

    # 新增進階特徵 (Advanced Features for Better Accuracy)
    # 1. 趨勢強度指標
    df['trend_strength'] = abs(df['MA_20'] - df['MA_50']) / df['Close']

    # 2. 動能比率
    df['momentum_ratio'] = df['price_change_5d'] / (df['volatility'] + 0.0001)

    # 3. 成交量變化率
    df['volume_change'] = df['Volume'].pct_change(5)
    df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # 4. RSI 變化率 (RSI momentum)
    df['rsi_change'] = df['rsi'].diff(5)

    # 5. MACD 交叉信號強度
    df['macd_strength'] = abs(df['macd'] - df['macd_signal'])

    # 6. 價格相對位置 (Price Position)
    df['price_position'] = (df['Close'] - df['Close'].rolling(20).min()) / \
                           (df['Close'].rolling(20).max() - df['Close'].rolling(20).min() + 0.0001)

    # 7. 波動率變化
    df['volatility_change'] = df['volatility'].pct_change(5)

    # 8. MA Golden/Death Cross 信號
    df['ma_cross_signal'] = np.where(df['MA_20'] > df['MA_50'], 1, -1)

    # 9. 布林通道寬度
    df['bb_width'] = (bb[bb_upper_col] - bb[bb_lower_col]) / df['Close']

    # 10. KD交叉強度
    df['kd_diff'] = df['K'] - df['D']

    # 3. 建立預測目標 (Target)
    # 改進：預測未來3天的平均漲幅，而非只看明天
    future_returns = (df['Close'].shift(-3) - df['Close']) / df['Close']
    df['Target'] = np.where(future_returns > 0.01, 1, 0)  # 漲幅超過1%才算買入信號

    # 清除空值 (因計算 MA200 會產生前 200 天 NaN)
    df.dropna(inplace=True)

    # 4. 準備 ML 訓練資料 (包含原始特徵 + 新增進階特徵)
    features = [
        # 原始特徵
        "OBV_MA", "volatility", "MA_50", "MA_200", "OBV", "MA_20",
        "macd_hist", "ATR", "MA50_slope", "macd", "macd_signal",
        "price_change_5d", "bb_position", "rsi", "K", "price_change_20d", "D",
        "price_change_1d",
        # 新增進階特徵
        "trend_strength", "momentum_ratio", "volume_change", "volume_ma_ratio",
        "rsi_change", "macd_strength", "price_position", "volatility_change",
        "ma_cross_signal", "bb_width", "kd_diff"
    ]

    X = df[features]
    y = df['Target']

    # 分割訓練集與測試集 (不打亂順序，因為是時間序列)
    # 使用更大的訓練集 (80% -> 85%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)

    # 特徵標準化 (Feature Scaling) - 提升模型性能
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 訓練改進的隨機森林模型 (更多樹、更深的樹)
    model = RandomForestClassifier(
        n_estimators=200,           # 增加樹的數量
        max_depth=15,               # 限制深度避免過擬合
        min_samples_split=5,        # 減少最小分割樣本
        min_samples_leaf=2,         # 葉節點最小樣本
        max_features='sqrt',        # 使用sqrt特徵數
        class_weight='balanced',    # 平衡類別權重
        random_state=42,
        n_jobs=-1                   # 使用所有CPU核心
    )
    model.fit(X_train_scaled, y_train)

    # 6. 計算準確率與重要性
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    # 計算訓練集準確率以檢測過擬合
    train_predictions = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, train_predictions)

    # 提取特徵重要性並排序
    importances = model.feature_importances_
    feature_importance_dict = dict(zip(features, importances))
    sorted_importance = dict(sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True))

    # 7. 格式化輸出
    result = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "test_accuracy": round(accuracy, 4),      # 測試集準確度
        "train_accuracy": round(train_accuracy, 4), # 訓練集準確度
        "overfitting_gap": round(train_accuracy - accuracy, 4), # 過擬合指標
        "sample_size": {
            "train": len(X_train),
            "test": len(X_test)
        },
        "feature_importance": sorted_importance
    }

    return result

# 執行並印出 JSON 結果
results_list = []
for t in tickers:
    try:
        res = get_stock_feature_importance(t)
        if res:
            results_list.append(res)
            # 若要單獨印出每個股票的 JSON
            print(json.dumps(res, indent=2, ensure_ascii=False))
            print("-" * 30)
    except Exception as e:
        print(f"Error processing {t}: {e}")