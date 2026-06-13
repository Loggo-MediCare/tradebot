import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import json
import sys
import warnings
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Fix console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Suppress warnings
warnings.filterwarnings('ignore')

# 定義目標股票
tickers = ['8499.TW']  # 鼎炫-KY - 電子測試設備 (V1.0最佳表現者: 74.42%)

def get_enhanced_stock_prediction(ticker):
    print(f"🔥 V2.0 深度運算中: {ticker} (包含滯後特徵與梯度提升)...")
    
    # 1. 擴大數據範圍：抓取 10 年數據以涵蓋完整的「半導體景氣循環」
    df = yf.download(ticker, period="10y", progress=False)
    
    if len(df) < 500: # 資料過少則跳過
        return None
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. 特徵工程 (Feature Engineering) - V2.0 強化版
    
    # --- A. 基礎技術指標 ---
    df['MA_20'] = ta.sma(df['Close'], length=20)
    df['MA_50'] = ta.sma(df['Close'], length=50)
    df['MA_200'] = ta.sma(df['Close'], length=200)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # MACD
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']
    
    # ATR & 波動率
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
    
    # --- B. 關鍵升級：滯後特徵 (Lag Features) ---
    # 讓模型知道「昨天」與「前天」的變化，而不僅僅是「今天」
    for lag in [1, 3, 5]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        df[f'Vol_Change_Lag_{lag}'] = df['Volume'].pct_change(lag)
    
    # --- C. 距離特徵 (Distance from MA) ---
    # 乖離率：價格距離年線多遠？(判斷是否超漲/超跌)
    df['Dist_MA200'] = (df['Close'] - df['MA_200']) / (df['MA_200'] + 0.0001)

    # 3. 定義預測目標 (Target)
    # 為了降低雜訊，我們改預測：未來 5 天後的股價是否上漲？ (趨勢預測比單日漲跌更準)
    prediction_days = 5
    df['Target'] = np.where(df['Close'].shift(-prediction_days) > df['Close'], 1, 0)

    # 清除空值和無限值
    df.dropna(inplace=True)

    # 替換無限值為NaN，然後刪除
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 4. 準備訓練資料
    features = [
        'RSI', 'MACD', 'MACD_hist', 'ATR', 'Volatility', 
        'Dist_MA200', 'Return_Lag_1', 'Return_Lag_3', 'Return_Lag_5',
        'Vol_Change_Lag_1', 'Vol_Change_Lag_5'
    ]
    
    X = df[features]
    y = df['Target']

    # 再次檢查並清理無限值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())  # 用中位數填充剩餘NaN

    # 檢查是否有足夠的數據
    if len(X) < 500:
        print(f"   ⚠️  數據不足 ({len(X)} 樣本)")
        return None

    # 5. 滾動式時間序列驗證 (Walk-Forward Validation)
    # 不使用簡單的 train_test_split，而是模擬真實時間軸：用過去 5 年預測下一年
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    feature_importances = np.zeros(len(features))

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 特徵標準化 (每次折疊都重新標準化)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 使用 Gradient Boosting (通常比 Random Forest 更精準)
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        preds = model.predict(X_test_scaled)
        scores.append(accuracy_score(y_test, preds))
        feature_importances += model.feature_importances_

    # 平均準確率與特徵重要性
    avg_accuracy = np.mean(scores)
    avg_importance = feature_importances / tscv.get_n_splits()
    
    # 排序特徵
    feature_map = dict(zip(features, avg_importance))
    sorted_importance = dict(sorted(feature_map.items(), key=lambda item: item[1], reverse=True))

    # 6. 輸出結果
    result = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d"),
        "model_type": "GradientBoosting + Walk-Forward",
        "prediction_horizon": "5 Days",
        "model_accuracy": avg_accuracy,
        "feature_importance": sorted_importance
    }
    
    return result

# 執行分析
for t in tickers:
    try:
        res = get_enhanced_stock_prediction(t)
        if res:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            print("-" * 30)
    except Exception as e:
        print(f"Error {t}: {e}")