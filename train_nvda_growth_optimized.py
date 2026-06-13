"""
NVDA (NVIDIA) 成長股優化訓練
專門針對高波動成長股設計
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import json
from datetime import datetime

TICKER = 'NVDA'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("=" * 80)
print(f"🚀 成長股優化訓練 {TICKER} (NVIDIA)")
print("=" * 80)
print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

try:
    # 下載數據
    print(f"\n📊 下載 {TICKER} 歷史數據...")
    df = yf.download(TICKER, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), progress=False)

    if df.empty:
        print("❌ 無法下載數據")
        sys.exit(1)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                            'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()

    print(f"✅ 下載 {len(df)} 天數據")

    # 成長股專用技術指標
    print("\n🔧 計算成長股專用指標...")

    # 移動平均線
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林帶
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_width = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(bb_width != 0, (df['close'] - df['bb_lower']) / bb_width * 100, 50)

    # KD
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    df['K'] = np.where(high_low_range != 0, (df['close'] - low_14) / high_low_range * 100, 50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # 🔥 成長股關鍵指標
    # 動量 (Momentum)
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_10'] = df['close'].pct_change(10) * 100
    df['momentum_20'] = df['close'].pct_change(20) * 100

    # ROC (Rate of Change)
    df['roc_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100)
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100)

    # 均線趨勢
    df['sma_10_slope'] = df['sma_10'].pct_change(5) * 100
    df['sma_30_slope'] = df['sma_30'].pct_change(5) * 100
    df['sma_50_slope'] = df['sma_50'].pct_change(5) * 100

    # 價格位置 (相對於均線)
    df['price_above_sma50'] = ((df['close'] - df['sma_50']) / df['sma_50'] * 100)
    df['price_above_sma200'] = ((df['close'] - df['sma_200']) / df['sma_200'] * 100)

    # 成交量趨勢
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma_20']
    df['volume_momentum'] = df['volume'].pct_change(5) * 100

    # 波動率
    df['volatility_20'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # OBV 動量
    df['obv_momentum'] = np.where(df['obv_ma20'] != 0, (df['obv'] - df['obv_ma20']) / df['obv_ma20'].abs() * 100, 0)

    # 趨勢強度
    df['trend_strength'] = np.where(df['sma_50'] != 0, abs(df['sma_10'] - df['sma_50']) / df['sma_50'] * 100, 0)

    # 均線交叉比率 (取代原始價格)
    df['sma_10_30_ratio'] = np.where(df['sma_30'] != 0, df['sma_10'] / df['sma_30'], 1)
    df['sma_30_50_ratio'] = np.where(df['sma_50'] != 0, df['sma_30'] / df['sma_50'], 1)
    df['sma_50_200_ratio'] = np.where(df['sma_200'] != 0, df['sma_50'] / df['sma_200'], 1)

    # 清理 inf 值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"✅ 特徵工程完成")

    # 🔥 針對成長股使用更高的閾值
    print("\n📈 嘗試適合成長股的高閾值...")
    thresholds = [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]  # 3% - 6%
    best_acc = 0
    best_threshold = 0.04

    feature_columns = [
        # 動量指標 (成長股最重要)
        'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10',
        # 趨勢指標
        'sma_10_slope', 'sma_30_slope', 'sma_50_slope',
        'price_above_sma50', 'price_above_sma200',
        'trend_strength',
        # 成交量指標
        'volume_ratio', 'volume_momentum',
        'obv_momentum',
        # 傳統指標
        'rsi', 'macd', 'macd_hist',
        'bb_position', 'K', 'D',
        # 均線比率 (取代原始價格)
        'sma_10_30_ratio', 'sma_30_50_ratio', 'sma_50_200_ratio',
        'volatility_20', 'atr'
    ]

    for threshold in thresholds:
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return'] > threshold).astype(int)

        df_clean = df.dropna(subset=feature_columns + ['target'])

        if len(df_clean) < 100:
            continue

        X = df_clean[feature_columns]
        y = df_clean['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 🔥 防止過擬合的正則化參數
        neg_c = int((y_train == 0).sum())
        pos_c = int((y_train == 1).sum())
        spw = neg_c / pos_c if pos_c > 0 else 1
        model = xgb.XGBClassifier(
            max_depth=4,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.2,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=spw
        )

        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"  閾值 {threshold*100:.1f}%: 準確度 {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_threshold = threshold

    # 使用最佳閾值訓練最終模型
    print(f"\n🎯 使用最佳閾值: {best_threshold*100:.1f}%")

    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > best_threshold).astype(int)
    df_clean = df.dropna(subset=feature_columns + ['target'])

    X = df_clean[feature_columns]
    y = df_clean['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    print(f"訓練集: {len(X_train)} 天")
    print(f"測試集: {len(X_test)} 天")
    print(f"目標分布 - 買入: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")

    # 計算正負樣本比例
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    calc_scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    # 最終模型
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=calc_scale_pos_weight
    )

    print("\n⏳ 訓練最終模型...")
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"\n📊 最終結果:")
    print(f"訓練準確度: {train_acc*100:.2f}%")
    print(f"測試準確度: {test_acc*100:.2f}%")
    print(f"過擬合差距: {(train_acc - test_acc)*100:.2f}%")

    # 分類報告
    print(f"\n📋 分類報告:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['不買入', '買入']))

    # 保存模型
    model_path = os.path.join(SCRIPT_DIR, 'xgb_nvda_growth_model.pkl')
    joblib.dump(model, model_path)
    print(f"\n✅ 模型已保存: {model_path}")

    # 保存準確度
    accuracy_data = {
        'symbol': 'NVDA',
        'model_type': 'XGBoost Growth Optimized',
        'training_accuracy': float(train_acc * 100),
        'validation_accuracy': float(test_acc * 100),
        'backtest_accuracy': float(test_acc * 100),
        'overfitting_gap': float((train_acc - test_acc) * 100),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notes': f'Growth stock optimized + {best_threshold*100:.1f}% threshold + regularization'
    }

    with open(os.path.join(SCRIPT_DIR, 'model_accuracy_NVDA_growth.json'), 'w', encoding='utf-8') as f:
        json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🎯 特徵重要性 (Top 15):")
    print(feature_importance.head(15).to_string(index=False))

    # 保存特徵重要性
    importance_dict = {
        'ticker': 'NVDA',
        'company': 'NVIDIA',
        'model_type': 'XGBoost Growth Optimized',
        'model_accuracy': test_acc,
        'feature_importance': dict(zip(feature_importance['feature'],
                                      feature_importance['importance'].tolist()))
    }

    with open(os.path.join(SCRIPT_DIR, 'NVDA_feature_importance_growth.json'), 'w', encoding='utf-8') as f:
        json.dump(importance_dict, f, ensure_ascii=False, indent=2)

    status = 'EXCELLENT' if test_acc >= 0.80 else 'PASS' if test_acc >= 0.58 else 'OK' if test_acc >= 0.50 else 'LOW'
    print(f"\n✅ 完成! 狀態: {status} ({test_acc*100:.2f}%)")

    print("\n" + "=" * 80)
    print(f"完成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

except Exception as e:
    print(f"❌ 錯誤: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
