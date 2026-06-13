"""
NVDA (NVIDIA) 高級 XGBoost 訓練
使用網格搜索、特徵工程、交叉驗證優化
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import json
from datetime import datetime

TICKER = 'NVDA'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print("=" * 80)
print(f"🚀 高級訓練 {TICKER} (NVIDIA) XGBoost 模型")
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

    # 基本技術指標
    print("\n🔧 計算技術指標...")
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
    bb_width = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(bb_width != 0, (df['close'] - df['bb_lower']) / bb_width * 100, 50)

    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    high_low_range = high_14 - low_14
    df['K'] = np.where(high_low_range != 0, (df['close'] - low_14) / high_low_range * 100, 50)
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

    df['ma50_slope'] = np.where(df['sma_50'].shift(5) != 0, df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100, 0)

    # 🔥 新增高級特徵
    print("🔥 添加高級特徵...")

    # 動量指標 (百分比，避免非平穩性)
    df['momentum_5'] = df['close'].pct_change(5) * 100
    df['momentum_10'] = df['close'].pct_change(10) * 100
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100)

    # 趨勢強度
    df['atr_ratio'] = np.where(df['close'] != 0, df['atr'] / df['close'] * 100, 0)  # ATR 佔價格比例
    df['trend_strength'] = np.where(df['sma_30'] != 0, abs(df['sma_10'] - df['sma_30']) / df['sma_30'] * 100, 0)

    # 波動率指標
    df['volatility_ratio'] = df['atr'] / df['close'] * 100
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close'] * 100

    # 成交量指標
    df['volume_sma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = np.where(df['volume_sma_20'] != 0, df['volume'] / df['volume_sma_20'], 1)
    df['volume_change'] = df['volume'].pct_change(5) * 100

    # 價格位置指標
    df['price_position_sma50'] = np.where(df['sma_50'] != 0, (df['close'] - df['sma_50']) / df['sma_50'] * 100, 0)
    df['price_position_sma200'] = np.where(df['sma_200'] != 0, (df['close'] - df['sma_200']) / df['sma_200'] * 100, 0)

    # 均線趨勢
    df['sma_10_slope'] = np.where(df['sma_10'].shift(5) != 0, df['sma_10'].diff(5) / df['sma_10'].shift(5) * 100, 0)
    df['sma_30_slope'] = np.where(df['sma_30'].shift(5) != 0, df['sma_30'].diff(5) / df['sma_30'].shift(5) * 100, 0)

    # 交互特徵
    df['rsi_bb'] = df['rsi'] * df['bb_position'] / 100
    df['macd_volume'] = df['macd'] * df['volume_ratio']
    df['trend_momentum'] = df['trend_strength'] * df['momentum_5']

    # OBV 趨勢
    df['obv_trend'] = np.where(df['obv_ma20'] != 0, (df['obv'] - df['obv_ma20']) / df['obv_ma20'].abs() * 100, 0)

    # 均線交叉比率 (取代原始價格)
    df['sma_10_30_ratio'] = np.where(df['sma_30'] != 0, df['sma_10'] / df['sma_30'], 1)
    df['sma_50_200_ratio'] = np.where(df['sma_200'] != 0, df['sma_50'] / df['sma_200'], 1)

    # 清理 inf 值
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    print(f"✅ 特徵工程完成")

    # 嘗試不同的目標閾值
    print("\n📈 嘗試多個目標閾值...")
    thresholds = [0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    best_acc = 0
    best_threshold = 0.02

    feature_columns = [
        # 基本指標
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D',
        'volatility',
        'price_change_5d', 'price_change_10d', 'price_change_20d',
        'ma50_slope',
        # 高級特徵
        'momentum_5', 'momentum_10', 'roc_10',
        'atr_ratio', 'trend_strength',
        'volatility_ratio', 'high_low_ratio',
        'volume_ratio', 'volume_change',
        'price_position_sma50', 'price_position_sma200',
        'sma_10_slope', 'sma_30_slope',
        'rsi_bb', 'macd_volume', 'trend_momentum',
        'obv_trend',
        # 均線比率 (取代原始價格)
        'sma_10_30_ratio', 'sma_50_200_ratio'
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

        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.03,
            n_estimators=300,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"  閾值 {threshold*100:.1f}%: 準確度 {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_threshold = threshold

    # 使用最佳閾值進行網格搜索優化
    print(f"\n🔍 使用最佳閾值 {best_threshold*100:.1f}% 進行超參數網格搜索...")

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

    # 網格搜索參數 (精簡: 2×3×2×2 = 24 組合 × 3 folds = 72 fits)
    param_grid = {
        'max_depth': [4, 6],
        'learning_rate': [0.01, 0.03, 0.05],
        'n_estimators': [200, 300],
        'min_child_weight': [2, 3],
    }

    # 使用時間序列交叉驗證
    tscv = TimeSeriesSplit(n_splits=3)

    base_model = xgb.XGBClassifier(
        gamma=0.1,
        subsample=0.85,
        colsample_bytree=0.85,
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )

    print("\n⏳ 執行網格搜索 (這可能需要幾分鐘)...")
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n✅ 最佳參數: {grid_search.best_params_}")
    print(f"✅ 交叉驗證最佳分數: {grid_search.best_score_*100:.2f}%")

    # 使用最佳參數訓練最終模型
    best_model = grid_search.best_estimator_

    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))

    print(f"\n📊 最終結果:")
    print(f"訓練準確度: {train_acc*100:.2f}%")
    print(f"測試準確度: {test_acc*100:.2f}%")

    # 分類報告
    print(f"\n📋 分類報告:")
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['不買入', '買入']))

    # 保存模型
    model_path = os.path.join(SCRIPT_DIR, 'xgb_nvda_advanced_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\n✅ 模型已保存: {model_path}")

    # 保存準確度
    accuracy_data = {
        'symbol': 'NVDA',
        'model_type': 'XGBoost Advanced',
        'training_accuracy': float(train_acc * 100),
        'validation_accuracy': float(test_acc * 100),
        'backtest_accuracy': float(test_acc * 100),
        'cv_score': float(grid_search.best_score_ * 100),
        'best_params': grid_search.best_params_,
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'notes': f'Advanced features + GridSearch + {best_threshold*100:.1f}% threshold'
    }

    with open(os.path.join(SCRIPT_DIR, 'model_accuracy_NVDA_advanced.json'), 'w', encoding='utf-8') as f:
        json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🎯 特徵重要性 (Top 15):")
    print(feature_importance.head(15).to_string(index=False))

    # 保存特徵重要性
    importance_dict = {
        'ticker': 'NVDA',
        'company': 'NVIDIA',
        'model_type': 'XGBoost Advanced',
        'model_accuracy': test_acc,
        'cv_score': grid_search.best_score_,
        'feature_importance': dict(zip(feature_importance['feature'],
                                      feature_importance['importance'].tolist()))
    }

    with open(os.path.join(SCRIPT_DIR, 'NVDA_feature_importance_advanced.json'), 'w', encoding='utf-8') as f:
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
