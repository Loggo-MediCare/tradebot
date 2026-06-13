"""
批次訓練剩餘低準確度美股 XGBoost 模型
排除已訓練的優先股票: MSFT, GOOGL, AMD, INTC, META, ORCL
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
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
import json
from datetime import datetime

# 剩餘需要訓練的股票 (27支)
remaining_stocks = [
    'RDW', 'WDC', 'IONQ', 'HSAI', 'SMR', 'VRT',
    'DECK', 'MRVL', 'INVZ', 'MPWR', 'MRNA', 'GILD',
    'SNPS', 'QUBT', 'NVO', 'KLAC', 'ARM', 'NEM',
    'SATL', 'STX', 'CDNS', 'RGTI', 'DOCN', 'SMCI',
    'APLD', 'TXN', 'FN'
]

def add_technical_indicators(df):
    """添加技術指標"""
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

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)

    # KD
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    # 波動性
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # 價格變化率
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    # MA50 斜率
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    return df

def train_stock(ticker):
    """訓練單支股票"""
    print(f"\n{'='*80}")
    print(f"處理: {ticker}")
    print(f"{'='*80}")

    try:
        # 下載數據
        df = yf.download(ticker, start='2015-01-01', end='2025-12-31', progress=False)

        if df.empty:
            print(f"FAILED: {ticker} - 無法下載數據")
            return False, 0

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                                'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()

        print(f"下載 {len(df)} 天數據")

        # 添加技術指標
        df = add_technical_indicators(df)

        # 定義目標
        df['future_return'] = df['close'].shift(-5) / df['close'] - 1
        df['target'] = (df['future_return'] > 0.02).astype(int)

        # 特徵
        feature_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'K', 'D', 'obv', 'obv_ma20',
            'sma_10', 'sma_30', 'sma_50', 'sma_200',
            'volatility', 'atr',
            'price_change_5d', 'price_change_10d', 'price_change_20d',
            'ma50_slope'
        ]

        df_clean = df.dropna(subset=feature_columns + ['target'])
        X = df_clean[feature_columns]
        y = df_clean['target']

        if len(X) < 100:
            print(f"FAILED: {ticker} - 數據不足 (僅 {len(X)} 筆)")
            return False, 0

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # 訓練模型
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train)

        # 評估
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        print(f"訓練準確度: {train_acc*100:.2f}%")
        print(f"測試準確度: {test_acc*100:.2f}%")

        # 保存模型
        model_filename = f'xgb_{ticker.lower()}_model.pkl'
        joblib.dump(model, model_filename)

        # 保存準確度
        accuracy_data = {
            'symbol': ticker,
            'model_type': 'XGBoost',
            'training_accuracy': float(train_acc * 100),
            'validation_accuracy': float(test_acc * 100),
            'backtest_accuracy': float(test_acc * 100),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(f'model_accuracy_{ticker}.json', 'w', encoding='utf-8') as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

        # 保存特徵重要性
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fi_data = {
            'ticker': ticker,
            'model_type': 'XGBoost',
            'model_accuracy': float(test_acc),
            'feature_importance': {
                row['feature']: float(row['importance'])
                for _, row in feature_importance.iterrows()
            }
        }

        with open(f'{ticker}_feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump(fi_data, f, ensure_ascii=False, indent=2)

        status = "GOOD" if test_acc >= 0.58 else "LOW"
        print(f"SUCCESS: {ticker} - {test_acc*100:.2f}% ({status})")
        return True, test_acc * 100

    except Exception as e:
        print(f"FAILED: {ticker} - {str(e)}")
        return False, 0

# 主程序
if __name__ == '__main__':
    print("="*80)
    print("批次訓練剩餘低準確度美股 XGBoost 模型")
    print("="*80)
    print(f"開始時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"總共: {len(remaining_stocks)} 支股票")
    print("="*80)

    results = []

    for i, ticker in enumerate(remaining_stocks, 1):
        print(f"\n進度: [{i}/{len(remaining_stocks)}]")
        success, acc = train_stock(ticker)

        results.append({
            'ticker': ticker,
            'success': success,
            'accuracy': acc
        })

    # 總結
    print("\n" + "="*80)
    print("批量訓練完成!")
    print("="*80)

    success_count = sum(1 for r in results if r['success'])
    print(f"成功: {success_count}/{len(remaining_stocks)}")

    # 分類結果
    good_stocks = [r for r in results if r['success'] and r['accuracy'] >= 58]
    low_stocks = [r for r in results if r['success'] and r['accuracy'] < 58]
    failed_stocks = [r for r in results if not r['success']]

    print("\n結果摘要:")
    print("-"*80)
    print(f"{'股票':<10} {'準確度':<15} {'狀態'}")
    print("-"*80)

    for r in results:
        if r['success']:
            status = "OK (>=58%)" if r['accuracy'] >= 58 else "LOW (<58%)"
            print(f"{r['ticker']:<10} {r['accuracy']:>10.2f}%    {status}")
        else:
            print(f"{r['ticker']:<10} {'失敗':<15} FAILED")

    print("-"*80)
    print(f"\n準確度 >=58%: {len(good_stocks)} 支")
    print(f"準確度 <58%: {len(low_stocks)} 支")
    print(f"訓練失敗: {len(failed_stocks)} 支")

    if good_stocks:
        print(f"\n達標股票 (>=58%):")
        for r in good_stocks:
            print(f"  - {r['ticker']}: {r['accuracy']:.2f}%")

    if low_stocks:
        print(f"\n仍低於58%:")
        for r in low_stocks:
            print(f"  - {r['ticker']}: {r['accuracy']:.2f}%")

    if failed_stocks:
        print(f"\n訓練失敗:")
        for r in failed_stocks:
            print(f"  - {r['ticker']}")

    print(f"\n結束時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
