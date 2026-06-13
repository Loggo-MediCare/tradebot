"""
批量訓練 18 支新台股 XGBoost 模型
6146 耕興 / 6706 惠特 / 3152 環德 / 6643 M31 / 6533 晶心科 /
3680 家登 / 1717 長興 / 3008 大立光 / 6789 采鈺 / 1785 光洋科 /
3028 增你強 / 1795 美時 / 8299 群聯 / 3443 創意 / 3481 群創 /
3563 牧德 / 2481 強茂 / 3081 聯亞
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, io
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

STOCKS = [
    ('6146.TWO', '6146_TWO', '耕興'),
    ('6706.TW',  '6706_TW',  '惠特'),
    ('3152.TWO', '3152_TWO', '環德'),
    ('6643.TWO', '6643_TWO', 'M31'),
    ('6533.TW',  '6533_TW',  '晶心科'),
    ('3680.TWO', '3680_TWO', '家登'),
    ('1717.TW',  '1717_TW',  '長興'),
    ('3008.TW',  '3008_TW',  '大立光'),
    ('6789.TW',  '6789_TW',  '采鈺'),
    ('1785.TWO', '1785_TWO', '光洋科'),
    ('3028.TW',  '3028_TW',  '增你強'),
    ('1795.TW',  '1795_TW',  '美時'),
    ('8299.TWO', '8299_TWO', '群聯'),
    ('3443.TW',  '3443_TW',  '創意'),
    ('3481.TW',  '3481_TW',  '群創'),
    ('3563.TW',  '3563_TW',  '牧德'),
    ('2481.TW',  '2481_TW',  '強茂'),
    ('3081.TWO', '3081_TWO', '聯亞'),
]

FEATURE_COLUMNS = [
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'bb_position', 'K', 'D', 'obv', 'obv_ma20',
    'sma_10', 'sma_30', 'sma_50', 'sma_200',
    'volatility', 'atr',
    'price_change_5d', 'price_change_10d', 'price_change_20d',
    'ma50_slope'
]


def add_features(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']

    df['bb_middle']   = df['close'].rolling(20).mean()
    df['bb_std']      = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_middle'] + df['bb_std'] * 2
    df['bb_lower']    = df['bb_middle'] - df['bb_std'] * 2
    df['bb_position'] = ((df['close'] - df['bb_lower']) /
                         (df['bb_upper'] - df['bb_lower'] + 1e-10) * 100).fillna(50)

    low_14  = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14 + 1e-10) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    df['obv']      = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma20'] = df['obv'].rolling(20).mean()

    df['volatility'] = df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-10)

    hl  = df['high'] - df['low']
    hc  = np.abs(df['high'] - df['close'].shift())
    lc  = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()

    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / (df['sma_50'].shift(5) + 1e-10) * 100

    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)
    return df


def train_xgboost(ticker, symbol, name):
    print(f"\n{'='*70}")
    print(f"XGBoost  {ticker} ({name})")
    print('='*70)
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty or len(df) < 100:
            print("  ❌ 無資料或資料不足"); return False, 0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume',
                                'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"  下載 {len(df)} 天數據")

        df = add_features(df)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])
        print(f"  清理後 {len(df_clean)} 天")
        if len(df_clean) < 100:
            print("  ❌ 數據不足"); return False, 0

        X = df_clean[FEATURE_COLUMNS]
        y = df_clean['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=200,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', random_state=42, eval_metric='logloss'
        )
        print("  訓練中...")
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        print(f"  訓練準確度: {train_acc*100:.2f}%")
        print(f"  測試準確度: {test_acc*100:.2f}%")

        model_file = f'xgb_{symbol.lower()}_model.pkl'
        joblib.dump(model, model_file)
        print(f"  ✅ 模型已保存: {model_file}")

        acc_file = f'model_accuracy_{symbol}.json'
        with open(acc_file, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': ticker, 'model_type': 'XGBoost',
                'training_accuracy': float(train_acc * 100),
                'validation_accuracy': float(test_acc * 100),
                'backtest_accuracy':   float(test_acc * 100),
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }, f, ensure_ascii=False, indent=2)
        print(f"  ✅ 準確度已保存: {acc_file}")

        fi_file = f'{symbol}_feature_importance.json'
        fi_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        with open(fi_file, 'w', encoding='utf-8') as f:
            json.dump({
                'ticker': ticker, 'model_type': 'XGBoost',
                'model_accuracy': float(test_acc),
                'feature_importance': {
                    row['feature']: float(row['importance'])
                    for _, row in fi_df.iterrows()
                }
            }, f, ensure_ascii=False, indent=2)

        tag = '🌟 EXCELLENT' if test_acc >= 0.65 else ('✅' if test_acc >= 0.50 else '⚠️')
        print(f"  {tag} ({test_acc*100:.2f}%)")
        return True, test_acc * 100

    except Exception as e:
        print(f"  ❌ {e}")
        import traceback; traceback.print_exc()
        return False, 0


if __name__ == '__main__':
    print("=" * 70)
    print("批量訓練 18 支新台股 XGBoost 模型")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for ticker, symbol, name in STOCKS:
        success, acc = train_xgboost(ticker, symbol, name)
        results.append((ticker, name, success, acc))

    print(f"\n\n{'='*70}")
    print("XGBoost 訓練完成摘要")
    print('='*70)
    print(f"{'代號':<14} {'名稱':<12} {'狀態':<4} {'測試準確度':>10}")
    print("-" * 46)
    for ticker, name, success, acc in results:
        status  = "✅" if success else "❌"
        acc_str = f"{acc:.2f}%" if success else "N/A"
        print(f"{ticker:<14} {name:<12} {status:<4} {acc_str:>10}")
    print(f"\n成功: {sum(1 for _,_,s,_ in results if s)}/{len(results)}")
