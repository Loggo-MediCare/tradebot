"""
批量訓練 衛星/網路/國防 相關股票 XGBoost 模型
============================================
台股:
  3105 穩懋 / 2455 全新 / 3491 昇達科 / 6285 啟碁 / 2317 鴻海
  3062 建漢 / 3380 明泰 / 2313 華通 / 2367 燿華 / 3305 昇貿 / 4916 事欣科

國際股:
  ETCMY Eutelsat/OneWeb ADR / AMZN Amazon / AVGO Broadcom

注意: SpaceX 尚未上市 (私人公司), 無法取得市場資料訓練。
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
    # Taiwan stocks
    ('3105.TWO', '3105_TWO', '穩懋'),
    ('2455.TW',  '2455_TW',  '全新'),
    ('3491.TWO', '3491_TWO', '昇達科'),
    ('6285.TW',  '6285_TW',  '啟碁'),
    ('2317.TW',  '2317_TW',  '鴻海'),
    ('3062.TW',  '3062_TW',  '建漢'),
    ('3380.TW',  '3380_TW',  '明泰'),
    ('2313.TW',  '2313_TW',  '華通'),
    ('2367.TW',  '2367_TW',  '燿華'),
    ('3305.TWO', '3305_TWO', '昇貿'),
    ('4916.TWO', '4916_TWO', '事欣科'),
    # International (SpaceX is private - skipped)
    ('ETCMY',    'ETCMY',    'Eutelsat/OneWeb ADR'),
    ('AMZN',     'AMZN',     'Amazon'),
    ('AVGO',     'AVGO',     'Broadcom'),
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

        fi_df = pd.DataFrame({
            'feature': FEATURE_COLUMNS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        with open(f'{symbol}_feature_importance.json', 'w', encoding='utf-8') as f:
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
    print("批量訓練 衛星/網路/國防 相關股票 XGBoost 模型")
    print("⚠️  SpaceX 為私人公司，無公開市場資料，已略過")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for ticker, symbol, name in STOCKS:
        success, acc = train_xgboost(ticker, symbol, name)
        results.append((ticker, name, success, acc))

    print(f"\n\n{'='*70}")
    print("XGBoost 訓練完成摘要")
    print('='*70)
    print(f"{'代號':<14} {'名稱':<20} {'狀態':<4} {'測試準確度':>10}")
    print("-" * 52)
    for ticker, name, success, acc in results:
        status  = "✅" if success else "❌"
        acc_str = f"{acc:.2f}%" if success else "N/A"
        print(f"{ticker:<14} {name:<20} {status:<4} {acc_str:>10}")
    print(f"\n成功: {sum(1 for _,_,s,_ in results if s)}/{len(results)}")
    print("⚠️  SpaceX (私人公司) 已略過")
