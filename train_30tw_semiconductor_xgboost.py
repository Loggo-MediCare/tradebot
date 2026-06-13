"""
批量訓練 30 支台灣半導體/IC設計/封測 XGBoost 模型
=================================================
2330 台積電 / 2454 聯發科 / 2303 聯電 / 3034 聯詠 / 6415 矽力-KY
2368 金像電 / 3443 創意 / 6488 環球晶 / 6223 旺矽 / 3529 力旺
6510 精測 / 6669 緯穎 / 8299 群聯 / 1717 長興 / 3711 日月光投控
2379 瑞昱 / 3037 欣興 / 5483 中美晶 / 5347 世界 / 3653 健策
5269 祥碩 / 3035 智原 / 2371 大同 / 5434 崇越 / 6414 樺漢
2449 京元電子 / 3189 景碩 / 6239 力成 / 4915 致伸 / 2408 南亞科
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
    ('2330.TW',  '2330_TW',  '台積電'),
    ('2454.TW',  '2454_TW',  '聯發科'),
    ('2303.TW',  '2303_TW',  '聯電'),
    ('3034.TW',  '3034_TW',  '聯詠'),
    ('6415.TW',  '6415_TW',  '矽力-KY'),
    ('2368.TW',  '2368_TW',  '金像電'),
    ('3443.TW',  '3443_TW',  '創意'),
    ('6488.TWO', '6488_TWO', '環球晶'),
    ('6223.TWO', '6223_TWO', '旺矽'),
    ('3529.TW',  '3529_TW',  '力旺'),
    ('6510.TWO', '6510_TWO', '精測'),
    ('6669.TW',  '6669_TW',  '緯穎'),
    ('8299.TWO', '8299_TWO', '群聯'),
    ('1717.TW',  '1717_TW',  '長興'),
    ('3711.TW',  '3711_TW',  '日月光投控'),
    ('2379.TW',  '2379_TW',  '瑞昱'),
    ('3037.TW',  '3037_TW',  '欣興'),
    ('5483.TWO', '5483_TWO', '中美晶'),
    ('5347.TW',  '5347_TW',  '世界'),
    ('3653.TW',  '3653_TW',  '健策'),
    ('5269.TW',  '5269_TW',  '祥碩'),
    ('3035.TW',  '3035_TW',  '智原'),
    ('2371.TW',  '2371_TW',  '大同'),
    ('5434.TW',  '5434_TW',  '崇越'),
    ('6414.TW',  '6414_TW',  '樺漢'),
    ('2449.TW',  '2449_TW',  '京元電子'),
    ('3189.TW',  '3189_TW',  '景碩'),
    ('6239.TW',  '6239_TW',  '力成'),
    ('4915.TW',  '4915_TW',  '致伸'),
    ('2408.TW',  '2408_TW',  '南亞科'),
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
    df['rsi']         = 100 - (100 / (1 + gain / (loss + 1e-10)))
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
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / (df['sma_50'].shift(5) + 1e-10) * 100
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.02).astype(int)
    return df


def train_xgboost(ticker, symbol, name):
    print(f"\n{'='*70}\nXGBoost  {ticker} ({name})\n{'='*70}")
    try:
        df = yf.download(ticker, start='2015-01-01', end='2026-12-31', progress=False)
        if df.empty or len(df) < 100:
            print("  ❌ 無資料或資料不足"); return False, 0
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume',
                                'Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"  下載 {len(df)} 天數據")
        df = add_features(df)
        df_clean = df.dropna(subset=FEATURE_COLUMNS + ['target'])
        print(f"  清理後 {len(df_clean)} 天")
        if len(df_clean) < 100:
            print("  ❌ 數據不足"); return False, 0

        X = df_clean[FEATURE_COLUMNS]; y = df_clean['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBClassifier(
            max_depth=5, learning_rate=0.05, n_estimators=200,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            objective='binary:logistic', random_state=42, eval_metric='logloss'
        )
        print("  訓練中...")
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        print(f"  訓練準確度: {train_acc*100:.2f}%  |  測試準確度: {test_acc*100:.2f}%")

        joblib.dump(model, f'xgb_{symbol.lower()}_model.pkl')
        with open(f'model_accuracy_{symbol}.json', 'w', encoding='utf-8') as f:
            json.dump({'symbol': ticker, 'model_type': 'XGBoost',
                       'training_accuracy': float(train_acc*100),
                       'validation_accuracy': float(test_acc*100),
                       'backtest_accuracy': float(test_acc*100),
                       'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                      f, ensure_ascii=False, indent=2)

        fi_df = pd.DataFrame({'feature': FEATURE_COLUMNS,
                               'importance': model.feature_importances_}
                             ).sort_values('importance', ascending=False)
        with open(f'{symbol}_feature_importance.json', 'w', encoding='utf-8') as f:
            json.dump({'ticker': ticker, 'model_type': 'XGBoost',
                       'model_accuracy': float(test_acc),
                       'feature_importance': {r['feature']: float(r['importance'])
                                              for _, r in fi_df.iterrows()}},
                      f, ensure_ascii=False, indent=2)

        tag = '🌟 EXCELLENT' if test_acc >= 0.65 else ('✅' if test_acc >= 0.50 else '⚠️')
        print(f"  {tag} ({test_acc*100:.2f}%)")
        return True, test_acc * 100
    except Exception as e:
        print(f"  ❌ {e}"); import traceback; traceback.print_exc(); return False, 0


if __name__ == '__main__':
    print("=" * 70)
    print("批量訓練 30 支台灣半導體/IC設計/封測 XGBoost 模型")
    print(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []
    for ticker, symbol, name in STOCKS:
        success, acc = train_xgboost(ticker, symbol, name)
        results.append((ticker, name, success, acc))

    print(f"\n\n{'='*70}\nXGBoost 訓練完成摘要\n{'='*70}")
    print(f"{'代號':<14} {'名稱':<12} {'狀態':<4} {'測試準確度':>10}")
    print("-" * 44)
    for ticker, name, success, acc in results:
        print(f"{ticker:<14} {name:<12} {'✅' if success else '❌':<4} {f'{acc:.2f}%' if success else 'N/A':>10}")

    ok = [r for r in results if r[2]]
    print(f"\n成功: {len(ok)}/{len(results)}")
    if ok:
        best = sorted(ok, key=lambda x: x[3], reverse=True)[:5]
        print("\nTop 5 準確度:")
        for t, n, _, a in best:
            tag = ' 🌟' if a >= 65 else ''
            print(f"  {t:<14} {n:<12} {a:.2f}%{tag}")
