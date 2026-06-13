# =========================================================
# 期貨交易系統 - 多階段趨勢捕捉策略 (第三次優化版：MA200 趨勢過濾)
# =========================================================



import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import os
import subprocess
import json


warnings.filterwarnings('ignore')

print("=" * 70)
print("期貨交易系統 - 多階段趨勢捕捉策略 (第三次優化版)")
print("Futures Trading System - Multi-Phase Trend Capture (Third Optimization)")
print("=" * 70 + "\n")

#-----------CHINESE FONT SETUP-------------------------------
import matplotlib.font_manager as fm

print("正在檢查並配置中文字體...")

try:
    # 嘗試安裝並配置中文字體
    subprocess.run(['apt-get', 'update'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    subprocess.run(['apt-get', 'install', '-y', 'fonts-wqy-microhei'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
except Exception:
    pass

font_path_wqy = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
if os.path.exists(font_path_wqy):
    try:
        fm.fontManager.addfont(font_path_wqy)
    except Exception:
        pass

font_options = ['WenQuanYi Micro Hei', 'Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'Noto Sans CJK TC', 'DejaVu Sans']

selected_font = None
for font in font_options:
    if font in [f.name for f in fm.fontManager.ttflist]:
        plt.rcParams['font.sans-serif'] = [font]
        selected_font = font
        break

if not selected_font:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
print(f"✓ 已設置字體：{selected_font if selected_font else 'DejaVu Sans'}")

# ==================== DATA DOWNLOAD ====================
print("\n" + "=" * 70)
print("📊 第 1 步：下載期貨數據")
print("=" * 70 + "\n")

futures_symbol = "MU"
print(f"標的: {futures_symbol}")

try:
    futures_data = yf.download(futures_symbol, start="2025-01-01", progress=False)

    if isinstance(futures_data.columns, pd.MultiIndex):
        futures_data.columns = futures_data.columns.get_level_values(0)

    futures_data.columns = futures_data.columns.str.lower()
    futures_data.index = pd.to_datetime(futures_data.index)

    print(f"✅ 下載成功: {len(futures_data)} 個交易日")
    # print(f"   最新收盤價: ${futures_data['close'].iloc[-1]:.2f}\n")

except Exception as e:
    print(f"❌ 下載失敗: {str(e)}")
    raise

# ==================== TECHNICAL INDICATORS ====================
print("=" * 70)
print("📈 第 2 步：計算技術指標")
print("=" * 70 + "\n")

# RSI (Wilder's RSI via EWM for better performance)
def calculate_rsi(data, window=14):
    delta = data.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing: alpha = 1 / window
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


futures_data['RSI'] = calculate_rsi(futures_data['close'])

# MACD
futures_data['EMA_12'] = futures_data['close'].ewm(span=12, adjust=False).mean()
futures_data['EMA_26'] = futures_data['close'].ewm(span=26, adjust=False).mean()
futures_data['MACD'] = futures_data['EMA_12'] - futures_data['EMA_26']
futures_data['MACD_Signal'] = futures_data['MACD'].ewm(span=9, adjust=False).mean()
futures_data['MACD_Histogram'] = futures_data['MACD'] - futures_data['MACD_Signal']

# Bollinger Bands
futures_data['BB_Middle'] = futures_data['close'].rolling(window=20).mean()
bb_std = futures_data['close'].rolling(window=20).std()
futures_data['BB_Upper'] = futures_data['BB_Middle'] + (bb_std * 2)
futures_data['BB_Lower'] = futures_data['BB_Middle'] - (bb_std * 2)
futures_data['BB_Position'] = (futures_data['close'] - futures_data['BB_Lower']) / (futures_data['BB_Upper'] - futures_data['BB_Lower'])

# KD (Stochastic)
low_min = futures_data['low'].rolling(window=9).min()
high_max = futures_data['high'].rolling(window=9).max()
futures_data['RSV'] = (futures_data['close'] - low_min) / (high_max - low_min) * 100
futures_data['K'] = futures_data['RSV'].ewm(alpha=1/3, adjust=False).mean()
futures_data['D'] = futures_data['K'].ewm(alpha=1/3, adjust=False).mean()

# OBV
obv_change = np.where(futures_data['close'] > futures_data['close'].shift(1), futures_data['volume'],
             np.where(futures_data['close'] < futures_data['close'].shift(1), -futures_data['volume'], 0))
futures_data['OBV'] = pd.Series(obv_change, index=futures_data.index).cumsum()
futures_data['OBV_MA'] = futures_data['OBV'].rolling(window=20).mean()

# Moving Averages
futures_data['MA_20'] = futures_data['close'].rolling(window=20).mean()
futures_data['MA_50'] = futures_data['close'].rolling(window=50).mean()
futures_data['MA_200'] = futures_data['close'].rolling(window=200).mean()

# Volatility & ATR
futures_data['volatility'] = futures_data['close'].pct_change().rolling(window=20).std()
futures_data['tr'] = np.maximum(
    futures_data['high'] - futures_data['low'],
    np.maximum(abs(futures_data['high'] - futures_data['close'].shift(1)),
               abs(futures_data['low'] - futures_data['close'].shift(1)))
)
futures_data['ATR'] = futures_data['tr'].rolling(window=14).mean()

# Price momentum
futures_data['price_change_5d'] = futures_data['close'].pct_change(5)
futures_data['price_change_20d'] = futures_data['close'].pct_change(20)

# MA Slope (trend strength)
futures_data['MA50_slope'] = futures_data['MA_50'].diff(5) / futures_data['MA_50'].shift(5)

print(f"✅ 指標計算完成\n")

# ==================== STEP 3: THIRD OPTIMIZATION (MA200 趨勢過濾與 REVERSAL 修正) ====================
print("=" * 70)
print("🎯 第 3 步：【三次優化】融入 MA200 過濾")
print("=" * 70 + "\n")

futures_data['signal'] = 0
futures_data['signal_type'] = ''

# === PHASE 1: REVERSAL CATCHING (長期趨勢反轉捕捉) - 修正無訊號問題 ===
reversal_buy = (
    # 1. 【長期趨勢過濾】價格低於 MA200 或距離 MA200 較近 (築底區)
    (futures_data['close'] < futures_data['MA_200'] * 1.03) & # 在 MA200 的 3% 之下

    # 2. 價格必須已經回到短期均線 MA20 之上 (確認突破底部盤整區)
    (futures_data['close'] > futures_data['MA_20']) &

    # 3. KD開始黃金交叉且位置必須極低 (動能從極度疲弱中復甦)
    (futures_data['K'] > futures_data['D']) &
    (futures_data['K'].shift(1) <= futures_data['D'].shift(1)) &
    (futures_data['K'] < 30) &

    # 4. OBV 顯示資金開始流入
    (futures_data['OBV'] > futures_data['OBV_MA'])
)


# === PHASE 2: TREND FOLLOWING (順勢追蹤) - 引入 MA200 過濾，要求極高品質回檔 ===
trend_buy = (
    # 1. 【關鍵優化】價格必須在 MA200 之上 (只在長期牛市中追蹤)
    (futures_data['close'] > futures_data['MA_200']) &

    # 2. 明確多頭趨勢，且價格在短期均線之上
    (futures_data['close'] > futures_data['MA_50']) &
    (futures_data['close'] > futures_data['MA_20']) &

    # 3. 強化 MA50 斜率：必須更陡峭
    (futures_data['MA50_slope'] > 0.015) &

    # 4. KD回檔後再次金叉 (K/D 必須相對低，確保是真正的健康回檔)
    (futures_data['K'] > futures_data['D']) &
    (futures_data['K'] < 60) &

    # 5. 資金持續流入
    (futures_data['OBV'] > futures_data['OBV_MA']) &

    # 6. 短期價格動能必須仍然強勁 (20日報酬必須大於 5%)
    (futures_data['price_change_20d'] > 0.05)
)

# === PHASE 3: BREAKOUT CONFIRMATION (突破確認) - 保持高效模塊不變 ===
breakout_buy = (
    # 1. 剛突破 MA50 (過去 3 天內)
    (futures_data['close'] > futures_data['MA_50']) &
    (futures_data['close'].shift(3) < futures_data['MA_50'].shift(3)) &

    # 2. 強化量能：成交量放大必須極為顯著
    (futures_data['volume'] > futures_data['volume'].rolling(20).mean() * 1.8) &

    # 3. OBV 確認
    (futures_data['OBV'] > futures_data['OBV_MA']) &

    # 4. 強化 RSI：RSI 必須介於 65 到 75 之間
    (futures_data['RSI'] >= 65) & (futures_data['RSI'] < 75)
)

# === SELL SIGNALS (賣出訊號不變) ===
sell_condition = (
    # KD 死叉且位置偏高
    (futures_data['K'] < futures_data['D']) &
    (futures_data['K'] > 50) &

    # 趨勢轉弱
    ((futures_data['close'] < futures_data['MA_50']) |
     (futures_data['MA50_slope'] < -0.01)) &

    # 資金流出
    (futures_data['OBV'] < futures_data['OBV_MA'])
)

# Apply signals
futures_data.loc[reversal_buy, 'signal'] = 1
futures_data.loc[reversal_buy, 'signal_type'] = 'REVERSAL'

futures_data.loc[trend_buy, 'signal'] = 1
futures_data.loc[trend_buy, 'signal_type'] = 'TREND'

futures_data.loc[breakout_buy, 'signal'] = 1
futures_data.loc[breakout_buy, 'signal_type'] = 'BREAKOUT'

futures_data.loc[sell_condition, 'signal'] = -1
futures_data.loc[sell_condition, 'signal_type'] = 'SELL'

# Calculate returns
futures_data['future_return_20d'] = futures_data['close'].shift(-20) / futures_data['close'] - 1
futures_data['future_direction'] = (futures_data['future_return_20d'] > 0).astype(int)

# Backtest
futures_test = futures_data.dropna(subset=['future_return_20d'])
if len(futures_test) > 0:
    buy_signals = futures_test[futures_test['signal'] == 1]
    if len(buy_signals) > 0:
        correct_buys = ((buy_signals['future_direction'] == 1)).sum()
        buy_accuracy = (correct_buys / len(buy_signals) * 100)
        avg_return = buy_signals['future_return_20d'].mean() * 100

        print(f"✅ 【三次優化後】策略表現:")
        print(f"   • 買入訊號總數: {len(buy_signals)}")
        print(f"   • 準確率: {buy_accuracy:.2f}%")
        print(f"   • 平均報酬: {avg_return:+.2f}%")

        # 分析各類訊號表現
        print(f"\n📊 【三次優化後】訊號類型分析:")
        for sig_type in ['REVERSAL', 'TREND', 'BREAKOUT']:
            type_signals = buy_signals[buy_signals['signal_type'] == sig_type]
            if len(type_signals) > 0:
                type_acc = ((type_signals['future_direction'] == 1)).sum() / len(type_signals) * 100
                type_ret = type_signals['future_return_20d'].mean() * 100
                print(f"   • {sig_type:10s}: {len(type_signals):2d} 次, 準確率 {type_acc:.1f}%, 平均報酬 {type_ret:+.2f}%")
            else:
                 print(f"   • {sig_type:10s}:  0 次, 無訊號")


# ==================== STEP 4: VISUALIZATION ====================
print("\n" + "=" * 70)
print("📊 第 4 步：視覺化")
print("=" * 70 + "\n")

fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

# 1. Main chart
axes[0].plot(futures_data.index, futures_data['close'], label='收盤價', color='black', linewidth=1.5)
axes[0].plot(futures_data.index, futures_data['MA_200'], label='MA 200', color='purple', linewidth=2, linestyle='--', alpha=0.5)
axes[0].plot(futures_data.index, futures_data['MA_50'], label='MA 50', color='blue', linewidth=2, alpha=0.7)
axes[0].plot(futures_data.index, futures_data['MA_20'], label='MA 20', color='orange', linewidth=1, alpha=0.5)

# 標記不同類型的買入信號
reversal_signals = futures_data[futures_data['signal_type'] == 'REVERSAL']
trend_signals = futures_data[futures_data['signal_type'] == 'TREND']
breakout_signals = futures_data[futures_data['signal_type'] == 'BREAKOUT']

axes[0].scatter(reversal_signals.index, reversal_signals['close'],
                color='lime', marker='^', s=150, label='反轉訊號 (長期築底)', zorder=5, edgecolors='darkgreen', linewidths=2)
axes[0].scatter(trend_signals.index, trend_signals['close'],
                color='cyan', marker='^', s=120, label='順勢訊號 (強勢回檔)', zorder=5, edgecolors='darkblue', linewidths=1.5)
axes[0].scatter(breakout_signals.index, breakout_signals['close'],
                color='yellow', marker='^', s=120, label='突破訊號 (高量能)', zorder=5, edgecolors='orange', linewidths=1.5)

sell_signals = futures_data[futures_data['signal'] == -1]
axes[0].scatter(sell_signals.index, sell_signals['close'],
                color='red', marker='v', s=100, label='賣出訊號', zorder=5, alpha=0.7)

axes[0].set_title(f'【三次優化版】多階段趨勢捕捉系統 - {futures_symbol}', fontweight='bold', fontsize=14)
axes[0].legend(loc='upper left', fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylabel('價格 (TWD)', fontweight='bold')

# 2. OBV
axes[1].plot(futures_data.index, futures_data['OBV'], label='OBV', color='teal', linewidth=1.5)
axes[1].plot(futures_data.index, futures_data['OBV_MA'], label='OBV MA(20)', color='orange', linestyle='--', linewidth=1.5)
axes[1].fill_between(futures_data.index, futures_data['OBV'], futures_data['OBV_MA'],
                     where=(futures_data['OBV'] > futures_data['OBV_MA']), facecolor='green', alpha=0.2)
axes[1].fill_between(futures_data.index, futures_data['OBV'], futures_data['OBV_MA'],
                     where=(futures_data['OBV'] < futures_data['OBV_MA']), facecolor='red', alpha=0.2)
axes[1].set_ylabel('OBV', fontweight='bold')
axes[1].legend(loc='upper left', fontsize=8)
axes[1].grid(True, alpha=0.3)

# 3. KD
axes[2].plot(futures_data.index, futures_data['K'], label='K', color='#E67E22', linewidth=1.5)
axes[2].plot(futures_data.index, futures_data['D'], label='D', color='#2980B9', linewidth=1.5)
axes[2].axhline(y=80, color='red', linestyle=':', alpha=0.5)
axes[2].axhline(y=50, color='gray', linestyle=':', alpha=0.5)
axes[2].axhline(y=20, color='green', linestyle=':', alpha=0.5)
axes[2].fill_between(futures_data.index, 0, 100, where=(futures_data['K'] > futures_data['D']),
                     facecolor='green', alpha=0.1)
axes[2].set_ylabel('KD', fontweight='bold')
axes[2].set_ylim([0, 100])
axes[2].legend(loc='upper left', fontsize=8)
axes[2].grid(True, alpha=0.3)

# 4. MACD
axes[3].bar(futures_data.index, futures_data['MACD_Histogram'],
            color=np.where(futures_data['MACD_Histogram'] > 0, 'green', 'red'), alpha=0.4, width=1)
axes[3].plot(futures_data.index, futures_data['MACD'], color='blue', linewidth=1.5, label='MACD')
axes[3].plot(futures_data.index, futures_data['MACD_Signal'], color='red', linestyle='--', linewidth=1.5, label='Signal')
axes[3].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[3].set_ylabel('MACD', fontweight='bold')
axes[3].legend(loc='upper left', fontsize=8)
axes[3].grid(True, alpha=0.3)
axes[3].set_xlabel('日期', fontweight='bold')

plt.tight_layout()
plt.savefig('optimized_trend_system_v3.png', dpi=300, bbox_inches='tight')
print("✅ 圖表已儲存: optimized_trend_system_v3.png\n")
plt.close()

# ==================== STEP 5: FEATURE IMPORTANCE ANALYSIS ====================
print("\n" + "=" * 70)
print("🧠 第 5 步：特徵重要性分析 (ML Model)")
print("=" * 70 + "\n")

# 選擇特徵 (所有技術指標)
features = [
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
    'BB_Position', 'K', 'D', 'OBV', 'OBV_MA',
    'MA_20', 'MA_50', 'MA_200',
    'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
]

# 讓後續「系統建議(結構化輸出)」能安全引用（即使數據不足也不會 NameError）
rf_model = None
scaler = None
rf_accuracy = None
MODEL_NAME = 'RandomForestDirection'
MODEL_VERSION = 'v3'

# 排除 NaN 值
ml_data = futures_data.dropna(subset=features + ['future_direction'])


if len(ml_data) == 0:
    print("❌ 數據不足，無法執行特徵重要性分析。請檢查NaN值是否過多。")
else:
    X = ml_data[features]
    y = ml_data['future_direction']

    print(f"用於分析的數據點總數: {len(X)}")

    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False, stratify=None
    )

    # 訓練隨機森林分類器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

        # 獲取特徵重要性分數
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)


    # 預測準確度
    y_pred = rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    rf_accuracy = float(accuracy)
    print(f"模型在測試集上的預測準確率: {accuracy:.4f}\n")
    print("✅ 特徵重要性計算完成")


    print("\n" + "=" * 70)
    print("🏅 技術指標重要性排名")
    print("=" * 70)
    print(feature_importance_df.to_string(index=False))

    # 視覺化特徵重要性
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498DB')
    plt.xlabel("特徵重要性分數 (Feature Importance Score)", fontweight='bold')
    plt.ylabel("技術指標 (Technical Indicator)", fontweight='bold')
    plt.title("基於隨機森林模型的技術指標重要性", fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_v3.png', dpi=300)
    print("✅ 特徵重要性圖表已儲存: feature_importance_v3.png")
    plt.close()

# ==================== FINAL DIAGNOSIS ====================
print("\n" + "=" * 70)
print("🎯 系統診斷報告")
print("=" * 70)

curr_price = futures_data['close'].iloc[-1]
curr_ma50 = futures_data['MA_50'].iloc[-1]
curr_obv = futures_data['OBV'].iloc[-1]
curr_obv_ma = futures_data['OBV_MA'].iloc[-1]
curr_k = futures_data['K'].iloc[-1]
curr_d = futures_data['D'].iloc[-1]
curr_signal_type = futures_data['signal_type'].iloc[-1]

print(f"\n📍 當前市場狀態:")
print(f"   價格: ${curr_price:.2f}")
print(f"   MA50: ${curr_ma50:.2f} ({'多頭' if curr_price > curr_ma50 else '空頭'})")
print(f"   K值: {curr_k:.1f}, D值: {curr_d:.1f} ({'黃金交叉' if curr_k > curr_d else '死亡交叉'})")
print(f"   OBV: {'資金流入 ✓' if curr_obv > curr_obv_ma else '資金流出 ✗'}")

print(f"\n🤖 系統建議 (結構化輸出 / JSON):")

# --- 決策輸出 (Decision Output Contract) ---
signal_type = curr_signal_type if isinstance(curr_signal_type, str) and curr_signal_type else 'NONE'

action_by_signal = {
    'REVERSAL': 'BUY',
    'TREND': 'BUY',
    'BREAKOUT': 'BUY',
    'SELL': 'SELL',
    'NONE': 'WAIT'
}
action = action_by_signal.get(signal_type, 'WAIT')

reason_by_signal = {
    'REVERSAL': '長期築底信號，需確認股價已突破盤整區，建議分批進場',
    'TREND': '長期趨勢確立 (MA200之上)，高品質回檔買點，可追蹤進場',
    'BREAKOUT': '關鍵突破，極高量能確認，積極進場',
    'SELL': '趨勢轉弱，考慮出場',
    'NONE': '等待更明確訊號'
}
reason = reason_by_signal.get(signal_type, '等待更明確訊號')

blockers = []
# 常見「觀望」原因：資料不足（均線/指標尚未形成）或規則未觸發
if pd.isna(futures_data['MA_200'].iloc[-1]):
    blockers.append('INSUFFICIENT_DATA_MA200')
if pd.isna(futures_data['MA_50'].iloc[-1]):
    blockers.append('INSUFFICIENT_DATA_MA50')
if pd.isna(futures_data['K'].iloc[-1]) or pd.isna(futures_data['D'].iloc[-1]):
    blockers.append('INSUFFICIENT_DATA_KD')
if pd.isna(futures_data['OBV_MA'].iloc[-1]):
    blockers.append('INSUFFICIENT_DATA_OBV_MA')

if action == 'WAIT' and signal_type == 'NONE':
    blockers.append('NO_RULE_SIGNAL')

# --- 模型輸出（若可用）---
model_output = {
    'name': MODEL_NAME,
    'version': MODEL_VERSION,
    'predicted_label': None,
    'proba': None,
    'test_accuracy': rf_accuracy
}

confidence = 0.0
proba_up = None
proba_down = None

try:
    if rf_model is not None and scaler is not None:
        last_x = futures_data[features].iloc[-1]
        if not last_x.isna().any():
            x_last_scaled = scaler.transform([last_x.values])
            proba = rf_model.predict_proba(x_last_scaled)[0]
            cls = list(rf_model.classes_)

            if 1 in cls:
                proba_up = float(proba[cls.index(1)])
            if 0 in cls:
                proba_down = float(proba[cls.index(0)])

            if proba_up is not None and proba_down is not None:
                model_output['proba'] = {'DOWN': proba_down, 'UP': proba_up}
                model_output['predicted_label'] = 'UP' if proba_up >= 0.5 else 'DOWN'

                if action == 'BUY':
                    confidence = proba_up
                elif action == 'SELL':
                    confidence = proba_down
                else:
                    confidence = max(proba_up, proba_down)
        else:
            blockers.append('INSUFFICIENT_DATA_FOR_MODEL_FEATURES')
    else:
        blockers.append('MODEL_UNAVAILABLE')
except Exception:
    blockers.append('MODEL_PREDICTION_FAILED')

# --- 風控建議 ---
risk = {'position_size': 0, 'stop_loss': None, 'take_profit': None}
atr = futures_data['ATR'].iloc[-1]

if action == 'BUY':
    risk['position_size'] = 0.25
    if not pd.isna(atr):
        atr = float(atr)
        risk['stop_loss'] = float(curr_price - 2 * atr)
        risk['take_profit'] = float(curr_price + 4 * atr)

decision = {
    'action': action,
    'signal_type': signal_type,
    'confidence': float(confidence),
    'reason': reason,
    'model': model_output,
    'blockers': blockers,
    'risk': risk
}

print(json.dumps(decision, ensure_ascii=False, indent=2))

# 保留一行摘要，便於人類快速閱讀
action_zh = {'BUY': '進場', 'SELL': '出場', 'WAIT': '觀望'}
print(f"\n（摘要）【{action_zh.get(action, action)}】- {reason}")


print("\n" + "=" * 70)
print("💡 第三次優化重點回顧:")
print("=" * 70)
print("""
1. **反轉捕捉 (REVERSAL)**: 放寬 MA200 條件 (< 1.03 倍 MA200) 並要求股價站上 MA20，以確保訊號數量和品質。
2. **順勢追蹤 (TREND)**: 引入最重要的指標 MA200，強制要求價格必須在 MA200 之上才能買入回檔。
3. **突破確認 (BREAKOUT)**: 維持高效能高標準（1.8倍成交量，RSI 65-75）。
""")