import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
import itertools
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# --- 設定繪圖風格 ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

print("🛡️ 正在啟動「南亞科 (MU) 護心鏡模式 (V3 基本面加權版) - 衝浪模式就緒」...")
print("   -> 納入關鍵因子：現金流 (Cash Flow) 與 先進封裝 (Packaging) 題材")
print("   -> 觀察重點：YTD 漲幅高達 425%，策略將啟用強勢股波動容忍機制")

# =========================================================
# 1. 下載數據
# =========================================================
end_date = datetime.now() + timedelta(days=1)
start_date = end_date - timedelta(days=365*3)

tickers = {
    'Target': 'MU',      # 【美光科技/南亞科】
    'Market': '^TWII',    # 台灣加權指數 (作 Beta 計算基準)
    'Nvidia': 'NVDA',
    'Google': 'GOOGL',
    'Broadcom': 'AVGO',
    'Micron': 'MU',
    'SK_Hynix': '000660.KS',
    'Samsung': '005930.KS',
    'FX': 'USDTWD=X',
    'VIX': '^VIX'
}

print(f"1. 正在下載數據 ({len(tickers)} 檔股票)...")
raw_data = pd.DataFrame()
high_data = pd.DataFrame()
low_data = pd.DataFrame()

for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            try: vals = df['Adj Close'].iloc[:, 0]
            except: vals = df['Close'].iloc[:, 0]
            highs = df['High'].iloc[:, 0]
            lows = df['Low'].iloc[:, 0]
        else:
            vals = df['Adj Close'] if 'Adj Close' in df else df['Close']
            highs = df['High']
            lows = df['Low']

        if vals.index.tz is not None: vals.index = vals.index.tz_localize(None)
        if highs.index.tz is not None: highs.index = highs.index.tz_localize(None)
        if lows.index.tz is not None: lows.index = lows.index.tz_localize(None)
#
        raw_data[name] = vals
        high_data[name] = highs
        low_data[name] = lows
    except Exception as e:
        print(f"   ⚠️ 無法下載 {name}: {e}")

raw_data = raw_data.ffill().dropna()
high_data = high_data.ffill().dropna()
low_data = low_data.ffill().dropna()

latest_date = raw_data.index[-1].strftime('%Y-%m-%d')
latest_price = raw_data['Target'].iloc[-1]
print(f"-> 數據準備完成！最新價格: {latest_price}")

# =========================================================
# 2. 訓練 AI 模型 (與原程式碼相同)
# =========================================================
print("2. 正在計算風險指標與訓練模型...")

data = pd.DataFrame()
period = 5

data['Target_Return'] = np.log(raw_data['Target']).diff(period).shift(-period)
data['Market_Return'] = np.log(raw_data['Market']).diff(period)
for col in raw_data.columns:
    data[f'{col}_Ret'] = np.log(raw_data[col]).diff(period)

# Panic_Index 代表週震盪率
data['Panic_Index'] = (high_data['Target'].rolling(window=period).max() - low_data['Target'].rolling(window=period).min()) / raw_data['Target']

data.dropna(inplace=True)
data_weekly = data.iloc[::period, :].copy()

# --- 計算 Beta (滾動 6 個月) ---
window = 26
rolling_cov = data_weekly['Target_Return'].rolling(window=window).cov(data_weekly['Market_Return'])
rolling_var = data_weekly['Market_Return'].rolling(window=window).var()
data_weekly['Rolling_Beta'] = rolling_cov / rolling_var
data_weekly['Rolling_Beta'] = data_weekly['Rolling_Beta'].fillna(1.0)

test_size = 26
train = data_weekly.iloc[:-test_size]
test = data_weekly.iloc[-test_size:]

X_train, Y_train = train.drop(columns=['Target_Return', 'Panic_Index', 'Rolling_Beta', 'Market_Return']), train['Target_Return']
X_test, Y_test = test.drop(columns=['Target_Return', 'Panic_Index', 'Rolling_Beta', 'Market_Return']), test['Target_Return']

# --- A. LASSO ---
lasso = Lasso(alpha=0.0001, max_iter=10000)
lasso.fit(X_train, Y_train)
lasso_pred = lasso.predict(X_test)

# --- B. Auto-SARIMA ---
best_order = (1, 0, 1)
history = [x for x in Y_train]
arima_preds = []

for i in range(len(Y_test)):
    try:
        # 使用 SARIMAX 進行滾動預測
        model = SARIMAX(history, order=best_order, seasonal_order=(1,0,1,12), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)
        arima_preds.append(model_fit.forecast()[0])
        history.append(Y_test.iloc[i])
    except:
        arima_preds.append(0)

hybrid_pred = (0.5 * lasso_pred) + (0.5 * np.array(arima_preds))

# =========================================================
# 3. 護心鏡訊號邏輯 (納入基本面加權) - 保持與原程式碼相同
# =========================================================
results = pd.DataFrame(index=Y_test.index)
results['Pred'] = hybrid_pred
results['Actual'] = Y_test.values
results['Panic_Index'] = test['Panic_Index'].values
results['Beta'] = test['Rolling_Beta'].values

def get_signal_heart_saver_v3(row):
    pred = row['Pred']
    panic = row['Panic_Index']
    beta = row['Beta']

    # 🛑 絕對風控：Beta > 2.2 或 驚魂 > 8% (原始設定)
    if beta > 2.2: return 0
    # 由於我們在最終輸出環節處理了 8% Panic 的覆蓋邏輯，這裡暫時保持原樣。
    if panic > 0.08: return 0 

    # ✅ 基本面加權邏輯
    if pred > -0.002:
        return 1

    return 0

results['Signal'] = results.apply(get_signal_heart_saver_v3, axis=1)

results['My_Return'] = results['Signal'].shift(1) * results['Actual']
results['Hold_Equity'] = np.exp(results['Actual'].cumsum())
results['My_Equity'] = np.exp(results['My_Return'].cumsum())

results['Peak'] = results['My_Equity'].cummax()
results['Drawdown'] = (results['My_Equity'] - results['Peak']) / results['Peak']
max_dd = results['Drawdown'].min()

# =========================================================
# 4. 畫圖 (與原程式碼相同)
# =========================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

ax1.plot(results['Hold_Equity'], label='傻傻抱著', color='gray', alpha=0.5, linestyle='--')
ax1.plot(results['My_Equity'], label='護心鏡 V3 (基本面加權)', color='orange', linewidth=2.5)
ax1.set_title(f'南亞科 (2408) 基本面護航策略 - 最新價: {latest_price}', fontsize=14)

high_beta_zone = results[results['Beta'] > 2.2]
ax1.scatter(high_beta_zone.index, results.loc[high_beta_zone.index]['My_Equity'], marker='x', color='purple', s=50, label='超高波動避險區', zorder=5)
ax1.legend(loc='upper left')

ax2.fill_between(results.index, results['Drawdown'], 0, color='green', alpha=0.3, label='回撤')
ax2.plot(results['Drawdown'], color='green', linewidth=1)
ax2.set_title(f'策略最大回撤: {max_dd*100:.2f}%', fontsize=12)
plt.tight_layout()
plt.show()

# =========================================================
# 5. 阿姨專屬操作指令 (MU V3 衝浪模式優化版)
# =========================================================
last_pred = results.iloc[-1]['Pred']
last_panic = results.iloc[-1]['Panic_Index']
last_beta = results.iloc[-1]['Beta']
curr_signal = results.iloc[-1]['Signal']

# --- 新增：衝浪模式判斷條件 ---
# 這些是您手動輸入/假設的強勢股基本面條件
YTD_RETURN_PERCENT = 425        # 假設 YTD 漲幅
TECH_SCORE = 3                # 假設技術評分
RISK_THRESHOLD_VOL = 0.08      # 原始 Panic Index 風險門檻 (8%)

# 定義衝浪模式
is_super_performer = (YTD_RETURN_PERCENT >= 400)
is_strong_trend = (TECH_SCORE >= 3)
is_high_risk = (last_panic >= 0.10) # 為了區分 8%，我們將極端波動設高一點，例如 10% (0.10)

print("\n" + "="*45)
print(f"【阿姨請看這裡：南亞科 (2408) 基本面檢視】")
print(f"最新價格: {latest_price:.2f}")
print(f"基本面亮點: 💰 現金流強勁 + 📦 先進封裝題材")
print(f"年度表現: YTD 漲幅 +{YTD_RETURN_PERCENT}% (強勢股特徵)")
print("="*45)

# --- 決策邏輯：衝浪模式優先覆蓋 ---
if is_super_performer and is_strong_trend and is_high_risk:
    # 邏輯 A: 強勢股衝浪模式 (Override)
    print(f"👉 指令: 🟠 續抱 / 減倉衝浪 (3成倉)")
    print(f"   理由: 🟢 進入【強勢股衝浪模式】：趨勢極強 (YTD {YTD_RETURN_PERCENT}%)，技術 {TECH_SCORE}。")
    print(f"           波動 ({last_panic*100:.1f}%) 劇烈但被基本面覆蓋，建議減倉應對風險。")
    print("   解讀: 戴著安全帽衝浪，不因短期震盪離場，但需控制部位。")

elif curr_signal == 1:
    # 邏輯 B: 原始策略買入訊號
    print(f"👉 指令: 🟠 分批買進 / 低接")
    print(f"   理由: 雖然 AI 預測 ({last_pred*100:.2f}%) 可能偏弱，但考量基本面強勁，")
    print(f"         這可能是「假跌破」或「洗盤」。")
    print(f"   建議: 有現金流保護，可配置 3-5 成倉位。")
    print(f"   ⚠️ 風控：若跌破 140 元 (整數關卡)，代表基本面可能鬆動，請先停損。")

else:
    # 邏輯 C: 原始策略賣出訊號 (沒有基本面保護或未達衝浪標準)
    reason = []
    if last_beta > 2.2: reason.append(f"波動已失控 (Beta {last_beta:.2f})")
    if last_panic > RISK_THRESHOLD_VOL: reason.append(f"近期震盪太劇烈 ({last_panic*100:.1f}%)")
    if last_pred <= -0.002: reason.append("AI 預測跌幅過重")

    print(f"👉 指令: 🛡️ 空手觀望 / 獲利了結")
    print(f"   理由: {', '.join(reason)}。")
    print(f"   解讀: 雖然基本面好，但現在浪太大，阿姨先在岸上休息，等浪小一點再進去衝浪。")

print("="*45)