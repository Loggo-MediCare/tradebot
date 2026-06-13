import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except (AttributeError, io.UnsupportedOperation):
    pass

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 無 Tcl/Tk 環境，改用非互動式後端存檔
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# --- 設定繪圖風格 ---
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# =========================================================
# 1. 下載全球數據 (Global Radar)
# =========================================================
end_date = datetime.now()
start_date = end_date - timedelta(days=365*3) # 3年

tickers = {
    'Target': 'MU',      # 華東
    'Market': '^TWII',        # 台股
    'Nvidia': 'NVDA',         # AI 霸主
    'Google': 'GOOGL',        # Google
    'Broadcom': 'AVGO',       # 博通
    'Nanya': '2408.TW',           # 美光
    'SK_Hynix': '000660.KS',  # 海力士
    'Samsung': '005930.KS',   # 三星
    'FX': 'USDTWD=X',         # 匯率
    'VIX': '^VIX'             # 恐慌指數
}

print(f"1. 正在下載數據 ({len(tickers)} 檔股票)...")
raw_data = pd.DataFrame()

for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            vals = df['Adj Close'].iloc[:, 0] if 'Adj Close' in df.columns.get_level_values(0) else df['Close'].iloc[:, 0]
        else:
            vals = df['Adj Close'] if 'Adj Close' in df else df['Close']

        if vals.index.tz is not None: vals.index = vals.index.tz_localize(None)
        raw_data[name] = vals
    except: pass

raw_data = raw_data.ffill().dropna()
print(f"-> 數據準備完成！")

# 轉換為真正的週線資料 (W-FRI)，避免「N日報酬 + 每N天抽樣」造成的時間尺度混搭
raw_weekly = raw_data.resample('W-FRI').last().dropna()

# =========================================================
# 2. 訓練 AI 模型 (LASSO + ARIMA)
# =========================================================
print("2. 正在訓練雙引擎模型...")

data = pd.DataFrame()

# 目標與特徵 (皆為週線報酬，下週預測 = shift(-1))
data['Target_Return'] = np.log(raw_weekly['Target']).diff(1).shift(-1)
for col in raw_weekly.columns:
    data[f'{col}_Ret'] = np.log(raw_weekly[col]).diff(1)

data.dropna(inplace=True)
data_weekly = data

# 分割
test_size = 26
train = data_weekly.iloc[:-test_size]
test = data_weekly.iloc[-test_size:]

X_train, Y_train = train.drop(columns=['Target_Return']), train['Target_Return']
X_test, Y_test = test.drop(columns=['Target_Return']), test['Target_Return']

# --- A. LASSO ---
lasso = Lasso(alpha=0.0001, max_iter=10000)
lasso.fit(X_train, Y_train)
lasso_pred = lasso.predict(X_test)

# --- B. ARIMA (滾動預測) ---
history = [x for x in Y_train]
arima_preds = []
for i in range(len(Y_test)):
    try:
        model = ARIMA(history, order=(2,0,1))
        model_fit = model.fit()
        arima_preds.append(model_fit.forecast()[0])
        history.append(Y_test.iloc[i])
    except:
        arima_preds.append(0)

# --- C. 混合預測 ---
hybrid_pred = (0.5 * lasso_pred) + (0.5 * np.array(arima_preds))

# =========================================================
# 3. 計算波動率 (GARCH) - 用來算倉位，不是用來嚇跑我們
# =========================================================
try:
    residuals = Y_test - hybrid_pred
    garch = arch_model(residuals, p=1, q=1, vol='Garch', dist='Normal')
    res = garch.fit(disp='off')
    volatility = res.conditional_volatility
except:
    volatility = pd.Series([0]*len(Y_test), index=Y_test.index)

# =========================================================
# 4. 阿姨的「火箭模式」 (訊號產生器)
# =========================================================
results = pd.DataFrame(index=Y_test.index)
results['Pred'] = hybrid_pred
results['Vol'] = volatility.values
results['Y_next_week'] = Y_test.values                    # 下週報酬 (label，預測目標，尚未發生)
results['Actual_this_week'] = X_test['Target_Ret'].values # 本週已實現報酬 (決策當下已知，無未來函數)

print("\n🔥 [火箭模式] 已啟動：加入「趨勢強制令」，防止過早賣出！")

def get_signal_rocket(row):
    pred = row['Pred']                # AI 預測下週漲跌
    actual = row['Actual_this_week']  # 本週「已實現」的真實漲跌 (避免用到下週才知道的資料)

    # 邏輯 1: AI 說會漲 -> 買
    if pred > 0:
        return 1

    # 邏輯 2 (趨勢強制令):
    # AI 雖然說跌，但本週其實大漲超過 3% -> 代表趨勢超強 -> 強制買進/續抱！
    # 這能解決 "漲太多 AI 叫我賣" 的問題
    elif actual > 0.03:
        return 1

    # 邏輯 3: 真的轉弱了 -> 賣
    else:
        return 0

results['Signal'] = results.apply(get_signal_rocket, axis=1)

# 回測計算
# Signal[t-1] 是用「t-1 當下已知」的資訊決定、用來持有到 t 的部位，
# 因此對應的已實現報酬是 Actual_this_week[t] (= R[t])，而不是 Y_next_week[t] (= R[t+1])
results['My_Return'] = results['Signal'].shift(1) * results['Actual_this_week']
results['Hold_Equity'] = np.exp(results['Actual_this_week'].cumsum())
results['My_Equity'] = np.exp(results['My_Return'].cumsum())

# =========================================================
# 5. 畫圖 (成果驗收)
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(results['Hold_Equity'], label='傻傻抱著 (Buy & Hold)', color='gray', alpha=0.5)
plt.plot(results['My_Equity'], label='阿姨火箭隊 (AI Rocket)', color='red', linewidth=3)

# 標出買賣點
buys = results[results['Signal'].diff() == 1]
sells = results[results['Signal'].diff() == -1]
plt.scatter(buys.index, results.loc[buys.index]['My_Equity'], marker='^', color='green', s=120, label='買', zorder=5)
plt.scatter(sells.index, results.loc[sells.index]['My_Equity'], marker='v', color='black', s=120, label='賣', zorder=5)

plt.legend()
plt.savefig('mu_rocket_backtest.png', dpi=120, bbox_inches='tight')
print("\n📊 圖表已儲存: mu_rocket_backtest.png")

# =========================================================
# 6. 下週該怎麼做？ (操作指令)
# =========================================================
last_pred = results.iloc[-1]['Pred']
last_vol = results.iloc[-1]['Vol']
last_actual_return = results.iloc[-1]['Actual_this_week']
curr_price = raw_data['Target'].iloc[-1]
my_money = 1000000

print("\n" + "="*45)
print(f"【阿姨請看這裡：下週操作指令】")
print(f"現在股價: {curr_price:.2f} 元")
print("="*45)

# 判斷下週訊號 (使用相同的火箭邏輯)
signal = 0
reason = ""

if last_pred > 0:
    signal = 1
    reason = f"AI 預測下週會漲 {last_pred*100:.2f}%"
elif last_actual_return > 0.03:
    signal = 1
    reason = f"AI 雖保守，但本週大漲 {last_actual_return*100:.2f}%，趨勢極強，強制續抱！"
else:
    signal = 0
    reason = "AI 預測下跌，且無強勢訊號。"

if signal == 1:
    print(f"👉 指令: 🟢 買進 / 續抱！")
    print(f"   理由: {reason}")

    # 資金建議 (波動大買少點，波動小買多點)
    # 這裡我們用 20% 波動率當分界線，因為飆股波動本來就大
    if last_vol > 0.20:
        shares = int((my_money * 0.3) / curr_price)
        print(f"   建議: 風險仍高 ({last_vol*100:.1f}%)，買 {shares/1000:.1f} 張 (3成倉) 試水溫。")
    else:
        shares = int((my_money * 0.5) / curr_price) # 火箭模式最多 5 成，留點現金
        print(f"   建議: 趨勢明確，買 {shares/1000:.1f} 張 (5成倉) 坐轎！")

else:
    print(f"👉 指令: 🔴 全部賣出 / 空手觀望！")
    print(f"   理由: {reason}")