# -*- coding: utf-8 -*-
"""
阿姨火箭隊 v2 —— 針對三個被點名的弱點進行修補：

  [修補 1] 趨勢強制令的「高空接刀」漏洞
           原版：actual > 3% 就無條件強制續抱。
           問題：在正 Gamma 震盪盤（如 MU 衝 1065 摔回 1029 那天），
                 單週暴漲反而常是橡皮筋拉最緊的位置。
           v2  ：加入「噴出否決令 (Blow-off Veto)」——
                 本週雖大漲，但 AI 強烈看跌 (pred < -2%) 且波動率正在擴張
                 → 取消強制續抱，先退到場邊 (signal = 0)。

  [修補 2] 訊號只有 1/0，無法執行「把剩下 2/7 也空掉」的決策
           v2  ：加入 -1 (做空) 訊號，即「下檔協定 Downside Protocol」：
                 AI 預測大跌 (pred < -2%) ＋ 本週已實現報酬轉負 ＋ 波動率擴張
                 三者同時成立才追空，避免在正 Gamma 穩定盤裡亂放空被軋。

  [修補 3] GARCH 的未來函數 (look-ahead)
           原版：用「整段測試期殘差」一次擬合 GARCH，
                 第 1 週的倉位卻用到了第 26 週才知道的資訊。
           v2  ：改成滾動視窗——每週只用「當下已知」的歷史報酬
                 重新擬合 GARCH 並預測下一週波動率，回測才乾淨。
"""

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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import Lasso
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# 參數區（所有閾值集中管理，方便妳做敏感度測試）
# =========================================================
PRED_SHORT_TH   = -0.02   # AI 預測下週跌幅超過 2% 才考慮放空
TREND_TH        =  0.03   # 趨勢強制令：本週實漲 > 3%
VOL_EXPAND_MULT =  1.15   # 波動率 > 過去8週均值的 1.15 倍 → 視為「擴張中」
VOL_HIGH_WEEKLY =  0.06   # 週波動 6%（約年化 43%）以上 → 高風險、縮倉
COST_PER_SIDE   =  0.001  # 換倉成本 0.1%/邊（滑價+手續費，粗估）
TEST_SIZE       = 26      # 測試 26 週
MY_MONEY        = 1_000_000

# =========================================================
# 1. 下載全球數據 (Global Radar) —— 與原版相同
# =========================================================
end_date = datetime.now()
start_date = end_date - timedelta(days=365 * 3)

tickers = {
    'Target': 'MU',
    'Market': '^TWII',
    'Nvidia': 'NVDA',
    'Google': 'GOOGL',
    'Broadcom': 'AVGO',
    'Nanya': '2408.TW',
    'SK_Hynix': '000660.KS',
    'Samsung': '005930.KS',
    'FX': 'USDTWD=X',
    'VIX': '^VIX'
}

print(f"1. 正在下載數據 ({len(tickers)} 檔股票)...")
raw_data = pd.DataFrame()
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date,
                         auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            vals = (df['Adj Close'].iloc[:, 0]
                    if 'Adj Close' in df.columns.get_level_values(0)
                    else df['Close'].iloc[:, 0])
        else:
            vals = df['Adj Close'] if 'Adj Close' in df else df['Close']
        if vals.index.tz is not None:
            vals.index = vals.index.tz_localize(None)
        raw_data[name] = vals
    except Exception:
        pass

raw_data = raw_data.ffill().dropna()
print("-> 數據準備完成！")

raw_weekly = raw_data.resample('W-FRI').last().dropna()

# =========================================================
# 2. 訓練 AI 模型 (LASSO + ARIMA) —— 與原版相同
# =========================================================
print("2. 正在訓練雙引擎模型...")

data = pd.DataFrame()
data['Target_Return'] = np.log(raw_weekly['Target']).diff(1).shift(-1)
for col in raw_weekly.columns:
    data[f'{col}_Ret'] = np.log(raw_weekly[col]).diff(1)
data.dropna(inplace=True)

train = data.iloc[:-TEST_SIZE]
test  = data.iloc[-TEST_SIZE:]
X_train, Y_train = train.drop(columns=['Target_Return']), train['Target_Return']
X_test,  Y_test  = test.drop(columns=['Target_Return']),  test['Target_Return']

lasso = Lasso(alpha=0.0001, max_iter=10000)
lasso.fit(X_train, Y_train)
lasso_pred = lasso.predict(X_test)

history = list(Y_train)
arima_preds = []
for i in range(len(Y_test)):
    try:
        model_fit = ARIMA(history, order=(2, 0, 1)).fit()
        arima_preds.append(model_fit.forecast()[0])
    except Exception:
        arima_preds.append(0.0)
    history.append(Y_test.iloc[i])

hybrid_pred = 0.5 * lasso_pred + 0.5 * np.array(arima_preds)

# =========================================================
# 3. [修補 3] 滾動 GARCH —— 每週只用「當下已知」的資料預測下週波動率
# =========================================================
print("3. 正在滾動估計 GARCH 波動率（無未來函數版）...")

# 全部「已實現」週報酬序列（Target 自己的報酬，決策當下已知）
all_ret = pd.concat([X_train['Target_Ret'], X_test['Target_Ret']])

vol_forecasts = []
n_train = len(X_train)
for i in range(TEST_SIZE):
    # 用截至「本週」為止的歷史報酬擬合，預測「下週」的條件波動率
    past = all_ret.iloc[:n_train + i + 1] * 100  # arch 慣例：用 % 報酬數值較穩定
    try:
        res = arch_model(past, p=1, q=1, vol='Garch',
                         dist='Normal', mean='Constant').fit(disp='off')
        fc = float(np.sqrt(res.forecast(horizon=1).variance.values[-1, 0])) / 100
    except Exception:
        fc = float(past.std()) / 100
    vol_forecasts.append(fc)

vol_series = pd.Series(vol_forecasts, index=Y_test.index)
# 波動率擴張基準：過去 8 週預測值的均值（不含本週 → shift(1)，再次杜絕偷看）
vol_baseline = vol_series.rolling(8, min_periods=3).mean().shift(1)
vol_baseline = vol_baseline.fillna(vol_series.expanding().mean().shift(1)).fillna(vol_series.iloc[0])

# =========================================================
# 4. [修補 1 & 2] 火箭模式 v2：三態訊號 (1 多 / 0 空手 / -1 放空)
# =========================================================
results = pd.DataFrame(index=Y_test.index)
results['Pred'] = hybrid_pred
results['Vol'] = vol_series.values
results['Vol_Base'] = vol_baseline.values
results['Y_next_week'] = Y_test.values
results['Actual_this_week'] = X_test['Target_Ret'].values

print("\n🔥 [火箭模式 v2] 已啟動：趨勢強制令加裝『噴出否決令』，並開放下檔協定放空！")

def get_signal_v2(row):
    pred   = row['Pred']
    actual = row['Actual_this_week']
    vol_expanding = row['Vol'] > row['Vol_Base'] * VOL_EXPAND_MULT

    # ── [修補 2] 下檔協定 (Downside Protocol)：三條件共振才追空 ──
    # AI 強烈看跌 + 本週已轉弱 + 波動率擴張（負 Gamma 式的加速環境）
    if pred < PRED_SHORT_TH and actual < 0 and vol_expanding:
        return -1

    # ── [修補 1] 噴出否決令 (Blow-off Veto)：
    # 本週雖暴漲 >3%，但 AI 強烈看跌且波動率正在擴張
    # → 這是「橡皮筋拉最緊」的型態，取消強制續抱，先下車觀望 ──
    if actual > TREND_TH and pred < PRED_SHORT_TH and vol_expanding:
        return 0

    # 邏輯 1：AI 看漲 → 做多/續抱
    if pred > 0:
        return 1

    # 邏輯 2：趨勢強制令（僅在未被否決時生效）
    if actual > TREND_TH:
        return 1

    # 邏輯 3：轉弱但未達放空門檻 → 空手
    return 0

results['Signal'] = results.apply(get_signal_v2, axis=1)

# =========================================================
# 回測：Signal[t-1] 持有至 t，對應已實現報酬 Actual_this_week[t]
#        Signal = -1 時，下跌即獲利；並扣除換倉成本
# =========================================================
pos = results['Signal'].shift(1).fillna(0)
turnover = pos.diff().abs().fillna(pos.abs())          # 倉位變動量 (0→1 算 1, 1→-1 算 2)
results['Cost'] = turnover * COST_PER_SIDE
results['My_Return'] = pos * results['Actual_this_week'] - results['Cost']
results['Hold_Equity'] = np.exp(results['Actual_this_week'].cumsum())
results['My_Equity'] = np.exp(results['My_Return'].cumsum())

# 簡易績效統計
def perf(r):
    eq = np.exp(r.cumsum())
    mdd = (eq / eq.cummax() - 1).min()
    sharpe = r.mean() / (r.std() + 1e-12) * np.sqrt(52)
    return eq.iloc[-1] - 1, mdd, sharpe

s_ret, s_mdd, s_sharpe = perf(results['My_Return'])
h_ret, h_mdd, h_sharpe = perf(results['Actual_this_week'])
print(f"\n📈 26週績效  火箭v2: 報酬 {s_ret*100:+.1f}% | 最大回撤 {s_mdd*100:.1f}% | Sharpe {s_sharpe:.2f}")
print(f"            傻傻抱: 報酬 {h_ret*100:+.1f}% | 最大回撤 {h_mdd*100:.1f}% | Sharpe {h_sharpe:.2f}")

# =========================================================
# 5. 畫圖：多 / 空手 / 放空 三種狀態都標出來
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(results['Hold_Equity'], label='傻傻抱著 (Buy & Hold)', color='gray', alpha=0.5)
plt.plot(results['My_Equity'], label='阿姨火箭隊 v2 (多空雙向)', color='red', linewidth=3)

sig_prev = results['Signal'].shift(1).fillna(0)
longs  = results[(results['Signal'] == 1)  & (sig_prev != 1)]
flats  = results[(results['Signal'] == 0)  & (sig_prev != 0)]
shorts = results[(results['Signal'] == -1) & (sig_prev != -1)]
plt.scatter(longs.index,  results.loc[longs.index]['My_Equity'],  marker='^', color='green',  s=120, label='做多', zorder=5)
plt.scatter(flats.index,  results.loc[flats.index]['My_Equity'],  marker='o', color='orange', s=90,  label='空手', zorder=5)
plt.scatter(shorts.index, results.loc[shorts.index]['My_Equity'], marker='v', color='black',  s=120, label='放空', zorder=5)

plt.legend()
plt.savefig('mu_rocket_v2_backtest.png', dpi=120, bbox_inches='tight')
print("\n📊 圖表已儲存: mu_rocket_v2_backtest.png")

# =========================================================
# 6. 下週操作指令（對應妳的實際部位：已空 5/7，剩 2/7 多單）
# =========================================================
last = results.iloc[-1]
curr_price = raw_data['Target'].iloc[-1]
vol_expanding_now = last['Vol'] > last['Vol_Base'] * VOL_EXPAND_MULT

print("\n" + "=" * 50)
print("【下週操作指令】(妳目前部位：已減/空 5/7，剩 2/7 多單)")
print(f"現在股價: {curr_price:.2f} | 預測下週: {last['Pred']*100:+.2f}% | "
      f"本週實現: {last['Actual_this_week']*100:+.2f}%")
print(f"GARCH 週波動: {last['Vol']*100:.1f}% (基準 {last['Vol_Base']*100:.1f}%"
      f"{'，⚠️ 擴張中' if vol_expanding_now else '，平穩'})")
print("=" * 50)

sig = get_signal_v2(last)

if sig == -1:
    print("👉 指令: ⚫ 下檔協定觸發 —— 出清剩餘 2/7，並可反手放空！")
    print(f"   理由: AI 預測大跌 {last['Pred']*100:.2f}%、本週已轉弱、波動率擴張中。")
    size = 0.3 if last['Vol'] > VOL_HIGH_WEEKLY else 0.5
    sh = int((MY_MONEY * size) / curr_price)
    print(f"   建議: 空單試單 {size*10:.0f} 成倉（約 {sh} 股），波動大用小倉。")
elif sig == 1:
    print("👉 指令: 🟢 續抱剩餘 2/7（不急著空完）！")
    reason = (f"AI 預測下週漲 {last['Pred']*100:+.2f}%" if last['Pred'] > 0
              else f"本週大漲 {last['Actual_this_week']*100:+.2f}% 且未觸發噴出否決令，趨勢仍強")
    print(f"   理由: {reason}")
    if last['Vol'] > VOL_HIGH_WEEKLY:
        print(f"   建議: 波動偏高 ({last['Vol']*100:.1f}%/週)，維持 2/7 即可，不加碼。")
    else:
        print("   建議: 波動平穩，2/7 安心續抱，等下檔協定再動手。")
else:
    if last['Actual_this_week'] > TREND_TH and last['Pred'] < PRED_SHORT_TH:
        print("👉 指令: 🟠 噴出否決令觸發 —— 趁強勢把剩餘 2/7 出掉，先不放空！")
        print("   理由: 本週暴漲但 AI 強烈看跌＋波動擴張，典型『橡皮筋拉最緊』型態。")
    else:
        print("👉 指令: 🟠 出清剩餘 2/7、空手觀望（未達放空門檻）。")
        print("   理由: AI 偏空但跌勢/波動條件未共振，不亂追空，等 -1 訊號。")

print("\n⚠️ 本腳本僅供研究參考，不構成投資建議。")
