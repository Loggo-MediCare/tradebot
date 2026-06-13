import pandas as pd
import yfinance as yf
import numpy as np

def analyze_sndk_after_wdc_exit(ticker_symbol="SNDK"):
    # 1. 抓取近期數據 (包含 WDC 出清後的波動)
    print(f"正在分析 {ticker_symbol} 籌碼面與技術指標...")
    data = yf.download(ticker_symbol, period="3mo", interval="1d")
    
    if data.empty:
        print("無法取得數據，請確認代碼是否正確。")
        return

    # 2. 正規化 Close 欄位（兼容 yfinance 回傳單層/多層欄位）
    close = data['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors='coerce')

    # 3. 計算 RSI (判斷是否過熱)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # 4. 取得最新數據
    current_price = float(close.iloc[-1])
    last_rsi = float(data['RSI'].iloc[-1])
    wdc_exit_price = 545.00  # WDC 在 2/18-19 的大規模出貨均價
    
    print("-" * 30)
    print(f"【SNDK 現況分析 - 2026/02/24】")
    print(f"當前股價: ${current_price:.2f}")
    print(f"WDC 出貨支撐位: ${wdc_exit_price}")
    print(f"RSI 指標 (14日): {last_rsi:.2f}")
    
    # 5. 核心邏輯判斷
    print("-" * 30)
    print("【交易策略評估】")
    
    # 判斷支撐
    if current_price > wdc_exit_price:
        status_support = "[OK] 股價高於 WDC 出貨價，支撐強勁。新大戶（接盤俠）目前皆為獲利狀態。"
    else:
        status_support = "[WARN] 警訊！股價跌破 WDC 出貨價，大戶可能開始套牢，恐有連鎖賣壓。"

    # 判斷熱度
    if last_rsi > 75:
        status_heat = "[HOT] 極度超買：市場情緒過熱，『抬轎的』剛走，現在進場可能是幫散戶接盤。"
    elif last_rsi < 40:
        status_heat = "[COOL] 超賣區：可能存在技術性回調的買點。"
    else:
        status_heat = "[NEUTRAL] 震盪區：目前動能尚可，不具備極端風險。"

    print(status_support)
    print(status_heat)
    
    # 6. 操作建議
    if current_price > wdc_exit_price and last_rsi < 70:
        print("\n[結論] 趨勢向上且未過熱，適合小額參與。")
    elif last_rsi > 80:
        print("\n[結論] 太熱了！建議等回測 $600 附近再考慮。")
    else:
        print("\n[結論] 建議分批進場，並將止損設在 $545。")

if __name__ == "__main__":
    analyze_sndk_after_wdc_exit()
