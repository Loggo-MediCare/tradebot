import yfinance as yf
import pandas as pd
import numpy as np

def get_ai_agent_chain_analysis():
    # AI Agent × NVDA 生態 × 台股落地
    tickers = {
        "NVDA": "NVIDIA (AI Agent Platform)",
        "MU": "Micron (HBM / AI Memory)",
        "TSM": "TSMC (Foundry / SiPh)",
        "2368.TW": "GCE (AI Server PCB)",
        "2317.TW": "Foxconn (AI System Deployment)"
    }

    results = []

    print("📊 正在抓取 AI Agent 生態鏈數據...")

    for symbol, role in tickers.items():
        try:
            t = yf.Ticker(symbol)
            info = t.info

            price = info.get("currentPrice") or info.get("regularMarketPrice")
            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")

            trailing_pe = price / trailing_eps if trailing_eps and trailing_eps > 0 else np.nan
            forward_pe = price / forward_eps if forward_eps and forward_eps > 0 else np.nan

            results.append({
                "代號": symbol,
                "角色定位": role,
                "股價": f"{price:.2f}" if price else "N/A",
                "Trailing PE": f"{trailing_pe:.1f}" if not np.isnan(trailing_pe) else "N/A",
                "Forward PE": f"{forward_pe:.1f}" if not np.isnan(forward_pe) else "N/A",
                "估值語言": (
                    "平台型溢價" if symbol == "NVDA" else
                    "結構成長" if symbol == "MU" else
                    "核心底座" if symbol == "TSM" else
                    "工程瓶頸" if symbol == "2368.TW" else
                    "系統落地"
                )
            })

        except Exception as e:
            print(f"⚠️ 無法抓取 {symbol}: {e}")

    df = pd.DataFrame(results)

    print("\n" + "="*90)
    print("🤖 AI Agent × NVDA 生態 × 台股落地｜估值比較表")
    print("="*90)
    print(df.to_string(index=False))
    print("="*90)
    print("📌 重點：這不是誰 PE 低就買，而是誰『站在 AI Agent 商業化路徑上』")
    
    return df


if __name__ == "__main__":
    get_ai_agent_chain_analysis()
