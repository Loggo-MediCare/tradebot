"""
CIO 策略儀表板 — 可監控假設驅動的投資決策
===============================================
基於「不是評分，而是可監控假設」原則重構。

核心改變：
  1. 外部評分（82/98 分）降級為「弱訊號」，僅供參考，不參與決策權重。
  2. 賣方行為（調升目標價）翻譯為買方風險語言：追趕市場 / 擁擠交易。
  3. Bull-case 定位為「壓力測試上限」，而非基準情境。
  4. Baseline Anchor 必須附帶「假設監控儀表板」(Assumption Watchlist)。
  5. 下檔風險不用目標價，而用「regime break 條件式退出」。

使用方式：
  python cio_dashboard.py --ticker MU      # 分析單一標的（美光）
  python cio_dashboard.py --watchlist      # 掃描整個持倉清單
"""
import sys, io, os, re, json, warnings, logging, argparse, time, random, threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=ResourceWarning)
logging.getLogger('yfinance').setLevel(logging.ERROR)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= 配置區 =========================
CACHE_DIR = os.path.join(os.getcwd(), '.cache_cio')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_HOURS = 12
USE_CACHE = True
MAX_RETRIES = 3
MAX_CONCURRENT_DOWNLOADS = 2
DOWNLOAD_TIMEOUT = 20

# ========================= 弱訊號處理（原外部評分） =========================
class WeakSignal:
    """將外部評分（如 82/98 分）降級為僅供參考的弱訊號，不直接影響決策。"""
    def __init__(self, score: int = None, source: str = ""):
        self.score = score          # 原始分數，不參與權重
        self.source = source
        self.is_weak = True         # 標記為弱訊號
        self.note = "此分數為歷史regime投影，僅供參考，決策不依賴此分數。"

    def __str__(self):
        return f"[弱訊號] {self.source}: {self.score}分 — {self.note}"

# ========================= 買方風險語言（賣方行為翻譯） =========================
@dataclass
class SellSideTranslation:
    """將賣方大幅調升目標價等行為，翻譯為買方關心的風險因子。"""
    target_upgrade_pct: float          # 目標價調升幅度 (%)
    reasoning: str = "追趕市場"         # "追趕市場" 或 "共識擴散/擁擠交易"
    crowding_risk: float = 0.0         # 0-1，擁擠交易風險係數
    note: str = ""

    def describe(self) -> str:
        if self.reasoning == "追趕市場":
            return f"📈 調升{self.target_upgrade_pct:.0f}% → 行為意涵：追趕市場，可能引發短期追價但無新資訊"
        else:
            return f"⚠️ 調升{self.target_upgrade_pct:.0f}% → 共識擴散，擁擠風險上升 {self.crowding_risk:.0%}"

# ========================= Bull-case 壓力測試上限 =========================
@dataclass
class BullCaseStressTest:
    """Bull-case 不是基準情境，而是上限壓力測試。"""
    upside_price: float          # 壓力測試上限價格
    baseline_price: float        # 基準價格
    probability: float = 0.15    # 發生機率（通常很低）
    required_conditions: List[str] = field(default_factory=list)

    def stress_impact(self, current_price: float) -> str:
        upside_pct = (self.upside_price / current_price - 1) * 100
        return f"壓力測試上檔 {upside_pct:.0f}% — 僅在 {self.probability:.0%} 機率下且滿足 {len(self.required_conditions)} 項條件時可達"

# ========================= 假設監控儀表板（Assumption Watchlist） =========================
class AssumptionWatchlist:
    """可監控的假設清單：每個KPI有觸發條件與風險動作。"""

    def __init__(self):
        self.watch_items: List[Dict] = []

    def add_kpi(self, name: str, current_value: float, trigger_condition: str,
                risk_action: str, unit: str = "", direction: str = "上升"):
        """加入一個監控指標。
        direction: "上升" / "下降" — 觸發條件方向。
        """
        self.watch_items.append({
            "name": name,
            "current": current_value,
            "trigger": trigger_condition,
            "action": risk_action,
            "unit": unit,
            "direction": direction,
            "status": "正常"
        })

    def evaluate(self) -> List[Tuple[str, str]]:
        """檢查所有KPI，返回觸發的警報列表 (item_name, recommended_action)。"""
        alerts = []
        for item in self.watch_items:
            # 模擬評估邏輯（在真實系統中會從數據源更新 current）
            # 這裡僅示範框架
            pass
        return alerts

    def display(self):
        print("\n" + "═" * 80)
        print("  📋 假設監控儀表板 (Assumption Watchlist) — 每個KPI對應風險動作")
        print("═" * 80)
        for item in self.watch_items:
            print(f"\n  🔹 {item['name']}")
            print(f"      當前值: {item['current']} {item['unit']}")
            print(f"      觸發條件: {item['trigger']}")
            print(f"      ➜ 風險動作: {item['action']}")
        print("═" * 80)

# ========================= 下檔風險協議 (Downside Protocol) =========================
@dataclass
class DownsideProtocol:
    """不用目標價，用 regime 假設是否維持成立作為倉位開關。"""
    regime_assumptions: List[str]          # 核心假設清單
    breach_actions: Dict[str, str]         # 假設名稱 -> 對應降級動作
    current_weight: float = 1.0            # 1.0 = 滿倉 baseline

    def check_regime_break(self, broken_assumptions: List[str]) -> Tuple[float, str]:
        """傳入已打破的假設，返回新權重與建議。"""
        if not broken_assumptions:
            return self.current_weight, "所有假設成立，維持權重"

        # 若任一關鍵假設打破，降級
        for ass in broken_assumptions:
            if ass in self.breach_actions:
                action = self.breach_actions[ass]
                return 0.3, f"⚠️ 假設「{ass}」打破 → {action} → 權重降為 30%"
        return 0.5, "⚠️ 部分假設偏離，權重降為 50%，持續觀察"

# ========================= CIO Sanity Check =========================
def cio_sanity_check(ticker: str, current_pe: float, forward_pe: float, eps_ttm: float) -> str:
    """CIO 式估值檢核：用最少數字講最硬的話。"""
    msg = f"\n🔍 CIO Sanity Check — {ticker}\n"
    msg += f"   近四季 EPS: {eps_ttm:.2f} | 本益比: {current_pe:.1f}x\n"
    if current_pe > 30:
        msg += "   ⚠️ 估值偏高。Baseline 能成立的前提：市場相信 EPS 可持續性還會上修，且多重估值不會因供給釋放而快速再定價。\n"
        msg += "   Bull-case 能成立的前提：不只是 EPS 上修，還需『供需缺口延續年限』被市場重新錨定（例如從 1 年變 2–3 年）。\n"
    else:
        msg += "   ✅ 估值處於合理區間，但仍需監控假設變化。\n"
    return msg

# ========================= 主分析函數（整合以上概念） =========================
def analyse_cio(ticker: str, period: str = '2y') -> Dict[str, Any]:
    """
    針對單一標的（如 MU）輸出 CIO 風格分析報告：
      - 弱訊號（原外部評分）
      - 賣方行為翻譯
      - Bull-case 壓力測試
      - 假設監控儀表板實例
      - 下檔風險協議
      - CIO sanity check
    """
    # 模擬獲取財務數據（真實環境從 yfinance 或資料庫取得）
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        current_price = info.get('regularMarketPrice', 0)
        eps_ttm = info.get('trailingEps', 0)
        pe_ttm = current_price / eps_ttm if eps_ttm > 0 else 0
        forward_pe = info.get('forwardPE', 0)

        # 獲取最近季度的毛利率（模擬，因yfinance可能沒有直接欄位）
        # 實際可從 financials 取得
        gross_margin = 0.38  # placeholder
    except Exception as e:
        print(f"獲取 {ticker} 數據失敗: {e}")
        return {}

    # ---------- 1. 弱訊號 ----------
    weak_score = WeakSignal(score=82, source="某外部模型")   # 例如 82/98 分

    # ---------- 2. 賣方行為翻譯 ----------
    # 假設最近有分析師大幅調升目標價（例如 UBS 從 120 -> 180，漲幅 50%）
    sellside = SellSideTranslation(target_upgrade_pct=50.0,
                                   reasoning="追趕市場",
                                   crowding_risk=0.65)

    # ---------- 3. Bull-case 壓力測試 ----------
    # 基準價 140，壓力測試上限 1625（誇大示範，真實應合理）
    bull = BullCaseStressTest(upside_price=1625.0,
                              baseline_price=140.0,
                              probability=0.10,
                              required_conditions=[
                                  "HBM/伺服器DRAM ASP連續四季雙位數增長",
                                  "長約覆蓋率 > 85%",
                                  "同業無大幅擴產",
                                  "毛利率 > 45%"
                              ])

    # ---------- 4. 假設監控儀表板 ----------
    watchlist = AssumptionWatchlist()
    # A. 結構性需求監控
    watchlist.add_kpi("HBM/伺服器DRAM ASP 走勢", current_value=12.5, unit="美元/GB",
                      trigger_condition="連續兩季低於公司指引下緣",
                      risk_action="降級 baseline 權重 50%，啟動避險")
    watchlist.add_kpi("長約覆蓋率 (3-5年協議)", current_value=0.82, unit="%",
                      trigger_condition="覆蓋率停滯或下滑",
                      risk_action="降至觀察池，減少曝險 30%")
    # B. 結構性供給監控
    watchlist.add_kpi("同業擴產/資本開支節奏", current_value=1.0, unit="倍（前移程度）",
                      trigger_condition="擴產前移導致供需缺口收斂加速",
                      risk_action="bull-case 權重回收至情境性曝險，主權重減半")
    watchlist.add_kpi("先進製程/封裝良率", current_value=0.92, unit="%",
                      trigger_condition="良率異常下降超過 5%",
                      risk_action="立即減碼 80%，重新審視假設")
    # C. 財務落地監控
    watchlist.add_kpi("毛利率維持高檔區間", current_value=gross_margin, unit="%",
                      trigger_condition="毛利率連續兩季低於 35%",
                      risk_action="估值容忍度下修，權重降至 40%")
    watchlist.add_kpi("CapEx → 產出/獲利轉換效率", current_value=0.25, unit="FCF/CapEx",
                      trigger_condition="效率低於 0.15 且 CapEx 持續走高",
                      risk_action="降級為觀察標的，暫停加倉")

    # ---------- 5. 下檔風險協議 ----------
    regime_assumptions = [
        "AI 記憶體需求結構性增長 (HBM/伺服器DRAM)",
        "供給端無大幅擴產前移",
        "長約覆蓋率維持 80% 以上且持續擴張",
        "毛利率維持 38% 以上新平台水準"
    ]
    breach_actions = {
        "AI 記憶體需求結構性增長": "回歸傳統循環，權重降至 20%",
        "供給端無大幅擴產前移": "降低 bull-case 權重，主倉位減半",
        "長約覆蓋率維持 80% 以上且持續擴張": "權重降至 30%，等待下一季驗證",
        "毛利率維持 38% 以上新平台水準": "下修目標價 cluster，啟動每週審查"
    }
    downside = DownsideProtocol(regime_assumptions, breach_actions, current_weight=1.0)

    # ---------- 6. CIO sanity check ----------
    sanity = cio_sanity_check(ticker, pe_ttm, forward_pe, eps_ttm)

    # ---------- 彙整 ----------
    result = {
        "ticker": ticker,
        "price": current_price,
        "weak_signal": weak_score,
        "sellside_translation": sellside,
        "bull_stress": bull,
        "watchlist": watchlist,
        "downside_protocol": downside,
        "sanity_check": sanity,
        "timestamp": datetime.now().isoformat()
    }
    return result

# ========================= 列印 CIO 儀表板 =========================
def print_cio_dashboard(analysis: Dict[str, Any]):
    if not analysis:
        print("無分析結果")
        return

    ticker = analysis['ticker']
    price = analysis['price']
    print("\n" + "═" * 80)
    print(f"  🧠 CIO 策略儀表板 — {ticker}  (基準日: {analysis['timestamp'][:10]})")
    print(f"  當前價格: ${price:.2f}")
    print("═" * 80)

    # 1. 弱訊號
    weak = analysis['weak_signal']
    print(f"\n📉 外部評分處理: {weak}")

    # 2. 賣方行為翻譯
    sell = analysis['sellside_translation']
    print(f"\n📊 賣方行為解讀: {sell.describe()}")

    # 3. Bull-case 壓力測試
    bull = analysis['bull_stress']
    print(f"\n🚀 Bull-case 壓力測試上限: ${bull.upside_price:.2f}")
    print(f"   發生機率: {bull.probability:.0%}")
    print(f"   必要條件: {', '.join(bull.required_conditions)}")
    print(f"   {bull.stress_impact(price)}")

    # 4. 假設監控儀表板
    analysis['watchlist'].display()

    # 5. 下檔風險協議
    downside = analysis['downside_protocol']
    print("\n" + "═" * 80)
    print("  🛡️ 下檔風險協議 (Downside Protocol) — 不用目標價，用 regime 假設")
    print("═" * 80)
    print("核心假設清單:")
    for i, ass in enumerate(downside.regime_assumptions, 1):
        print(f"   {i}. {ass}")
    print("\n假設打破時對應動作:")
    for ass, action in downside.breach_actions.items():
        print(f"   ❌ {ass} → {action}")
    # 模擬檢查
    broken = []  # 此處可實際從 watchlist 評估哪些假設已偏離
    new_weight, msg = downside.check_regime_break(broken)
    print(f"\n目前權重: {downside.current_weight:.0%} → 檢查結果: {msg}")

    # 6. CIO sanity check
    print(analysis['sanity_check'])

    # 7. 結語（買方決策原則）
    print("═" * 80)
    print("  📌 投資決策原則（改寫自原結語）")
    print("  我們將賣方評分視為噪音較高的輔助訊號；決策權重取決於假設是否透明、")
    print("  可監控、且具容錯率，並以情境用途（baseline vs stress test）決定")
    print("  資本配置大小與風險邊界。")
    print("═" * 80)

# ========================= 命令行入口 =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIO 策略儀表板 — 可監控假設驅動")
    parser.add_argument("--ticker", type=str, default="MU", help="分析單一股票代號 (預設 MU)")
    parser.add_argument("--watchlist", action="store_true", help="掃描完整持倉清單")
    args = parser.parse_args()

    if args.watchlist:
        # 範例持倉清單（可自行擴充）
        portfolio = ["MU", "NVDA", "TSM", "AMZN", "GOOGL"]
        for t in portfolio:
            print(f"\n\n{'='*80}\n分析 {t}\n{'='*80}")
            res = analyse_cio(t)
            print_cio_dashboard(res)
            time.sleep(1)
    else:
        res = analyse_cio(args.ticker)
        print_cio_dashboard(res)