# ============================================================
# BUY 信號總分排序器 v3.0
# 目標：每一檔 BUY 信號 → 勝率分數(final_score) + 建議動作(action) + 停損/停利
# 核心： (模型準確度 × 量能 × 型態風險) + 分析師目標價/評等 + Upside Gate(倒掛懲罰)
# ============================================================

import pandas as pd
import numpy as np

# =========================
# I/O
# =========================
SIGNALS_CSV  = "taiwan_buy_signals.csv"              # 你的「今日所有 BUY 信號」輸出（自行對應欄位）
ANALYST_CSV  = "analyst_targets_from_signals_clean.csv"  # 你已經有的分析師目標價檔
OUTPUT_CSV   = "buy_ranked_with_action_and_risk.csv"

# 如果你有單獨的 VaR/CVaR 或 MonteCarlo 輸出，也可以做成 CSV 併進來（可選）
# 例如欄位：ticker, horizon_days, var95_ret, cvar95_ret, var_price, cvar_price, p_hit_var, mdd_p05, mdd_cvar95
RISK_CSV     = None  # e.g. "risk_metrics.csv" 或 None


# =========================
# Helper: normalize / clamp
# =========================
def clamp(x, lo=0.0, hi=1.0):
    if pd.isna(x):
        return np.nan
    return max(lo, min(hi, float(x)))

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def to_pct01(x):
    """
    把可能是 69.5% / 0.695 / '69.5%' / 69.5 之類都轉成 0~1
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace("%", "")
        x = safe_float(s)
    x = safe_float(x)
    if pd.isna(x):
        return np.nan
    # 若大於 1.5，視為百分比
    if x > 1.5:
        return x / 100.0
    return x


# =========================
# Analyst: rec_key → score
# =========================
REC_SCORE_MAP = {
    "strong_buy": 1.00,
    "buy":        0.80,
    "hold":       0.50,
    "sell":       0.20,
    "strong_sell":0.00,
    "none":       np.nan,
    "":           np.nan
}

def rec_key_to_score(rec_key):
    rk = (rec_key or "").strip().lower()
    return REC_SCORE_MAP.get(rk, np.nan)

def analyst_count_score(n):
    """
    覆蓋數越多越可信：0~1
    """
    if pd.isna(n):
        return np.nan
    n = float(n)
    # 0-5: 0.2, 10:0.5, 20:0.75, 30+:1.0
    if n <= 0:
        return 0.0
    if n < 5:
        return 0.2
    if n < 10:
        return 0.35
    if n < 20:
        return 0.55
    if n < 30:
        return 0.75
    return 1.0

def upside_score(upside_pct):
    """
    正向上行：+0~1；倒掛：給 0（倒掛交給 penalty 扣）
    """
    if pd.isna(upside_pct):
        return np.nan
    u = float(upside_pct)
    if u <= 0:
        return 0.0
    # 0%→0, 10%→0.5, 20%→0.75, 30%→0.9, 50%→1
    if u >= 50:
        return 1.0
    return clamp(np.log1p(u/10.0) / np.log1p(50/10.0), 0, 1)


# =========================
# Upside Gate / Penalty
# =========================
def analyst_upside_penalty(upside_pct, analyst_count):
    """
    目標價倒掛懲罰：
    - upside 缺：扣 0.12（避免 NaN 霸榜）
    - upside < 0：越倒掛扣越重，且覆蓋數越多扣越重
    """
    if pd.isna(upside_pct):
        return 0.12  # 缺資料：小扣分
    u = float(upside_pct)
    if u >= 0:
        return 0.0

    # 倒掛幅度轉成 0~0.35 的 base
    base = min(0.35, abs(u) / 100.0)

    # 覆蓋數越多越「可信」→ 扣更重
    if pd.isna(analyst_count):
        w = 0.9
    else:
        n = float(analyst_count)
        w = 1.0 if n >= 10 else 0.7

    return base * w


# =========================
# Risk level from score / optional risk metrics
# =========================
def risk_level_from_score(final_score):
    if pd.isna(final_score):
        return "未知"
    if final_score >= 0.72:
        return "中低风险"
    if final_score >= 0.58:
        return "中等风险"
    return "偏高风险"


# =========================
# Action / StopLoss / TakeProfit
# =========================
def decide_action_with_gate(final_score, upside_pct, rec_key):
    rec = (rec_key or "").lower()

    if pd.isna(upside_pct):
        return "資料不足：不動作/再查"

    if float(upside_pct) < 0:
        if rec in {"strong_buy", "buy"}:
            return "目標價倒掛：不加碼/只留核心（可小減碼控風險）"
        if rec == "hold":
            return "倒掛且持有：偏減碼/移動停利"
        return "倒掛偏空：減碼/退出"

    # upside >= 0
    if final_score >= 0.70:
        return "高勝率加碼：分批買/回檔加"
    if final_score >= 0.58:
        return "可買：分批買/回檔買"
    if final_score >= 0.50:
        return "觀望：等量能或型態確認"
    return "保守：降低曝險"

def default_stop_take(price, risk_level):
    """
    沒有 VaR 時的保守版風控（你可自行調整）
    """
    if pd.isna(price):
        return (np.nan, np.nan)

    p = float(price)

    # 越保守止損越緊；越高風險止損放寬一些避免假跌破
    if risk_level == "中低风险":
        stop = p * 0.93   # -7%
        take = p * 1.12   # +12%
    elif risk_level == "中等风险":
        stop = p * 0.92   # -8%
        take = p * 1.16   # +16%
    else:
        stop = p * 0.90   # -10%
        take = p * 1.20   # +20%

    return (stop, take)

def stop_take_from_var(price, var_price, optimistic_price=None):
    """
    若你有 MonteCarlo 的 conservative(VaR 5% price) 和 optimistic(95% price)
    stop = var_price * 0.98 (略低於 VaR 價位)
    take = optimistic_price (若缺就用 price*1.12)
    """
    if pd.isna(price):
        return (np.nan, np.nan)

    p = float(price)

    if pd.isna(var_price):
        stop = p * 0.93
    else:
        stop = float(var_price) * 0.98

    if pd.isna(optimistic_price):
        take = p * 1.12
    else:
        take = float(optimistic_price)

    return (stop, take)


# =========================
# Main scoring
# =========================
def compute_final_score(row):
    """
    你要的總分公式（可調權重）：
    base = (model_accuracy × volume_score × pattern_score)
    analyst = (upside_score + rec_score + coverage_score)/3
    final = w_base*base + w_analyst*analyst - penalty
    """
    # -------------- base (你的模型三件套) --------------
    model_acc = to_pct01(row.get("model_accuracy"))  # 0~1
    vol_score = to_pct01(row.get("volume_score"))    # 0~1（若你沒有就先用 0.5）
    pat_score = to_pct01(row.get("pattern_risk_score"))  # 0~1（風險越低分越高；若你用相反請自行改）

    # 缺欄位時給中性值
    if pd.isna(vol_score):
        vol_score = 0.5
    if pd.isna(pat_score):
        pat_score = 0.5

    if pd.isna(model_acc):
        # 沒有模型準確度就不應該排前面：給很低
        model_acc = 0.4

    base = clamp(model_acc * vol_score * pat_score, 0, 1)

    # -------------- analyst block --------------
    upside = row.get("upside_to_target_mean_%")
    n_analyst = row.get("num_analyst_opinions")
    rec_key = row.get("recommendation_key")

    u_score = upside_score(upside)
    r_score = rec_key_to_score(rec_key)
    c_score = analyst_count_score(n_analyst)

    # 缺資料 → 用平均時忽略 NaN
    analyst_parts = [u_score, r_score, c_score]
    analyst_parts = [x for x in analyst_parts if not pd.isna(x)]
    analyst = np.mean(analyst_parts) if len(analyst_parts) else 0.35

    # -------------- penalty --------------
    penalty = analyst_upside_penalty(upside, n_analyst)

    # -------------- weights --------------
    w_base = 0.62
    w_analyst = 0.38

    final = w_base * base + w_analyst * analyst - penalty
    return clamp(final, 0, 1)


def main():
    # -------- load signals --------
    sig = pd.read_csv(SIGNALS_CSV)

    # 你 signals 至少要有：ticker（或 ticker_raw）, price（現價）, model_accuracy（%或0~1）
    # 若你沒有 volume_score / pattern_risk_score，就會自動用 0.5

    # 統一 ticker 欄位名
    if "ticker" not in sig.columns and "ticker_raw" in sig.columns:
        sig["ticker"] = sig["ticker_raw"]

    # -------- load analyst --------
    ana = pd.read_csv(ANALYST_CSV)

    # 統一 ticker 欄位名
    if "ticker" not in ana.columns and "ticker_raw" in ana.columns:
        ana["ticker"] = ana["ticker_raw"]

    # 僅保留需要欄位
    keep_ana = [
        "ticker",
        "current_price",
        "target_mean_price",
        "upside_to_target_mean_%",
        "num_analyst_opinions",
        "recommendation_key",
        "recommendation_mean"
    ]
    ana = ana[[c for c in keep_ana if c in ana.columns]].copy()

    # 修正 upside 欄位名（你檔案裡是 upside_to_target_mean_%）
    if "upside_to_target_mean_%" not in ana.columns and "upside_to_target_mean_%" not in ana.columns:
        # 兼容你原本欄名 upside_to_target_mean_%
        pass
    if "upside_to_target_mean_%" in ana.columns and "upside_to_target_mean_%" not in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 兼容：你的原檔欄名就是 upside_to_target_mean_%
    if "upside_to_target_mean_%" not in ana.columns and "upside_to_target_mean_%" in ana.columns:
        pass
    if "upside_to_target_mean_%" in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 最終統一用 upside_to_target_mean_% 這個欄名（跟你一直用的一樣）
    if "upside_to_target_mean_%" in ana.columns and "upside_to_target_mean_%" not in ana.columns:
        pass

    if "upside_to_target_mean_%" in ana.columns and "upside_to_target_mean_%" not in ana.columns:
        pass

    # 如果 ana 內本來就是 upside_to_target_mean_%：
    if "upside_to_target_mean_%" in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 但你貼的檔案欄名是 upside_to_target_mean_%（尾巴有 %），我們最後保留原名：
    if "upside_to_target_mean_%" not in ana.columns and "upside_to_target_mean_%" in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 這裡直接做一個兼容：如果 ana 有 upside_to_target_mean_% 就用它
    if "upside_to_target_mean_%" in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 把欄名改回你使用的標準：upside_to_target_mean_%
    if "upside_to_target_mean_%" in ana.columns:
        ana.rename(columns={"upside_to_target_mean_%": "upside_to_target_mean_%"}, inplace=True)

    # 最簡單：偵測你檔案的真欄名
    upside_col = None
    for c in ana.columns:
        if c.strip() == "upside_to_target_mean_%":
            upside_col = c
            break
    if upside_col is None:
        raise ValueError("ANALYST_CSV 找不到欄位：upside_to_target_mean_%")

    # -------- merge --------
    df = sig.merge(ana, on="ticker", how="left", suffixes=("", "_ana"))

    # 把 analyst upside 欄位統一成你後面函數要用的 key
    df.rename(columns={upside_col: "upside_to_target_mean_%"}, inplace=True)

    # -------- optional risk merge --------
    if RISK_CSV:
        r = pd.read_csv(RISK_CSV)
        if "ticker" not in r.columns and "ticker_raw" in r.columns:
            r["ticker"] = r["ticker_raw"]
        df = df.merge(r, on="ticker", how="left")

    # -------- compute score --------
    df["final_score"] = df.apply(compute_final_score, axis=1)

    # -------- risk level --------
    df["risk_level"] = df["final_score"].apply(risk_level_from_score)

    # -------- action --------
    df["action"] = df.apply(
        lambda row: decide_action_with_gate(
            row.get("final_score", np.nan),
            row.get("upside_to_target_mean_%", np.nan),
            row.get("recommendation_key", "")
        ),
        axis=1
    )

    # -------- stop loss / take profit --------
    # 若你有 MonteCarlo/VAR 欄位：var_price / optimistic_price（可自訂欄名）
    # 這裡用 var_price, optimistic_price 當示例；沒有就用 default
    def _stop_take(row):
        price = row.get("price", row.get("current_price", np.nan))
        # 如果你 signals 的現價欄位叫 price，就會抓到；不然就用 analyst 的 current_price
        var_price = row.get("var_price", np.nan)              # 可選
        opt_price = row.get("optimistic_price", np.nan)       # 可選
        if not pd.isna(var_price) or not pd.isna(opt_price):
            return stop_take_from_var(price, var_price, opt_price)
        return default_stop_take(price, row.get("risk_level", "未知"))

    st = df.apply(_stop_take, axis=1, result_type="expand")
    st.columns = ["stop_loss", "take_profit"]
    df = pd.concat([df, st], axis=1)

    # -------- clean & sort --------
    # 你要的展示欄位（依你貼的表）
    show_cols = [
        "ticker",
        "final_score",
        "risk_level",
        "price",  # signals 的現價欄
        "target_mean_price",
        "upside_to_target_mean_%",
        "num_analyst_opinions",
        "recommendation_key",
        "model_accuracy",
        "action",
        "stop_loss",
        "take_profit",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    # 排序：final_score 由大到小
    df = df.sort_values(by=["final_score"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    # 輸出
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # 印前 30
    print(f"Saved: {OUTPUT_CSV}\n")
    print(df[["rank"] + show_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
