# ============================================================
# BUY 信號總分排序器 v1.0
# 模型準確度 × 量能 × 型態風險 + 分析師目標價/評等 + VaR/CVaR/MDD
# 輸出：勝率分數 + 建議動作 + 停損/停利
# ============================================================

import re
import math
import numpy as np
import pandas as pd
import yfinance as yf

# -------- 你的檔案路徑（依需要改）--------
SIGNAL_TXTS = [
    "taiwan_signals_output_202601151615.txt",
    "US_signals_output_202601151712.txt",
]
ANALYST_CSV = "analyst_targets_fixed_blind_point.csv"  # 你已經產出的修正版
OUTPUT_CSV = "buy_signals_ranked_with_risk.csv"

# -------- 風控參數 --------
HORIZON_DAYS = 22
SIMS = 20000
CONF = 0.95
LOOKBACK_YEARS = 5  # 抓歷史報酬用
SEED = 42

# -------- 權重（你之後可調）--------
W = {
    "model": 0.35,     # 模型準確度/信心
    "volume": 0.10,    # 量能確認
    "pattern": 0.15,   # 型態風險（越低風險越高分）
    "analyst": 0.20,   # 分析師（目標價/評等/覆蓋數）
    "risk": 0.20,      # VaR/CVaR/觸及機率/MDD
}

# ============================================================
# 1) 解析 txt：抓出 BUY 票 + 可能的分數/量能資訊
# ============================================================

def parse_signals_from_txt(txt: str) -> pd.DataFrame:
    """
    解析信號輸出檔案，支援多種格式：
    1. 台股/美股區塊格式：股票: 1101.TW + 🟢 信号: 买入 (BUY)
    2. 美股格式：Amkor (AMKR): Signal is BUY (Score 0.39)
    """
    rows = []

    # === 方法1：按區塊解析（台股/美股信號生成器格式）===
    # 使用 "进度:" 分割區塊
    progress_pattern = r"进度: \[(\d+)/\d+\]"
    progress_matches = list(re.finditer(progress_pattern, txt))

    if progress_matches:
        for i, match in enumerate(progress_matches):
            start_pos = match.end()
            if i + 1 < len(progress_matches):
                end_pos = progress_matches[i + 1].start()
            else:
                end_pos = len(txt)

            block = txt[start_pos:end_pos]

            # 從區塊中提取 ticker
            ticker = None
            # 格式1: 股票: 1101.TW
            m_ticker = re.search(r"股票:\s*(\d{4}\.TW|\w+)", block)
            if m_ticker:
                ticker = m_ticker.group(1).upper()
            else:
                # 格式2: 运行: 1101 台泥
                m_ticker2 = re.search(r"运行:\s*(\d{4})", block)
                if m_ticker2:
                    ticker = f"{m_ticker2.group(1)}.TW"
                else:
                    # 格式3: 美股 AAPL AI 交易信号
                    m_ticker3 = re.search(r"(?:美股|US)\s+(\w+)\s+AI", block)
                    if m_ticker3:
                        ticker = m_ticker3.group(1).upper()

            if not ticker:
                continue

            # 從區塊中提取信號
            # 格式: 🟢 信号: 买入 (BUY) 或 🔴 信号: 卖出 (SELL) 或 🟡 信号: 持有 (HOLD)
            sig = None
            if re.search(r"信号:\s*买入|Signal.*BUY", block, re.IGNORECASE):
                sig = "BUY"
            elif re.search(r"信号:\s*卖出|Signal.*SELL", block, re.IGNORECASE):
                sig = "SELL"
            elif re.search(r"信号:\s*持有|Signal.*HOLD", block, re.IGNORECASE):
                sig = "HOLD"

            if not sig:
                continue

            # 提取模型強度
            model_score = np.nan
            m_score = re.search(r"AI\s*模型强度:\s*([0-9.]+)\s*/\s*1\.00", block)
            if m_score:
                model_score = float(m_score.group(1))

            # 提取量比
            volume_ratio = np.nan
            m_vol = re.search(r"量比:\s*([0-9.]+)x", block)
            if m_vol:
                volume_ratio = float(m_vol.group(1))

            # 提取 RSI
            rsi = np.nan
            m_rsi = re.search(r"RSI\s*\(14\):\s*([0-9.]+)", block)
            if m_rsi:
                rsi = float(m_rsi.group(1))

            rows.append({
                "ticker": ticker,
                "signal": sig,
                "model_score": model_score,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "raw_line": block[:200].strip(),
            })

    # === 方法2：逐行解析（舊格式兼容）===
    if not rows:
        # 常見格式示例：
        # "Amkor (AMKR): Signal is BUY (Score 0.39)."
        pat_core = re.compile(
            r"(?P<name>.+?)\s*\((?P<ticker>[A-Z0-9\.\-]+)\)\s*:\s*Signal\s*is\s*(?P<sig>BUY|HOLD|SELL)\s*(?:\((?:Score|Strength)\s*(?P<score>[0-9]*\.?[0-9]+)\))?",
            re.IGNORECASE
        )

        # 也處理像 "Ticker: NVDA  Signal: BUY  Score: 0.71" 這類
        pat_alt = re.compile(
            r"\b(?P<ticker>[A-Z0-9\.\-]{1,12})\b.*?\bSignal\b.*?\b(?P<sig>BUY|HOLD|SELL)\b.*?(?:(?:Score|Strength)\s*[:=]?\s*(?P<score>[0-9]*\.?[0-9]+))?",
            re.IGNORECASE
        )

        # 量能倍數
        pat_vol = re.compile(r"(?P<vr>[0-9]*\.?[0-9]+)\s*x\s*(?:average|avg)", re.IGNORECASE)

        # RSI
        pat_rsi = re.compile(r"\bRSI\b\s*(?P<rsi>[0-9]*\.?[0-9]+)", re.IGNORECASE)

        lines = txt.splitlines()

        for i, line in enumerate(lines):
            m = pat_core.search(line)
            if not m:
                m = pat_alt.search(line)

            if m:
                ticker = m.group("ticker").upper()
                sig = m.group("sig").upper()
                score = m.group("score")
                score = float(score) if score is not None else np.nan

                window = "\n".join(lines[i:i+8])
                vm = pat_vol.search(window)
                rm = pat_rsi.search(window)

                volume_ratio = float(vm.group("vr")) if vm else np.nan
                rsi = float(rm.group("rsi")) if rm else np.nan

                rows.append({
                    "ticker": ticker,
                    "signal": sig,
                    "model_score": score,
                    "volume_ratio": volume_ratio,
                    "rsi": rsi,
                    "raw_line": line.strip(),
                })

    if not rows:
        return pd.DataFrame(columns=["ticker","signal","model_score","volume_ratio","rsi","raw_line"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="last")
    return df


def load_all_signals(files) -> pd.DataFrame:
    all_df = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        df = parse_signals_from_txt(txt)
        if not df.empty:
            df["source_file"] = fp
            all_df.append(df)
    if not all_df:
        return pd.DataFrame(columns=["ticker","signal","model_score","volume_ratio","rsi","source_file"])
    return pd.concat(all_df, ignore_index=True)


# ============================================================
# 2) 讀分析師目標價檔（你已經做過清洗/盲點修正）
# ============================================================

def load_analyst(analyst_csv: str) -> pd.DataFrame:
    a = pd.read_csv(analyst_csv)
    # 期待欄位至少有：
    # ticker_raw, current_price, target_mean_price, target_high_price(若有), upside_to_target_mean_%, recommendation_key, num_analyst_opinions
    # 若欄名略不同，就做容錯
    # Map CSV column names to expected names
    colmap = {
        "ticker_raw": "ticker",
        "Ticker": "ticker",
        "Price": "current_price",
        "Target (Avg)": "target_mean_price",
        "Target (High)": "target_high_price",
        "Upside (Avg) %": "upside_to_target_mean_%",
        "Verdict": "recommendation_key",  # Use Verdict (text) not Rating Score (numeric)
        "Analysts": "num_analyst_opinions",
    }
    a = a.rename(columns=colmap, errors="ignore")
    a["ticker"] = a["ticker"].astype(str).str.upper()
    return a


# ============================================================
# 3) 快速風控：bootstrap Monte Carlo（22天路徑）
#    產出：期末 VaR/CVaR（報酬與價格） + 觸及 VaR 機率 + MDD 分布分位數
# ============================================================

def bootstrap_paths(prices: pd.Series, horizon=22, sims=20000, seed=42):
    """
    使用歷史日對數報酬做 bootstrap，產出 sims 條價格路徑（horizon+1 長度）。
    """
    np.random.seed(seed)
    prices = prices.dropna()
    # Handle both Series and single-element cases
    last_val = prices.iloc[-1]
    last_price = float(last_val.iloc[0]) if hasattr(last_val, 'iloc') else float(last_val)

    logret = np.log(prices).diff().dropna().values.flatten()  # Ensure 1D
    if len(logret) < 100:
        raise ValueError("Not enough return history for bootstrap.")

    idx = np.random.randint(0, len(logret), size=(sims, horizon))
    sampled = logret[idx]  # sims x horizon

    # 累積 log-return
    cum = np.cumsum(sampled, axis=1)
    paths = last_price * np.exp(cum)
    # 加上 t0
    paths = np.concatenate([np.full((sims, 1), last_price), paths], axis=1)  # sims x (horizon+1)
    return last_price, paths


def risk_metrics_for_ticker(ticker: str, horizon=22, sims=20000, conf=0.95, seed=42, lookback_years=5):
    """
    回傳 dict：
    - last_price
    - var_price (期末價格 5%分位)
    - cvar_price (期末價格<=var的平均)
    - var_ret / cvar_ret (期末報酬分位)
    - hit_var_prob (期間任一天跌破var_price機率)
    - mdd_p50/p25/p10/p05, mdd_cvar95
    """
    period = f"{lookback_years}y"
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data.empty or "Close" not in data.columns:
        raise ValueError(f"yfinance no data: {ticker}")

    close = data["Close"].dropna()
    # Handle case where yfinance returns DataFrame instead of Series
    if hasattr(close, 'squeeze'):
        close = close.squeeze()
    last_price, paths = bootstrap_paths(close, horizon=horizon, sims=sims, seed=seed)

    final_prices = paths[:, -1]
    # 期末價格分位
    var_price = float(np.percentile(final_prices, (1-conf)*100))  # 5%
    cvar_price = float(final_prices[final_prices <= var_price].mean())

    # 期末報酬
    final_ret = final_prices / last_price - 1.0
    var_ret = float(np.percentile(final_ret, (1-conf)*100))
    cvar_ret = float(final_ret[final_ret <= var_ret].mean())

    # 期間觸及 var_price 機率（任一天跌破）
    hit = (paths[:, 1:] <= var_price).any(axis=1)
    hit_prob = float(hit.mean())

    # MDD
    running_max = np.maximum.accumulate(paths, axis=1)
    dd = paths / running_max - 1.0  # <=0
    mdd = dd.min(axis=1)  # 每條路徑最深回撤

    mdd_p50 = float(np.percentile(mdd, 50))
    mdd_p25 = float(np.percentile(mdd, 25))
    mdd_p10 = float(np.percentile(mdd, 10))
    mdd_p05 = float(np.percentile(mdd, (1-conf)*100))
    mdd_cvar = float(mdd[mdd <= mdd_p05].mean())

    return {
        "last_price": last_price,
        "var_price": var_price,
        "cvar_price": cvar_price,
        "var_ret": var_ret,
        "cvar_ret": cvar_ret,
        "hit_var_prob": hit_prob,
        "mdd_p50": mdd_p50,
        "mdd_p25": mdd_p25,
        "mdd_p10": mdd_p10,
        "mdd_p05": mdd_p05,
        "mdd_cvar95": mdd_cvar,
    }


# ============================================================
# 4) 分數設計：把每個維度壓到 0~1，再組合成 0~100
# ============================================================

def clip01(x):
    if pd.isna(x): return np.nan
    return float(max(0.0, min(1.0, x)))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def score_model(model_score):
    # 若你 model_score 本來就是 0~1：直接用；缺值給 0.5
    if pd.isna(model_score):
        return 0.5
    return clip01(model_score)

def score_volume(volume_ratio):
    # 量能倍數：1x=0.5；2x以上加分；太低扣分
    if pd.isna(volume_ratio):
        return 0.5
    # 用 log 壓縮，避免 10x 變成爆分
    v = math.log1p(volume_ratio) / math.log1p(5)  # 5x 近似=1
    return clip01(v)

def score_pattern(rsi):
    # RSI 越高越容易過熱，扣分；缺值給0.6
    if pd.isna(rsi):
        return 0.6
    if rsi >= 85: return 0.20
    if rsi >= 75: return 0.35
    if rsi >= 70: return 0.45
    if rsi <= 30: return 0.70  # 超賣反而有利（但仍保守）
    return 0.60

def score_analyst(upside_mean_pct, rec_key, n_analysts):
    # upside：用 sigmoid 拉伸（10%~20%很重要）
    # rec：strong_buy>buy>hold
    # n：覆蓋數越多越可信
    if pd.isna(upside_mean_pct):
        upside_s = 0.40
    else:
        # 以 0%為中心，20%大概到 0.8~0.9
        upside_s = sigmoid((upside_mean_pct - 5) / 7)

    rec = "" if pd.isna(rec_key) else str(rec_key).lower()
    rec_s = 0.5
    if rec == "strong_buy": rec_s = 0.85
    elif rec == "buy": rec_s = 0.70
    elif rec == "hold": rec_s = 0.45
    elif rec == "sell": rec_s = 0.20

    if pd.isna(n_analysts):
        n_s = 0.45
    else:
        n = float(n_analysts)
        if n >= 40: n_s = 0.85
        elif n >= 20: n_s = 0.70
        elif n >= 10: n_s = 0.60
        else: n_s = 0.45

    # 分析師總分：upside 50% + rec 30% + 覆蓋 20%
    return clip01(0.50*upside_s + 0.30*rec_s + 0.20*n_s)

def score_risk(var_ret, cvar_ret, hit_var_prob, mdd_p05):
    """
    風控：越安全越高分
    var_ret/cvar_ret/mdd 是負數（例如 -0.11）
    hit_var_prob 是 0~1
    """
    # 缺資料 → 中性
    if any(pd.isna(x) for x in [var_ret, cvar_ret, hit_var_prob, mdd_p05]):
        return 0.5

    # 把「損失程度」轉成分數（損失越大分越低）
    # 例如 VaR=-0.10 => 0.6~0.7；VaR=-0.25 => 很低
    var_s = clip01(1 - min(0.60, abs(var_ret)) / 0.30)     # 30%損失視為很糟
    cvar_s = clip01(1 - min(0.60, abs(cvar_ret)) / 0.35)   # CVaR容忍稍放寬
    mdd_s = clip01(1 - min(0.80, abs(mdd_p05)) / 0.40)     # 40% MDD 視為很糟
    hit_s = clip01(1 - min(0.90, hit_var_prob) / 0.30)     # 30%機率觸及VaR → 扣很兇

    return clip01(0.30*var_s + 0.30*cvar_s + 0.20*mdd_s + 0.20*hit_s)

def total_score(row):
    comps = {
        "model": score_model(row.get("model_score")),
        "volume": score_volume(row.get("volume_ratio")),
        "pattern": score_pattern(row.get("rsi")),
        "analyst": score_analyst(row.get("upside_to_target_mean_%"),
                                 row.get("recommendation_key"),
                                 row.get("num_analyst_opinions")),
        "risk": score_risk(row.get("var_ret"),
                           row.get("cvar_ret"),
                           row.get("hit_var_prob"),
                           row.get("mdd_p05")),
    }
    # 欠資料的 component 用 0.5
    for k,v in comps.items():
        if pd.isna(v):
            comps[k] = 0.5

    s = 0.0
    for k,w in W.items():
        s += w * comps[k]

    return float(round(s * 100, 2)), comps


# ============================================================
# 5) 交易動作 + 停損/停利（用 VaR/CVaR/樂觀目標/分析師高目標）
# ============================================================

def suggest_action(score_0_100, hit_var_prob, mdd_p05, analyst_upside):
    # 你可以依家族資金風格更保守
    if score_0_100 >= 75:
        return "高勝率加碼：分批買（回檔加）"
    if score_0_100 >= 60:
        return "可買：分批買/回檔買"
    if score_0_100 >= 50:
        return "觀望：等量能或型態確認"
    return "偏保守：不動作/減碼控風險"

def stops_and_targets(last_price, var_price, cvar_price, target_mean, target_high=None):
    """
    停損：用 VaR 價位再留 buffer（例如 0.98）
    停利：取 min(分析師高目標, 模型95%上緣) 的概念
    這裡先用 target_high / target_mean 做近似（你若也有 Monte Carlo optimistic_price 可再加）
    """
    if pd.isna(last_price):
        return np.nan, np.nan, np.nan, np.nan

    # 停損兩段：一般停損 / 極端停損（CVaR）
    if pd.isna(var_price):
        stop = last_price * 0.90
    else:
        stop = var_price * 0.98

    if pd.isna(cvar_price):
        hard_stop = last_price * 0.85
    else:
        hard_stop = cvar_price * 0.98

    # 停利：優先用 target_high（若有），沒有就用 target_mean
    tp = np.nan
    if target_high is not None and not pd.isna(target_high):
        tp = float(target_high)
    elif not pd.isna(target_mean):
        tp = float(target_mean)

    # 如果 tp 很接近現價，就至少設成 1.08x（避免甜度太低）
    if not pd.isna(tp) and tp < last_price * 1.05:
        tp = last_price * 1.08

    # 風險報酬比（以 stop 為風險）
    risk = max(1e-9, last_price - stop)
    reward = (tp - last_price) if not pd.isna(tp) else np.nan
    rr = (reward / risk) if (not pd.isna(reward)) else np.nan

    return float(stop), float(hard_stop), (float(tp) if not pd.isna(tp) else np.nan), (float(rr) if not pd.isna(rr) else np.nan)


# ============================================================
# main
# ============================================================

def main():
    # 1) load signals
    sig = load_all_signals(SIGNAL_TXTS)
    sig = sig[sig["signal"].str.upper() == "BUY"].copy()

    if sig.empty:
        print("❌ 沒抓到 BUY 信號，請確認 txt 內容是否含 'Signal is BUY'")
        return

    print(f"✅ BUY signals found: {len(sig)}")

    # 2) load analyst & merge
    ana = load_analyst(ANALYST_CSV)
    df = sig.merge(ana, on="ticker", how="left", suffixes=("", "_ana"))

    # 3) risk metrics per ticker (bootstrap, fast)
    risk_rows = []
    for t in df["ticker"].unique():
        try:
            r = risk_metrics_for_ticker(
                t,
                horizon=HORIZON_DAYS,
                sims=SIMS,
                conf=CONF,
                seed=SEED,
                lookback_years=LOOKBACK_YEARS
            )
            r["ticker"] = t
        except Exception as e:
            r = {"ticker": t}
            print(f"⚠️ risk calc failed for {t}: {e}")
        risk_rows.append(r)

    risk_df = pd.DataFrame(risk_rows)
    df = df.merge(risk_df, on="ticker", how="left")

    # 4) scoring
    scores = []
    comp_list = []
    for _, row in df.iterrows():
        s, comps = total_score(row)
        scores.append(s)
        comp_list.append(comps)

    df["win_score"] = scores
    df["score_model"] = [c["model"] for c in comp_list]
    df["score_volume"] = [c["volume"] for c in comp_list]
    df["score_pattern"] = [c["pattern"] for c in comp_list]
    df["score_analyst"] = [c["analyst"] for c in comp_list]
    df["score_risk"] = [c["risk"] for c in comp_list]

    # 5) action + stops/targets
    actions = []
    stops = []
    hard_stops = []
    tps = []
    rrs = []

    # analyst high target 可能欄名不同，做容錯
    target_high_col = None
    for c in ["target_high_price", "target_high", "target_high_price_usd"]:
        if c in df.columns:
            target_high_col = c
            break

    for _, row in df.iterrows():
        action = suggest_action(
            row["win_score"],
            row.get("hit_var_prob"),
            row.get("mdd_p05"),
            row.get("upside_to_target_mean_%"),
        )
        actions.append(action)

        stop, hard_stop, tp, rr = stops_and_targets(
            row.get("last_price"),
            row.get("var_price"),
            row.get("cvar_price"),
            row.get("target_mean_price"),
            row.get(target_high_col) if target_high_col else None,
        )
        stops.append(stop); hard_stops.append(hard_stop); tps.append(tp); rrs.append(rr)

    df["action"] = actions
    df["stop_loss"] = stops
    df["hard_stop"] = hard_stops
    df["take_profit"] = tps
    df["risk_reward"] = rrs

    # 6) sort & output
    df = df.sort_values(["win_score", "score_risk", "num_analyst_opinions"], ascending=[False, False, False])

    keep = [
        "ticker", "win_score",
        "model_score", "volume_ratio", "rsi",
        "current_price", "target_mean_price",
        "upside_to_target_mean_%", "recommendation_key", "num_analyst_opinions",
        "last_price", "var_price", "cvar_price", "var_ret", "cvar_ret",
        "hit_var_prob", "mdd_p05", "mdd_cvar95",
        "stop_loss", "hard_stop", "take_profit", "risk_reward",
        "action", "source_file"
    ]
    keep = [c for c in keep if c in df.columns]
    df_out = df[keep].copy()

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved: {OUTPUT_CSV}\n")

    # 預覽前 20 名
    print(df_out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()