import math
import numpy as np
import pandas as pd

# =========================
# 0) 常態分布 + BSM + Greeks
# =========================
def norm_cdf(x): return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
def norm_pdf(x): return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _d1_d2(S,K,r,sig,tau):
    if tau <= 0 or sig <= 0 or S <= 0 or K <= 0: return float("nan"), float("nan")
    d1 = (math.log(S/K) + (r + 0.5*sig*sig)*tau) / (sig*math.sqrt(tau))
    d2 = d1 - sig*math.sqrt(tau)
    return d1, d2

def bsm_call(S,K,r,sig,tau):
    if tau <= 0: return max(S-K, 0.0)
    d1,d2=_d1_d2(S,K,r,sig,tau)
    return S*norm_cdf(d1) - K*math.exp(-r*tau)*norm_cdf(d2)

def bsm_put(S,K,r,sig,tau):
    if tau <= 0: return max(K-S, 0.0)
    d1,d2=_d1_d2(S,K,r,sig,tau)
    return K*math.exp(-r*tau)*norm_cdf(-d2) - S*norm_cdf(-d1)

def delta_call(S,K,r,sig,tau):
    if tau <= 0: return 1.0 if S>K else 0.0
    d1,_=_d1_d2(S,K,r,sig,tau)
    return norm_cdf(d1)

def delta_put(S,K,r,sig,tau):
    if tau <= 0: return -1.0 if S<K else 0.0
    d1,_=_d1_d2(S,K,r,sig,tau)
    return norm_cdf(d1) - 1.0

def gamma(S,K,r,sig,tau):
    if tau <= 0: return 0.0
    d1,_=_d1_d2(S,K,r,sig,tau)
    return norm_pdf(d1)/(S*sig*math.sqrt(tau))

def vega(S,K,r,sig,tau):
    if tau <= 0: return 0.0
    d1,_=_d1_d2(S,K,r,sig,tau)
    return S*norm_pdf(d1)*math.sqrt(tau)

# =========================
# 1) 讀本地 2330 CSV（無 yfinance）
# =========================
def load_2330_csv(path="2330_TW.csv"):
    df = pd.read_csv(path)
    dc = "date" if "date" in df.columns else "Date" if "Date" in df.columns else None
    if not dc: raise ValueError("CSV 需要 date/Date 欄位")
    df[dc] = pd.to_datetime(df[dc])
    df = df.sort_values(dc).set_index(dc)

    if "close" not in df.columns:
        if "Close" in df.columns: df["close"]=df["Close"]
        else: raise ValueError("CSV 需要 close/Close")
    if "volume" not in df.columns:
        if "Volume" in df.columns: df["volume"]=df["Volume"]
        else: raise ValueError("CSV 需要 volume/Volume")

    df["ret"] = df["close"].pct_change()
    return df.dropna(subset=["ret"])

# =========================
# 2) 波動率：Hist / EWMA / GARCH(可用就用，否則 fallback)
# =========================
def vol_hist(r, window=60, td=252):
    return r.rolling(window).std()*math.sqrt(td)

def vol_ewma(r, lam=0.94, td=252):
    x = r.values
    var = np.zeros_like(x, dtype=float)
    var[0] = np.nanvar(x[:min(20,len(x))])
    for i in range(1,len(x)):
        var[i] = lam*var[i-1] + (1-lam)*(x[i-1]**2)
    return pd.Series(np.sqrt(var)*math.sqrt(td), index=r.index)

def vol_garch_or_fallback(r, td=252):
    try:
        from arch import arch_model
        rr = (r.dropna()*100.0)
        am = arch_model(rr, mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
        res = am.fit(disp="off")
        cond = (res.conditional_volatility/100.0)*math.sqrt(td)
        return cond.reindex(r.index).fillna(method="ffill").fillna(method="bfill")
    except Exception:
        return vol_ewma(r, lam=0.94, td=td)

# =========================
# 3) 成本：spread + impact（vol / volume / participation）
# =========================
def cost_model(price, qty_change, vol_ann, volume_shares, spread_bps=2.0, impact_k=0.10):
    if qty_change == 0: return 0.0
    spread = spread_bps * 1e-4
    adv = max(float(volume_shares), 1.0)
    part = abs(qty_change)/adv
    impact = impact_k*float(vol_ann)*math.sqrt(max(part, 0.0))
    return abs(qty_change)*price*(spread + impact)

# =========================
# 4) 策略：Protective Put / Collar + 事件觸發 roll
# =========================
def backtest_protection(
    df,
    mode="collar",            # "put" or "collar"
    shares=1000,              # 持有現股股數
    r=0.02,
    T_days=45,                # 每次買的期權到期天數（交易日）
    roll_days=10,             # 距離到期 <= roll_days 就 roll
    put_mny=0.95,             # Put strike = S0*put_mny
    call_mny=1.05,            # Call strike = S0*call_mny（collar 才用）
    vol_method="garch",       # "hist" | "ewma" | "garch"
    reb_delta_th=0.10,        # |Δ_total 變動| 超過門檻才調整「保險張數」（成本敏感）
    reb_gamma_th=1e-4,        # |Γ_total| 超過門檻（可選）
    spread_bps=2.0,
    impact_k=0.10,
    td=252
):
    df = df.copy()
    if vol_method=="hist":
        df["sigma"]=vol_hist(df["ret"], 60, td)
    elif vol_method=="ewma":
        df["sigma"]=vol_ewma(df["ret"], 0.94, td)
    else:
        df["sigma"]=vol_garch_or_fallback(df["ret"], td)

    df["sigma"]=df["sigma"].fillna(method="bfill").clip(0.05, 1.5)

    cash = 0.0
    put_qty = 0.0
    call_qty = 0.0  # collar: short call => qty negative
    put_K = None
    call_K = None
    tau_days_left = 0

    last_total_delta = None

    out = []

    for i,(dt, row) in enumerate(df.iterrows()):
        S = float(row["close"])
        vol = float(row["sigma"])
        volm = float(row["volume"])
        # 到期剩餘
        if tau_days_left > 0:
            tau_days_left -= 1

        # 當日 tau
        tau = max(tau_days_left/td, 0.0)

        # 期權當日價格
        put_px  = bsm_put(S, put_K, r, vol, tau) if (put_K is not None and tau>0) else (max(put_K-S,0.0) if put_K else 0.0)
        call_px = bsm_call(S, call_K, r, vol, tau) if (call_K is not None and tau>0) else (max(S-call_K,0.0) if call_K else 0.0)

        # Greeks
        put_d  = delta_put(S, put_K, r, vol, tau) if put_K else 0.0
        call_d = delta_call(S, call_K, r, vol, tau) if call_K else 0.0
        put_g  = gamma(S, put_K, r, vol, tau) if put_K else 0.0
        call_g = gamma(S, call_K, r, vol, tau) if call_K else 0.0
        put_v  = vega(S, put_K, r, vol, tau) if put_K else 0.0
        call_v = vega(S, call_K, r, vol, tau) if call_K else 0.0

        # 組合市值
        stock_val = shares*S
        opt_val = put_qty*put_px + call_qty*call_px
        port_val = cash + stock_val + opt_val

        # 組合總風險（對「標的價格」的敏感度）
        total_delta = shares*1.0 + put_qty*put_d + call_qty*call_d
        total_gamma = put_qty*put_g + call_qty*call_g
        total_vega  = put_qty*put_v + call_qty*call_v

        # ---- 事件觸發：是否 roll / 是否調整避險張數 ----
        need_roll = (tau_days_left <= roll_days) or (put_K is None)

        # (可選) delta/gamma 事件觸發：讓你「不是固定頻率」才調倉
        need_reb = False
        if last_total_delta is not None:
            if abs(total_delta - last_total_delta) >= reb_delta_th*shares:
                need_reb = True
        if abs(total_gamma) >= reb_gamma_th:
            need_reb = True

        # Roll：建立新的 put / call
        if need_roll:
            # 先把舊期權平倉（用成本模型）
            if put_K is not None:
                tc = cost_model(put_px, -put_qty, vol, volm, spread_bps, impact_k)
                cash += (-put_qty)*put_px - tc
                put_qty = 0.0
            if mode=="collar" and call_K is not None:
                tc = cost_model(call_px, -call_qty, vol, volm, spread_bps, impact_k)
                cash += (-call_qty)*call_px - tc
                call_qty = 0.0

            # 建新一期（用當日 S 設 strike）
            put_K = S*put_mny
            tau_days_left = T_days

            # 基本：protective put 的 put 張數可先設「shares 的 1:1 保險」
            # 台股一口期權的合約乘數要你自己改（這裡用 1 張=1股 的抽象化）
            target_put_qty = -shares/put_d if put_d != 0 else shares  # 讓總 delta 接近 0（較激進）
            # 但保護性買 put 通常不把 delta 拉到 0，常見是買「名目 1:1」：put_qty = shares
            # 你可以二選一：名目保險 vs delta-neutral
            put_qty = shares  # ★保守、直觀：1:1 保險（建議你先用這個）

            # 買 put
            put_px_new = bsm_put(S, put_K, r, vol, tau_days_left/td)
            tc = cost_model(put_px_new, put_qty, vol, volm, spread_bps, impact_k)
            cash += (-put_qty)*put_px_new - tc  # 買入 => 現金流出（qty 正表示 long）

            if mode=="collar":
                call_K = S*call_mny
                # 賣 call 的張數通常設成 cover：1:1（用來補貼 put）
                call_qty = -shares
                call_px_new = bsm_call(S, call_K, r, vol, tau_days_left/td)
                tc = cost_model(call_px_new, call_qty, vol, volm, spread_bps, impact_k)
                cash += (-call_qty)*call_px_new - tc  # 賣出 => 現金流入（call_qty 為負）

        # 非 roll 的情況下，若 need_reb，你可以「動態調整 put/call 張數」節省成本
        # 這裡給你最小版本：只調 put_qty（collar 同理可調 call_qty）
        elif need_reb:
            # 範例：把 put_qty 微調到讓 total_delta 落在 shares*(1±0.2) 之內
            # 目標 total_delta_ratio = 0.8 ~ 1.0 之間（保留多頭曝險，但降低跳水風險）
            desired = shares*0.9
            # 用 put delta 改 delta（put_d 是負）
            if put_d != 0:
                new_put_qty = put_qty + (desired - total_delta)/put_d
                d_put = new_put_qty - put_qty
                # 交易成本與現金流
                put_px = bsm_put(S, put_K, r, vol, tau)
                tc = cost_model(put_px, d_put, vol, volm, spread_bps, impact_k)
                cash += (-d_put)*put_px - tc
                put_qty = new_put_qty

        last_total_delta = total_delta

        # 重新計算當日（調倉後）
        tau = max(tau_days_left/td, 0.0)
        put_px  = bsm_put(S, put_K, r, vol, tau) if put_K else 0.0
        call_px = bsm_call(S, call_K, r, vol, tau) if (mode=="collar" and call_K) else 0.0

        stock_val = shares*S
        opt_val = put_qty*put_px + call_qty*call_px
        port_val = cash + stock_val + opt_val

        out.append({
            "date": dt,
            "S": S, "sigma": vol,
            "putK": put_K, "callK": call_K,
            "tau_days": tau_days_left,
            "cash": cash,
            "stock_val": stock_val,
            "opt_val": opt_val,
            "port_val": port_val,
            "put_qty": put_qty, "call_qty": call_qty
        })

    res = pd.DataFrame(out).set_index("date")
    res["port_ret"] = res["port_val"].pct_change()
    return res

# =========================
# 5) 風控 KPI：VaR / CVaR / MDD
# =========================
def var_cvar(x, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    v = np.quantile(x, alpha)
    cv = x[x <= v].mean() if np.any(x <= v) else v
    return v, cv

def max_drawdown(series):
    peak = series.cummax()
    dd = series/peak - 1.0
    return dd.min()

if __name__ == "__main__":
    df = load_2330_csv("2330_TW.csv")

    # 1) Protective Put
    put_res = backtest_protection(
        df, mode="put",
        shares=1000,
        T_days=45, roll_days=10,
        put_mny=0.95,
        vol_method="garch",
        reb_delta_th=0.10,
        spread_bps=2.0, impact_k=0.10
    )

    # 2) Collar
    col_res = backtest_protection(
        df, mode="collar",
        shares=1000,
        T_days=45, roll_days=10,
        put_mny=0.95, call_mny=1.05,
        vol_method="garch",
        reb_delta_th=0.10,
        spread_bps=2.0, impact_k=0.10
    )

    for name, res in [("ProtectivePut", put_res), ("Collar", col_res)]:
        r = res["port_ret"].dropna()
        v, cv = var_cvar(r.values, 0.05)
        mdd = max_drawdown(res["port_val"].dropna())
        print(f"\n=== {name} KPI ===")
        print(f"期間終值: {res['port_val'].iloc[-1]:,.2f}")
        print(f"日報酬均值: {r.mean():.6f}, 日報酬波動: {r.std(ddof=1):.6f}")
        print(f"VaR(5%) : {v:.6f}")
        print(f"CVaR(5%): {cv:.6f}")
        print(f"Max Drawdown: {mdd:.2%}")
