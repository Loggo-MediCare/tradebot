"""
Shared market checks for trading signal scripts.
Includes:
1) OBV calculation
2) Money-flow strength check
3) Fundamental profitability check used in oversold sell-protection branch
4) US intraday market detection
5) Intraday volume projection (time-adjusted volume ratio)
"""

from datetime import datetime, time
from zoneinfo import ZoneInfo
import pandas as pd

# NYSE total trading minutes: 9:30 AM – 4:00 PM = 390 min
_TOTAL_TRADING_MINUTES = 390


def calculate_obv(df):
    """Calculate OBV and OBV MA20."""
    obv = [0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_ma20"] = pd.Series(obv).rolling(20).mean()
    return df


def is_us_intraday(latest_date):
    """Return True when the latest row corresponds to today during NYSE trading hours."""
    try:
        if isinstance(latest_date, str):
            latest_date = pd.to_datetime(latest_date).date()
        elif hasattr(latest_date, 'date'):
            latest_date = latest_date.date()

        now_et = datetime.now(ZoneInfo('America/New_York'))
        return (
            latest_date == now_et.date()
            and now_et.weekday() < 5
            and time(9, 30) <= now_et.time() < time(16, 0)
        )
    except Exception:
        return False


def get_intraday_adjusted_volume(df):
    """
    Project partial intraday volume to an estimated full-day volume.

    When called during NYSE hours on the current trading day, the last row's
    volume only covers time elapsed since 9:30 AM ET. Dividing that raw number
    by a 20-day full-day average produces a falsely low volume ratio (e.g. 0.38x
    at 10:42 AM even when the pace is genuinely high). This function multiplies
    the raw volume by (390 / elapsed_minutes) so downstream ratio calculations
    compare apples-to-apples.

    Returns the raw volume unchanged for completed trading days or when the
    elapsed time is too short to give a reliable projection.
    """
    current_volume = df["volume"].iloc[-1]
    try:
        latest = df.index[-1]
        if is_us_intraday(latest):
            now_et = datetime.now(ZoneInfo("America/New_York"))
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            elapsed = (now_et - market_open).total_seconds() / 60
            # Require at least 5 minutes to avoid extreme multipliers at open
            if elapsed >= 5:
                return current_volume * (_TOTAL_TRADING_MINUTES / elapsed)
    except Exception:
        pass
    return current_volume


def money_flow_strength(df):
    """Advanced money-flow strength: up/down volume + money flow + OBV."""
    if len(df) < 20:
        return False, 1.0, {}

    adjusted_volume = get_intraday_adjusted_volume(df)
    avg_volume_20 = df["volume"].rolling(20).mean().iloc[-1]
    volume_ratio = adjusted_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

    recent = df.tail(30).copy()
    recent["up_volume"] = recent.apply(
        lambda x: x["volume"] if x["close"] > x["open"] else 0, axis=1
    )
    recent["down_volume"] = recent.apply(
        lambda x: x["volume"] if x["close"] <= x["open"] else 0, axis=1
    )
    up_vol = recent["up_volume"].sum()
    down_vol = recent["down_volume"].sum()
    up_down_ratio = up_vol / (down_vol + 1e-10)

    recent["money_flow"] = recent["volume"] * (recent["close"] - recent["open"])
    net_money_flow_30d = recent["money_flow"].sum()
    net_money_flow_5d = recent["money_flow"].tail(5).sum()

    latest = df.iloc[-1]
    capital_inflow = (volume_ratio > 1.5) and (latest["close"] > latest["open"])

    obv_now = df["obv"].iloc[-1] if "obv" in df.columns else 0
    obv_ma = df["obv_ma20"].iloc[-1] if "obv_ma20" in df.columns else 0
    obv_bullish = obv_now > obv_ma

    strong_money = (
        (capital_inflow or (up_down_ratio > 1.3 and volume_ratio > 1.0))
        and obv_bullish
    )

    details = {
        "up_volume_30d": int(up_vol),
        "down_volume_30d": int(down_vol),
        "up_down_ratio": round(up_down_ratio, 2),
        "net_money_flow_30d": net_money_flow_30d,
        "net_money_flow_5d": net_money_flow_5d,
        "capital_inflow": capital_inflow,
        "obv_bullish": obv_bullish,
    }

    return strong_money, volume_ratio, details


def calculate_growth_score_adjustment(yf_module, ticker_symbol):
    """
    Return a buy-score boost (and sell-score reduction) for high-growth stocks.

    Thresholds (user-defined):
      Revenue Growth > 33%  → +8 pts
      EPS Growth    > 100% → +8 pts

    When these conditions are true the model is penalising strong companies with
    false SELL signals driven purely by technicals. Adding these points counteracts
    that bias.

    Returns:
        dict: {
            "adjustment":     int (0–16),
            "reasons":        list[str],
            "revenue_growth": float | None  (percentage, e.g. 45.2),
            "eps_growth":     float | None  (percentage, e.g. 130.0),
        }
    """
    result = {
        "adjustment": 0,
        "reasons": [],
        "revenue_growth": None,
        "eps_growth": None,
    }
    try:
        info = yf_module.Ticker(ticker_symbol).info
        raw_rev = info.get("revenueGrowth")
        raw_eps = info.get("earningsGrowth")

        if raw_rev is not None:
            rev_pct = float(raw_rev) * 100
            result["revenue_growth"] = round(rev_pct, 1)
            if raw_rev > 0.33:
                result["adjustment"] += 8
                result["reasons"].append(
                    f"收入成長 {rev_pct:.1f}% > 33% (基本面強勢 +8分)"
                )

        if raw_eps is not None:
            eps_pct = float(raw_eps) * 100
            result["eps_growth"] = round(eps_pct, 1)
            if raw_eps > 1.0:
                result["adjustment"] += 8
                result["reasons"].append(
                    f"EPS成長 {eps_pct:.1f}% > 100% (獲利爆發 +8分)"
                )
    except Exception:
        pass
    return result


def evaluate_fundamentals_for_sell(yf_module, ticker_symbol):
    """
    Evaluate profitability flags for oversold sell-protection branch.

    Returns:
        dict: {
            "good": bool,
            "bad": bool,
            "profit_margin": float|None,
            "net_income": float|None
        }
    """
    result = {
        "good": False,
        "bad": False,
        "profit_margin": None,
        "net_income": None,
    }

    try:
        info = yf_module.Ticker(ticker_symbol).info
        net_income = info.get("netIncome", None)
        profit_margin = info.get("profitMargins", None)
        result["net_income"] = net_income
        result["profit_margin"] = profit_margin

        # Prefer profit margin when available.
        if profit_margin is not None and profit_margin > 0.10:
            result["good"] = True
        elif profit_margin is not None and profit_margin < 0:
            result["bad"] = True
        elif net_income is not None and net_income > 0:
            result["good"] = True
        elif net_income is not None and net_income < 0:
            result["bad"] = True
    except Exception:
        pass

    return result

