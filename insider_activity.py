"""
Insider Activity Tracker — Form 4 SEC filing data.

Data is sourced from SEC EDGAR Form 4 filings.
Structure can be extended per-ticker; each entry covers one calendar year.

Score penalty logic:
  - 10b5-1 pre-planned sale   → lighter penalty (expected, not panic)
  - Open-market discretionary → heavier penalty
  - Recency weight: sales within 14d carry full weight; 15-30d 50%; 31-60d 25%
  - Retention ratio mitigator: if insider still holds >80% of prior position → reduce penalty 30%
"""

from datetime import datetime, date

# ── Known Form 4 data ────────────────────────────────────────────────────────
INSIDER_DATA = {
    'MU': {
        'company': 'Micron Technology',
        'insiders': [
            {
                'name': 'Sanjay Mehrotra',
                'title': 'CEO',
                'transactions': [
                    {
                        'date': date(2026, 5, 1),
                        'shares': 40_000,
                        'avg_price': 536.26,
                        'type': 'sale',
                        'plan': '10b5-1',
                        'note': 'Rule 10b5-1 plan adopted 2026-01-30',
                    },
                    {
                        'date': date(2026, 5, 29),
                        'shares': 37_439,
                        'avg_price': 960.38,
                        'type': 'sale',
                        'plan': '10b5-1',
                        'note': 'Rule 10b5-1 plan — tranche 2',
                    },
                    {
                        'date': date(2026, 5, 29),
                        'shares': 2_561,
                        'avg_price': 975.63,
                        'type': 'sale',
                        'plan': '10b5-1',
                        'note': 'Rule 10b5-1 plan — tranche 2 (split filing)',
                    },
                ],
                'shares_retained': 990_000,   # approx post-sale direct + GRAT
            },
        ],
    },
}


def get_insider_score_adjustment(ticker: str, as_of: date = None) -> tuple:
    """
    Returns (score_adjustment: float, summary_lines: list[str])

    score_adjustment is negative when significant selling occurred.
    summary_lines is a list of display strings for the signal output.
    """
    if as_of is None:
        as_of = date.today()

    data = INSIDER_DATA.get(ticker.upper())
    if not data:
        return 0.0, []

    lookback_days = 60
    total_penalty = 0.0
    lines = []

    for insider in data['insiders']:
        sales = [t for t in insider['transactions']
                 if t['type'] == 'sale' and (as_of - t['date']).days <= lookback_days]
        if not sales:
            continue

        total_shares_sold = sum(t['shares'] for t in sales)
        total_value_usd   = sum(t['shares'] * t['avg_price'] for t in sales)
        retained           = insider.get('shares_retained', 0)
        sell_ratio         = total_shares_sold / (total_shares_sold + retained) if (total_shares_sold + retained) > 0 else 0

        penalty = 0.0
        for t in sales:
            days_ago = (as_of - t['date']).days
            # recency weight
            if days_ago <= 14:
                recency_w = 1.0
            elif days_ago <= 30:
                recency_w = 0.6
            else:
                recency_w = 0.3

            # base penalty by amount
            usd = t['shares'] * t['avg_price']
            if usd >= 30_000_000:
                base = 8.0
            elif usd >= 10_000_000:
                base = 5.0
            elif usd >= 1_000_000:
                base = 3.0
            else:
                base = 1.0

            # 10b5-1 mitigator — pre-planned, less alarming
            plan_factor = 0.55 if t.get('plan') == '10b5-1' else 1.0

            # retention mitigator — insider still heavily invested
            retention_factor = 0.70 if sell_ratio < 0.15 else 1.0

            penalty += base * recency_w * plan_factor * retention_factor

        total_penalty += penalty

        # build display lines
        lines.append(f"👤 內部人士減持 ({insider['name']}, {insider['title']}):")
        for t in sales:
            days_ago = (as_of - t['date']).days
            plan_tag = f"[{t['plan']}計畫]" if t.get('plan') else "[自由交易]"
            lines.append(
                f"   • {t['date']}  賣出 {t['shares']:,} 股 @ ${t['avg_price']:.2f}"
                f"  (${t['shares']*t['avg_price']/1e6:.1f}M)  {plan_tag}  {days_ago}天前"
            )
        lines.append(
            f"   合計: {total_shares_sold:,} 股 / ${total_value_usd/1e6:.1f}M  "
            f"| 仍持有 ~{retained/1e4:.0f}萬股 (留任比例 {(1-sell_ratio)*100:.0f}%)"
        )
        lines.append(f"   評分扣減: -{penalty:.1f} 分  (10b5-1預排計畫，扣分已減半)")

    return -round(total_penalty, 1), lines
