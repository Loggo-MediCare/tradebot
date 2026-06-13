"""
PPO Backtest ROI Utility
Shared across all get_trading_signal_*.py scripts.

Simulates PPO model trading on historical data and returns:
  - ppo_roi:       cumulative return following model signals
  - buy_hold_roi:  cumulative return of pure buy-and-hold
"""

import numpy as np


def _prepare_df(df):
    """Normalize df: lowercase columns, compute missing indicators."""
    df = df.copy()
    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
    # Handle MultiIndex from yfinance
    if hasattr(df.columns, 'levels'):
        df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    if 'adj close' in df.columns and 'close' not in df.columns:
        df = df.rename(columns={'adj close': 'close'})
    c = df['close']
    if 'sma_10' not in df.columns:
        df['sma_10'] = c.rolling(10).mean()
    if 'sma_30' not in df.columns:
        df['sma_30'] = c.rolling(30).mean()
    if 'sma_50' not in df.columns:
        df['sma_50'] = c.rolling(50).mean()
    if 'macd' not in df.columns:
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    if 'rsi' not in df.columns:
        delta = c.diff()
        gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    if 'bb_upper' not in df.columns:
        mid = c.rolling(20).mean()
        std = c.rolling(20).std()
        df['bb_upper'] = mid + std * 2
        df['bb_lower'] = mid - std * 2
    return df.fillna(method='bfill').fillna(method='ffill')


def calculate_ppo_backtest_roi(model, df, initial_balance: float = 10_000) -> tuple[float | None, float | None]:
    """
    Walk the PPO model through df step-by-step, simulate trades, return ROI%.

    Accepts df with uppercase (yfinance) or lowercase columns — auto-normalizes.
    Also auto-computes sma_10/30/50, macd, rsi, bb_upper/lower if missing.

    Returns (ppo_roi_pct, buy_hold_roi_pct) or (None, None) on failure.
    """
    try:
        if model is None:
            return None, None
        warmup = 30          # skip first N rows (indicators not stable)
        if len(df) < warmup + 5:
            return None, None

        df = _prepare_df(df)
        df = df.reset_index(drop=True)

        def _f(row, col, default=0.0):
            v = row.get(col, default)
            try:
                return float(v) if v is not None else default
            except Exception:
                return default

        balance = initial_balance
        shares  = 0.0

        for i in range(warmup, len(df)):
            row   = df.iloc[i]
            price = _f(row, 'close')
            if price <= 0:
                continue

            total_value = balance + shares * price
            stock_ratio = (shares * price) / total_value if total_value > 0 else 0.0
            cash_ratio  = balance / total_value           if total_value > 0 else 1.0
            total_profit = total_value - initial_balance

            sma50 = _f(row, 'sma_50') or _f(row, 'sma_30')

            obs = np.array([
                shares, balance, price,
                _f(row, 'sma_10'), _f(row, 'sma_30'), sma50,
                _f(row, 'rsi', 50), _f(row, 'macd'), _f(row, 'macd_signal'),
                _f(row, 'bb_upper'), _f(row, 'bb_lower'), _f(row, 'volume'),
                total_profit, stock_ratio, cash_ratio,
            ], dtype=np.float32)

            action, _ = model.predict(obs, deterministic=True)
            act = float(action[0]) if isinstance(action, np.ndarray) else float(action)

            if act < -0.1:                               # sell
                sell_shares = int(shares * abs(act))
                if sell_shares > 0:
                    balance += sell_shares * price
                    shares  -= sell_shares
            elif act > 0.1:                              # buy
                max_buy    = int(balance // price)
                buy_shares = int(max_buy * act)
                if buy_shares > 0 and buy_shares * price <= balance:
                    balance -= buy_shares * price
                    shares  += buy_shares

        final_price = float(df.iloc[-1]['close'])
        final_value = balance + shares * final_price
        ppo_roi = (final_value - initial_balance) / initial_balance * 100

        # Buy-and-hold: buy as many shares as possible at warmup open
        start_price  = float(df.iloc[warmup]['close'])
        bh_shares    = initial_balance // start_price if start_price > 0 else 0
        bh_cash_left = initial_balance - bh_shares * start_price
        bh_value     = bh_shares * final_price + bh_cash_left
        buy_hold_roi = (bh_value - initial_balance) / initial_balance * 100

        return round(ppo_roi, 2), round(buy_hold_roi, 2)

    except Exception:
        return None, None


def print_ppo_action_line(action_value: float,
                          ppo_roi: float | None,
                          buy_hold_roi: float | None,
                          currency: str = '') -> None:
    """
    Print the PPO action line + backtest ROI.

    PPO  動作:+0.2803  🟢 看多 (已持有→續抱 | 未持有→可進場)
       ℹ️  PPO以P&L評估: 回測+37.73% vs 買入持有+36.30%
    """
    if action_value > 0.1:
        direction = "🟢 看多 (已持有→續抱 | 未持有→可進場)"
    elif action_value < -0.1:
        direction = "🔴 看空 (已持有→考慮減碼 | 未持有→觀望)"
    else:
        direction = "🟡 中性 (觀望，等待方向確認)"

    print(f"\nPPO  動作:{action_value:+.4f}  {direction}")

    if ppo_roi is not None and buy_hold_roi is not None:
        print(f"   ℹ️  PPO以P&L評估: 回測{ppo_roi:+.2f}% vs 買入持有{buy_hold_roi:+.2f}%")
    else:
        print(f"   ℹ️  PPO回測資料不足")
