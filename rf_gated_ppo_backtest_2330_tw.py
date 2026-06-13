"""rf_gated_ppo_backtest_2330_tw.py

Example: Integration #3 (RF gating / action masking) with a single ticker (default: 2330.TW).

Integration #3 definition (as you described):
  - Train a RandomForest (RF) to estimate market regime.
  - Use RF output to *gate* the PPO actions:
      Bullish (P(up) >= bull_thr): disallow SELL (negative actions)
      Bearish (P(up) <= bear_thr): disallow BUY  (positive actions)
      Neutral: force HOLD (action = 0)
  - Train PPO on the train split, then backtest on the test split.

Important fixes vs the first draft:
  - Reward is based on portfolio value change from today -> next day (so PPO can learn).
  - Avoid backfilling (bfill) to reduce look-ahead leakage.
  - Add CLI args so you can run fast demos (small timesteps) or longer training.

Run example (fast):
  python rf_gated_ppo_backtest_2330_tw.py --ticker 2330.TW --period 3y --timesteps 5000 --n_steps 256 --rf_estimators 150 --bull_thr 0.52 --bear_thr 0.48

This is research code, not financial advice.
"""

import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces

import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


# --------------------------- indicators ---------------------------------

def rsi_wilder(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add indicators (no backfill to avoid look-ahead)."""
    out = df.copy()

    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_30"] = out["close"].rolling(30).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["ma_200"] = out["close"].rolling(200).mean()

    out["rsi"] = rsi_wilder(out["close"], 14)

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    m = out["close"].rolling(20).mean()
    s = out["close"].rolling(20).std()
    out["bb_upper"] = m + 2 * s
    out["bb_lower"] = m - 2 * s
    out["bb_pos"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # ATR (14)
    prev_close = out["close"].shift(1)
    tr = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()),
    )
    out["atr"] = pd.Series(tr, index=out.index).rolling(14).mean()

    # volatility (20d)
    out["ret_1d"] = out["close"].pct_change(1)
    out["volatility_20d"] = out["ret_1d"].rolling(20).std()

    # OBV
    delta = out["close"].diff()
    direction = np.where(delta > 0, 1, np.where(delta < 0, -1, 0))
    out["obv"] = (direction * out["volume"].fillna(0)).cumsum()
    out["obv_ma_20"] = out["obv"].rolling(20).mean()

    # KD (stochastic)
    low_min = out["low"].rolling(9).min()
    high_max = out["high"].rolling(9).max()
    rsv = (out["close"] - low_min) / (high_max - low_min + 1e-9) * 100
    out["k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    out["d"] = out["k"].ewm(alpha=1 / 3, adjust=False).mean()

    return out


# --------------------------- RF gating ---------------------------------

@dataclass
class GateConfig:
    proba_up_bull: float = 0.55
    proba_up_bear: float = 0.45


def gate_action(action: float, proba_up: float, gate_cfg: GateConfig) -> float:
    """Integration #3: gate the PPO action using RF regime."""
    if proba_up >= gate_cfg.proba_up_bull:
        return max(0.0, action)  # bullish -> no sell
    if proba_up <= gate_cfg.proba_up_bear:
        return min(0.0, action)  # bearish -> no buy
    return 0.0  # neutral -> hold


# --------------------------- Trading Env --------------------------------

class RFGatedTradingEnv(gym.Env):
    """Continuous-action trading env with RF gating.

    Action: scalar in [-1, 1]
      > 0: buy with fraction of cash
      < 0: sell fraction of shares

    Reward: portfolio value change from t -> t+1 (scaled by init_cash).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        rf_model: RandomForestClassifier,
        rf_scaler: StandardScaler,
        rf_features: list[str],
        gate_cfg: GateConfig,
        init_cash: float = 10000.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.rf_model = rf_model
        self.rf_scaler = rf_scaler
        self.rf_features = rf_features
        self.gate_cfg = gate_cfg
        self.init_cash = float(init_cash)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.balance = self.init_cash
        self.shares = 0
        self.trades = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._obs(), {}

    def _price(self) -> float:
        return float(self.df.loc[self.current_step, "close"])

    def _rf_proba_up(self) -> float:
        row = self.df.loc[self.current_step, self.rf_features].astype(float)
        x = self.rf_scaler.transform([row.values])
        proba = self.rf_model.predict_proba(x)[0]
        classes = list(self.rf_model.classes_)
        if 1 in classes:
            return float(proba[classes.index(1)])
        return float(proba[-1])

    def _obs(self):
        row = self.df.iloc[self.current_step]
        price = float(row.get("close", 0))
        tv = self.balance + self.shares * price
        return np.array(
            [
                float(self.shares),
                float(self.balance),
                price,
                float(row.get("sma_10", 0)),
                float(row.get("sma_30", 0)),
                float(row.get("sma_50", 0)),
                float(row.get("rsi", 50)),
                float(row.get("macd", 0)),
                float(row.get("macd_signal", 0)),
                float(row.get("bb_upper", 0)),
                float(row.get("bb_lower", 0)),
                float(row.get("volume", 0)),
                float(tv - self.init_cash),
                (self.shares * price) / tv if tv > 0 else 0.0,
                self.balance / tv if tv > 0 else 1.0,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        a = float(action[0]) if hasattr(action, "__len__") else float(action)
        a = max(-1.0, min(1.0, a))

        price = self._price()
        tv_before = self.balance + self.shares * price

        # RF gate
        proba_up = self._rf_proba_up()
        a = gate_action(a, proba_up, self.gate_cfg)

        # execute at today's close
        if a < -0.1 and self.shares > 0:
            sell = int(self.shares * abs(a))
            if sell > 0:
                self.balance += sell * price
                self.shares -= sell
                self.trades += 1
        elif a > 0.1:
            buy = int((self.balance * a) // price)
            if buy > 0:
                self.balance -= buy * price
                self.shares += buy
                self.trades += 1

        # advance to next day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = float(self.df.loc[self.current_step, "close"])

        tv_after = self.balance + self.shares * next_price
        if done:
            # liquidate at last price
            self.balance += self.shares * next_price
            self.shares = 0
            tv_after = self.balance

        reward = (tv_after - tv_before) / self.init_cash

        return self._obs(), float(reward), bool(done), False, {
            "proba_up": float(proba_up),
            "gated_action": float(a),
            "tv_before": float(tv_before),
            "tv_after": float(tv_after),
        }


# --------------------------- Backtest helpers ----------------------------


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def sharpe_ratio(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mu = daily_returns.mean()
    sd = daily_returns.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(252) * mu / sd)


def run_backtest(env: RFGatedTradingEnv, model: PPO) -> dict:
    obs, _ = env.reset()
    done = False

    # equity measured at each step after price update
    equity = [env.init_cash]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

        price = float(env.df.loc[env.current_step, "close"])
        total_value = env.balance + env.shares * price
        equity.append(total_value)

    equity = np.array(equity, dtype=float)
    rets = np.diff(equity) / (equity[:-1] + 1e-9)

    roi = (equity[-1] / equity[0] - 1.0) * 100
    mdd = max_drawdown(equity) * 100
    shrp = sharpe_ratio(rets)

    return {
        "final_equity": float(equity[-1]),
        "roi_pct": float(roi),
        "max_drawdown_pct": float(mdd),
        "sharpe": float(shrp),
        "trades": int(env.trades),
    }


# --------------------------- Main ----------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Integration #3: RF-gated PPO backtest")
    parser.add_argument("--ticker", default="2330.TW")
    parser.add_argument("--period", default="3y", help="yfinance period, e.g. 2y/3y/5y")
    parser.add_argument("--horizon", type=int, default=20, help="RF label horizon (days)")
    parser.add_argument("--rf_estimators", type=int, default=200)
    parser.add_argument("--bull_thr", type=float, default=0.55, help="gate: bullish if P(up) >= this")
    parser.add_argument("--bear_thr", type=float, default=0.45, help="gate: bearish if P(up) <= this")
    parser.add_argument("--timesteps", type=int, default=8000, help="PPO training timesteps")
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ticker = args.ticker
    horizon_days = int(args.horizon)

    print(f"Downloading {ticker} (period={args.period}) ...")
    raw = yf.download(ticker, period=args.period, interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit("No data downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})

    df = raw.reset_index().copy()
    df = add_indicators(df)

    # RF label: direction of forward return
    df["fwd_ret"] = df["close"].shift(-horizon_days) / df["close"] - 1.0
    df["y_up"] = (df["fwd_ret"] > 0).astype(int)

    rf_features = [
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_pos",
        "sma_10",
        "sma_30",
        "sma_50",
        "ma_200",
        "atr",
        "volatility_20d",
        "obv",
        "obv_ma_20",
        "k",
        "d",
        "ret_1d",
    ]

    df_ml = df.dropna(subset=rf_features + ["y_up", "close", "volume", "Date"]).copy()
    if len(df_ml) < 350:
        raise SystemExit(f"Not enough clean rows after indicators/labels: {len(df_ml)}")

    split = int(len(df_ml) * 0.7)
    train_df = df_ml.iloc[:split].reset_index(drop=True)
    test_df = df_ml.iloc[split:].reset_index(drop=True)

    print(f"Rows: train={len(train_df)}  test={len(test_df)}")

    gate_cfg = GateConfig(proba_up_bull=float(args.bull_thr), proba_up_bear=float(args.bear_thr))

    # Fit RF on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[rf_features].astype(float).values)
    y_train = train_df["y_up"].astype(int).values

    rf = RandomForestClassifier(
        n_estimators=int(args.rf_estimators),
        random_state=int(args.seed),
        class_weight="balanced",
        min_samples_leaf=5,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # RF sanity check on test
    X_test = scaler.transform(test_df[rf_features].astype(float).values)
    y_test = test_df["y_up"].astype(int).values
    rf_pred = rf.predict(X_test)
    rf_acc = float((rf_pred == y_test).mean())

    proba = rf.predict_proba(X_test)
    cls = list(rf.classes_)
    proba_up = proba[:, cls.index(1)] if 1 in cls else proba[:, -1]

    bull_pct = float((proba_up >= gate_cfg.proba_up_bull).mean() * 100)
    bear_pct = float((proba_up <= gate_cfg.proba_up_bear).mean() * 100)
    neu_pct = 100.0 - bull_pct - bear_pct

    print(f"RF test accuracy (direction, horizon={horizon_days}d): {rf_acc:.3f}")
    print(f"Gate distribution on test: bull={bull_pct:.1f}% bear={bear_pct:.1f}% neutral={neu_pct:.1f}%")

    # PPO train env
    train_env = DummyVecEnv([
        lambda: RFGatedTradingEnv(
            train_df,
            rf_model=rf,
            rf_scaler=scaler,
            rf_features=rf_features,
            gate_cfg=gate_cfg,
            init_cash=10000.0,
        )
    ])

    timesteps = int(args.timesteps)
    print(f"Training PPO (RF-gated) for {timesteps} timesteps ...")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=int(args.n_steps),
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        verbose=0,
        seed=int(args.seed),
    )
    model.learn(total_timesteps=timesteps)

    # Backtest on test split
    test_env = RFGatedTradingEnv(
        test_df,
        rf_model=rf,
        rf_scaler=scaler,
        rf_features=rf_features,
        gate_cfg=gate_cfg,
        init_cash=10000.0,
    )

    bt = run_backtest(test_env, model)

    # Buy & hold baseline on test period
    bh_roi = (float(test_df.loc[len(test_df) - 1, "close"]) / float(test_df.loc[0, "close"]) - 1.0) * 100

    print("\n=== Backtest — Integration #3 RF-gated PPO ===")
    print(f"Ticker: {ticker}")
    print(f"Test period: {test_df.loc[0, 'Date'].date()} -> {test_df.loc[len(test_df)-1, 'Date'].date()}")
    print(f"ROI: {bt['roi_pct']:.2f}%")
    print(f"Max Drawdown: {bt['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (approx): {bt['sharpe']:.2f}")
    print(f"Trades: {bt['trades']}")
    print(f"Buy&Hold ROI (test): {bh_roi:.2f}%")

    # Save PPO model
    out_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out_dir, exist_ok=True)
    safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").lower()
    out_path = os.path.join(out_dir, f"ppo_rf_gated_{safe_ticker}")
    model.save(out_path)
    print(f"Model saved: {out_path}.zip")


if __name__ == "__main__":
    main()
