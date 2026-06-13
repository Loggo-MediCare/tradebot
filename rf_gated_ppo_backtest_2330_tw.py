"""rf_gated_ppo_backtest_2330_tw.py

Example: Integration #3 (RF gating / action masking) with a single ticker: 2330.TW

Idea:
  1) Train a RandomForestClassifier on historical technical indicators to estimate
     P(up) over a forward horizon.
  2) During PPO trading, use the RF output as a *gate*:
       - Bullish regime  (P(up) >= 0.55): disallow SELL (negative actions)
       - Bearish regime  (P(up) <= 0.45): disallow BUY  (positive actions)
       - Neutral regime  (0.45 < P(up) < 0.55): force HOLD (action = 0)
  3) Train PPO on the train period and backtest on a held-out test period.

Notes:
  - This is a research/demo script, not financial advice.
  - It avoids look-ahead by fitting RF on the train split only, then applying
    the fixed RF gate on the test split.

Run:
  python rf_gated_ppo_backtest_2330_tw.py
"""

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("MPLBACKEND", "Agg")

warnings.filterwarnings("ignore")

from roi_control import print_roi
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
    # Required columns: close, open, high, low, volume
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

    return out.bfill().ffill()


# --------------------------- RF gating ---------------------------------

@dataclass
class GateConfig:
    proba_up_bull: float = 0.55
    proba_up_bear: float = 0.45


def gate_action(action: float, proba_up: float, gate_cfg: GateConfig) -> float:
    """Integration #3: gate the PPO action using RF regime."""
    if proba_up >= gate_cfg.proba_up_bull:
        # bullish -> no sell
        return max(0.0, action)
    if proba_up <= gate_cfg.proba_up_bear:
        # bearish -> no buy
        return min(0.0, action)
    # neutral -> hold
    return 0.0


# --------------------------- Trading Env --------------------------------

class RFGatedTradingEnv(gym.Env):
    """Continuous-action env with RF gating.

    Action: scalar in [-1, 1]
      > 0: buy with fraction of cash
      < 0: sell fraction of shares
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
        # Keep same 15 dims as your PPO template to make it familiar.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.balance = self.init_cash
        self.shares = 0
        self.trades = 0
        self.last_total_value = self.init_cash

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
        # fallback
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
        proba_up = self._rf_proba_up()
        a = gate_action(a, proba_up, self.gate_cfg)

        # execute
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

        # advance
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # reward: 1-step change in total value (scaled)
        total_value = self.balance + self.shares * price
        reward = (total_value - self.last_total_value) / self.init_cash
        self.last_total_value = total_value

        if done:
            # liquidate at last price
            last_price = self._price()
            self.balance += self.shares * last_price
            self.shares = 0

        return self._obs(), float(reward), bool(done), False, {"proba_up": float(proba_up), "gated_action": float(a)}


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
    equity = [env.init_cash]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
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
        "equity_curve": equity,
    }


# --------------------------- Main ----------------------------------------


def main():
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Integration #3: RF-gated PPO backtest")
    parser.add_argument("--ticker", default="2330.TW")
    parser.add_argument("--period", default="3y", help="yfinance period, e.g. 2y/3y/5y")
    parser.add_argument("--horizon", type=int, default=20, help="RF label horizon (days)")
    parser.add_argument("--rf_estimators", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=8000, help="PPO training timesteps")
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ticker = args.ticker
    horizon_days = int(args.horizon)  # label horizon for RF

    print(f"Downloading {ticker} (period={args.period}) ...")
    raw = yf.download(ticker, period=args.period, interval="1d", progress=False, auto_adjust=True)
    if raw.empty:
        raise SystemExit("No data downloaded.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    raw = raw.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = raw.reset_index().copy()
    df = add_indicators(df)

    # RF label: direction of forward return
    df["fwd_ret"] = df["close"].shift(-horizon_days) / df["close"] - 1.0
    df["y_up"] = (df["fwd_ret"] > 0).astype(int)

    # RF features: keep it explicit and stable
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

    df_ml = df.dropna(subset=rf_features + ["y_up", "close", "volume"]).copy()
    if len(df_ml) < 400:
        raise SystemExit(f"Not enough clean rows after indicators/labels: {len(df_ml)}")

    # time split
    split = int(len(df_ml) * 0.7)
    train_df = df_ml.iloc[:split].reset_index(drop=True)
    test_df = df_ml.iloc[split:].reset_index(drop=True)

    print(f"Rows: train={len(train_df)}  test={len(test_df)}")

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

    # Simple RF sanity check on test (classification accuracy)
    X_test = scaler.transform(test_df[rf_features].astype(float).values)
    y_test = test_df["y_up"].astype(int).values
    rf_acc = float((rf.predict(X_test) == y_test).mean())
    print(f"RF test accuracy (direction, horizon={horizon_days}d): {rf_acc:.3f}")

    gate_cfg = GateConfig(proba_up_bull=0.55, proba_up_bear=0.45)

    # PPO train env (RF gating active during training)
    train_env = DummyVecEnv(
        [
            lambda: RFGatedTradingEnv(
                train_df,
                rf_model=rf,
                rf_scaler=scaler,
                rf_features=rf_features,
                gate_cfg=gate_cfg,
                init_cash=10000.0,
            )
        ]
    )

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

    # Buy-and-hold baseline on test period
    bh_roi = (test_df.loc[len(test_df) - 1, "close"] / test_df.loc[0, "close"] - 1) * 100

    print("\n=== Backtest (2330.TW) — Integration #3 RF-gated PPO ===")
    print(f"Test period: {test_df.loc[0, 'Date'].date()} -> {test_df.loc[len(test_df)-1, 'Date'].date()}")
    print_roi(f"ROI: {bt['roi_pct']:.2f}%")
    print(f"Max Drawdown: {bt['max_drawdown_pct']:.2f}%")
    print(f"Sharpe (approx): {bt['sharpe']:.2f}")
    print(f"Trades: {bt['trades']}")
    print_roi(f"Buy&Hold ROI (test): {bh_roi:.2f}%")

    # Save model
    out_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(out_dir, exist_ok=True)
    safe_ticker = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_").lower()
    out_path = os.path.join(out_dir, f"ppo_rf_gated_{safe_ticker}")
    model.save(out_path)
    print(f"Model saved: {out_path}.zip")


if __name__ == "__main__":
    main()
