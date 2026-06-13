"""
Batch train requested USA tickers with PPO.

Requested tickers:
RKLB, ASTS, FLY, SATS, PL, AMZN, TMUS, QCOM, FLYX, GOOGL, BAC, DXYZ, XOVR, VCX
"""
import argparse
import io
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLBACKEND"] = "Agg"

ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_TMP = os.path.join(ROOT, "TMP")
LOCAL_YF_CACHE = os.path.join(ROOT, ".cache_yfinance_runtime")
os.makedirs(LOCAL_TMP, exist_ok=True)
os.makedirs(LOCAL_YF_CACHE, exist_ok=True)
os.environ.setdefault("TMP", LOCAL_TMP)
os.environ.setdefault("TEMP", LOCAL_TMP)
os.environ.setdefault("YFINANCE_CACHE_DIR", LOCAL_YF_CACHE)
os.environ.setdefault("YF_CACHE_DIR", LOCAL_YF_CACHE)
os.environ.setdefault("REQUESTS_CACHE_PATH", os.path.join(LOCAL_YF_CACHE, "requests_cache"))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
warnings.filterwarnings("ignore")
os.chdir(ROOT)

import gymnasium as gym
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import spaces
from sklearn.metrics import accuracy_score
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    yf.set_tz_cache_location(LOCAL_YF_CACHE)
    yf.cache.set_cache_location(LOCAL_YF_CACHE)
except Exception:
    pass


REQUESTED_STOCKS = [
    {"ticker": "RKLB", "name": "Rocket Lab USA"},
    {"ticker": "ASTS", "name": "AST SpaceMobile"},
    {"ticker": "FLY", "name": "FLY"},
    {"ticker": "SATS", "name": "EchoStar / SATS"},
    {"ticker": "PL", "name": "Planet Labs"},
    {"ticker": "AMZN", "name": "Amazon"},
    {"ticker": "TMUS", "name": "T-Mobile US"},
    {"ticker": "QCOM", "name": "Qualcomm"},
    {"ticker": "FLYX", "name": "flyExclusive"},
    {"ticker": "GOOGL", "name": "Alphabet"},
    {"ticker": "BAC", "name": "Bank of America"},
    {"ticker": "DXYZ", "name": "Destiny Tech100"},
    {"ticker": "XOVR", "name": "XOVR"},
    {"ticker": "VCX", "name": "VCX"},
]

START_DATE = "2015-01-01"
END_DATE = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
MIN_CLEAN_ROWS = 120


class PPOTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.i = 0
        self.balance = self.initial_balance
        self.shares = 0
        self.total_profit = 0.0
        self.total_trades = 0
        return self._obs(), {}

    def _obs(self):
        row = self.df.iloc[self.i]
        price = float(row["close"])
        total_value = self.balance + self.shares * price
        stock_ratio = (self.shares * price) / total_value if total_value > 0 else 0.0
        cash_ratio = self.balance / total_value if total_value > 0 else 1.0
        return np.array([
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
            float(self.total_profit),
            float(stock_ratio),
            float(cash_ratio),
        ], dtype=np.float32)

    def step(self, action):
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action_value = float(np.clip(action_value, -1.0, 1.0))
        price = float(self.df.iloc[self.i]["close"])
        old_value = self.balance + self.shares * price

        if action_value < -0.1:
            shares_to_sell = int(self.shares * abs(action_value))
            if shares_to_sell > 0:
                self.balance += shares_to_sell * price
                self.shares -= shares_to_sell
                self.total_trades += 1
        elif action_value > 0.1:
            max_can_buy = int(self.balance // price)
            shares_to_buy = int(max_can_buy * action_value)
            if shares_to_buy > 0:
                self.balance -= shares_to_buy * price
                self.shares += shares_to_buy
                self.total_trades += 1

        new_value = self.balance + self.shares * price
        self.total_profit = new_value - self.initial_balance
        daily_return_reward = (new_value - old_value) / self.initial_balance
        total_profit_reward = self.total_profit / self.initial_balance
        trade_incentive = 0.002 if abs(action_value) > 0.1 else 0.0
        cash_penalty = -0.001 if self.balance > new_value * 0.9 else 0.0
        reward = daily_return_reward + total_profit_reward + trade_incentive + cash_penalty

        self.i += 1
        done = self.i >= len(self.df) - 1
        return self._obs(), float(reward), done, False, {}


def add_technical_indicators(df):
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_30"] = df["close"].rolling(30).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_200"] = df["close"].rolling(200).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_middle"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_middle"] + df["bb_std"] * 2
    df["bb_lower"] = df["bb_middle"] - df["bb_std"] * 2

    df["future_return"] = df["close"].shift(-5) / df["close"] - 1
    df["target"] = (df["future_return"] > 0.02).astype(int)
    return df.bfill().ffill()


def download_stock(ticker):
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        progress=False,
        auto_adjust=True,
        threads=False,
        timeout=30,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.empty:
        return df
    return df.rename(
        columns={
            "Close": "close",
            "Volume": "volume",
            "Open": "open",
            "High": "high",
            "Low": "low",
        }
    ).reset_index()


def evaluate_model(model, test_df):
    env = PPOTradingEnv(test_df.reset_index(drop=True))
    obs, _ = env.reset()
    preds = []
    labels = []
    portfolio_values = []

    for i in range(len(test_df) - 6):
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        preds.append(1 if action_value > 0.1 else 0)
        labels.append(int(test_df.iloc[i]["target"]))
        obs, _, done, _, _ = env.step(action)
        price = float(test_df.iloc[min(i, len(test_df) - 1)]["close"])
        portfolio_values.append(env.balance + env.shares * price)
        if done:
            break

    accuracy = accuracy_score(labels, preds) if preds else 0.0
    buy_signals = int(sum(preds))
    final_value = portfolio_values[-1] if portfolio_values else env.initial_balance
    total_return = (final_value / env.initial_balance - 1) * 100
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = 0.0
    if len(returns) > 5 and returns.std() > 0:
        sharpe = float((returns.mean() / returns.std()) * np.sqrt(252))
    return {
        "accuracy": round(float(accuracy * 100), 2),
        "buy_signals": buy_signals,
        "predictions": len(preds),
        "total_return": round(float(total_return), 2),
        "sharpe_ratio": round(float(sharpe), 3),
    }


def train_stock(stock, timesteps):
    ticker = stock["ticker"].upper().strip()
    name = stock["name"]
    print("\n" + "=" * 80)
    print(f"Training PPO {ticker} - {name}")
    print("=" * 80)

    try:
        df = download_stock(ticker)
        if df.empty:
            return {"ticker": ticker, "name": name, "success": False, "reason": "empty Yahoo data"}

        print(f"Downloaded rows: {len(df)}")
        df = add_technical_indicators(df)
        clean = df.dropna(subset=[
            "close", "volume", "sma_10", "sma_30", "sma_50",
            "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "target",
        ]).copy()

        if len(clean) < MIN_CLEAN_ROWS:
            return {
                "ticker": ticker,
                "name": name,
                "success": False,
                "reason": f"not enough clean rows: {len(clean)}",
                "rows": int(len(df)),
                "clean_rows": int(len(clean)),
            }

        split = int(len(clean) * 0.8)
        train_df = clean.iloc[:split].copy()
        test_df = clean.iloc[split:].copy()
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")

        env = DummyVecEnv([lambda: PPOTradingEnv(train_df)])
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.01,
            seed=42,
        )
        model.learn(total_timesteps=int(timesteps))

        model_base = f"ppo_{ticker.lower()}_improved"
        model.save(model_base)

        eval_stats = evaluate_model(model, test_df)
        accuracy_data = {
            "symbol": ticker,
            "model_type": "PPO",
            "training_accuracy": eval_stats["accuracy"],
            "validation_accuracy": eval_stats["accuracy"],
            "backtest_accuracy": eval_stats["accuracy"],
            "win_rate": eval_stats["accuracy"],
            "sharpe_ratio": eval_stats["sharpe_ratio"],
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "timesteps": int(timesteps),
            "rows": int(len(df)),
            "clean_rows": int(len(clean)),
            "test_return_pct": eval_stats["total_return"],
            "buy_signals": eval_stats["buy_signals"],
            "predictions": eval_stats["predictions"],
            "start_date": START_DATE,
            "end_date": END_DATE,
        }
        with open(f"model_accuracy_{ticker}.json", "w", encoding="utf-8") as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
        with open(f"model_accuracy_{ticker}_ppo.json", "w", encoding="utf-8") as f:
            json.dump(accuracy_data, f, ensure_ascii=False, indent=2)

        print(f"Saved: {model_base}.zip")
        print(f"PPO validation accuracy: {eval_stats['accuracy']:.2f}%")
        print(f"Backtest return: {eval_stats['total_return']:.2f}% | Sharpe: {eval_stats['sharpe_ratio']:.3f}")
        return {
            "ticker": ticker,
            "name": name,
            "success": True,
            "model_file": f"{model_base}.zip",
            "validation_accuracy": eval_stats["accuracy"],
            "test_return_pct": eval_stats["total_return"],
            "sharpe_ratio": eval_stats["sharpe_ratio"],
            "buy_signals": eval_stats["buy_signals"],
            "predictions": eval_stats["predictions"],
            "rows": int(len(df)),
            "clean_rows": int(len(clean)),
            "timesteps": int(timesteps),
        }
    except Exception as exc:
        return {"ticker": ticker, "name": name, "success": False, "reason": str(exc)[:300]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000, help="PPO timesteps per ticker")
    parser.add_argument("--tickers", nargs="*", help="Optional subset of tickers to train")
    return parser.parse_args()


def main():
    args = parse_args()
    stocks = REQUESTED_STOCKS
    if args.tickers:
        wanted = {t.upper() for t in args.tickers}
        stocks = [s for s in REQUESTED_STOCKS if s["ticker"].upper() in wanted]

    print("=" * 80)
    print("Requested USA PPO training")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Timesteps per ticker: {args.timesteps:,}")
    print(f"Tickers: {', '.join(s['ticker'] for s in stocks)}")
    print("=" * 80)

    results = [train_stock(stock, args.timesteps) for stock in stocks]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = f"requested_usa_ppo_training_summary_{timestamp}.json"
    summary_csv = f"requested_usa_ppo_training_summary_{timestamp}.csv"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    pd.DataFrame(results).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("PPO training summary")
    print("=" * 80)
    for result in results:
        if result["success"]:
            print(
                f"OK   {result['ticker']:<6} "
                f"acc={result['validation_accuracy']:>6.2f}% "
                f"ret={result['test_return_pct']:>7.2f}% "
                f"rows={result['clean_rows']:>4} "
                f"{result['model_file']}"
            )
        else:
            print(f"FAIL {result['ticker']:<6} {result.get('reason', 'unknown')}")
    print("-" * 80)
    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV:  {summary_csv}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
