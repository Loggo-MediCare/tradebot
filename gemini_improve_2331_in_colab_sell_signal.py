import os
import sys
import io
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# !pip install stable-baselines3

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import warnings

# Suppress warnings and configure encoding
warnings.filterwarnings('ignore')
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # Removed this line
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. Advanced Feature Engineering
# ==========================================
def add_technical_indicators(df):
    """
    Computes normalized technical indicators.
    Crucial Change: Inputs are converted to Ratios/Percentages to help the AI converge.
    """
    df = df.copy()

    # 1. Log Returns (Better for AI than simple % change)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # 2. Normalized SMA (Distance from price)
    df['sma_10'] = df['close'] / df['close'].rolling(10).mean() - 1
    df['sma_30'] = df['close'] / df['close'].rolling(30).mean() - 1
    df['sma_50'] = df['close'] / df['close'].rolling(50).mean() - 1

    # 3. RSI (Standard 0-100, scaled to 0-1)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    # Handle division by zero for rs
    rs = np.where(loss == 0, np.inf, gain / loss)
    df['rsi'] = (100 - (100 / (1 + rs))) / 100.0  # Scaled 0 to 1
    df['rsi'] = df['rsi'].fillna(0.5) # Fill NaN (e.g., 0/0) with 0.5 (mid-RSI)

    # 4. MACD (Normalized by price)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean() # adjust=False for typical MACD
    ema_26 = df['close'].ewm(span=26, adjust=False).mean() # adjust=False for typical MACD
    df['macd'] = (ema_12 - ema_26) / df['close']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 5. Bollinger Bands (Position relative to bands)
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    # Position: 0 = Lower Band, 0.5 = Mid, 1 = Upper Band
    # Robustly handle bb_std == 0
    df['bb_position'] = (df['close'] - (bb_mid - 2 * bb_std)) / (4 * bb_std)
    df['bb_position'] = df['bb_position'].fillna(0.5) # If bb_std was NaN
    # If bb_std was 0, meaning (4 * bb_std) is 0, the result might be inf or NaN.
    # In such a case, the price is flat, so it's effectively at the mid-band.
    df['bb_position'] = np.where(bb_std == 0, 0.5, df['bb_position'])


    # 6. Volume Change (Log volume delta)
    df['vol_change'] = np.log(df['volume'] / df['volume'].shift(1) + 1e-5)

    # 7. Volatility (ATR-like normalized)
    df['volatility'] = (df['high'] - df['low']) / df['close']

    # Drop NaNs created by rolling windows (this is crucial)
    df.dropna(inplace=True)

    # IMPT: Ensure no infinite values after all calculations, replace with max/min finite values
    # Or, perhaps better, clip them to a reasonable range. VecNormalize should handle scaling,
    # but infinities will cause NaNs after normalization.
    for col in ['log_ret', 'sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bb_position', 'vol_change', 'volatility']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            # A final fillna before returning, though dropna should have caught most
            # For remaining NaNs (e.g., from divisions by zero for very specific edge cases),
            # fill with a sensible value like 0 or the mean. Mean might be better for VecNormalize.
            # However, for continuous values, 0 is often a safe 'no change' value.
            df[col] = df[col].fillna(0.0) # Filling NaNs that somehow survived with 0.

    # Reset index to ensure alignment
    df.reset_index(drop=True, inplace=True)

    # Check for any remaining NaNs or Infs - this should not happen if previous steps are robust
    if df.isnull().values.any() or np.isinf(df.values).any():
        print("Warning: NaNs or Infs still present after feature engineering. This might cause issues.")
        # Further debugging could involve saving df to CSV here and inspecting
        # df.to_csv('debug_df_after_features.csv')

    return df

# ==========================================
# 2. Windowed Trading Environment
# ==========================================
class AdvancedTradingEnv(gym.Env):
    """
    Advanced Environment with:
    1. Windowed Observation (See past N days)
    2. Sharpe-Ratio based Reward
    3. Realistic Commission Fees
    """
    def __init__(self, df, initial_balance=100000, window_size=10):
        super(AdvancedTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.commission = 0.001425  # Taiwan Stock Fee (0.1425%)
        self.tax = 0.003            # Taiwan Transaction Tax (0.3%)

        # Action: Continuous [-1, 1] (Sell 100% ... Hold ... Buy 100%)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation: [Window_Size, Features]
        # Features used: log_ret, sma_10, sma_30, rsi, macd, bb_pos, vol_change, volatility
        self.n_features = 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.n_features),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.history_returns = []

        return self._get_observation(), {}

    def _get_observation(self):
        # Return a window of data (Shape: window_size x n_features)
        # We slice from (current - window) to current
        obs_data = self.df.iloc[self.current_step - self.window_size : self.current_step]

        obs = obs_data[[
            'log_ret', 'sma_10', 'sma_30', 'rsi',
            'macd', 'bb_position', 'vol_change', 'volatility'
        ]].values

        return obs.astype(np.float32)

    def step(self, action):
        # 1. Execute Trade
        current_price = self.df.iloc[self.current_step]['close']
        action = float(action[0])

        prev_net_worth = self.net_worth

        # Trade Logic
        if action > 0.1: # BUY
            buy_fraction = action
            cost_per_share = current_price * (1 + self.commission)
            max_shares = (self.balance * buy_fraction) // cost_per_share

            if max_shares > 0:
                self.balance -= max_shares * cost_per_share
                self.shares_held += max_shares

        elif action < -0.1 and self.shares_held > 0: # SELL
            sell_fraction = abs(action)
            shares_sold = int(self.shares_held * sell_fraction)

            if shares_sold > 0:
                revenue = shares_sold * current_price * (1 - self.commission - self.tax)
                self.balance += revenue
                self.shares_held -= shares_sold

        # 2. Update State
        self.net_worth = self.balance + (self.shares_held * current_price)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)

        # 3. Calculate Reward (Sortino/Sharpe like)
        step_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.history_returns.append(step_return)

        # Reward Component A: Raw Return
        reward = step_return * 100

        # Reward Component B: Volatility Penalty (Risk Management)
        if len(self.history_returns) > 5:
            std_dev = np.std(self.history_returns[-5:])
            if std_dev > 0:
                reward -= (std_dev * 5) # Penalize erratic behavior

        # Reward Component C: Drawdown Penalty
        drawdown = (self.net_worth - self.max_net_worth) / self.max_net_worth
        reward += drawdown * 10 # Drawdown is negative, so this subtracts reward

        # 4. Advance Step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

# ==========================================
# 3. Training & Execution
# ==========================================
def download_data(ticker='2330.TW', start='2015-01-01', end='2025-01-01'):
    print(f"📥 Downloading {ticker}...")
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'High': 'high', 'Low': 'low', 'Open': 'open'})

    # Basic data validation
    if df.empty or 'close' not in df.columns:
        raise ValueError("Data download failed.")

    return df

def train_and_save(ticker='2330.TW'):
    # 1. Prepare Data
    df = download_data(ticker)
    df = add_technical_indicators(df)

    # Split Train/Test (Keep chronological order!)
    split = int(len(df) * 0.85)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]

    print(f"📊 Training Data: {len(train_df)} | Test Data: {len(test_df)}")

    # 2. Create Environment
    # We wrap the env in a lambda for DummyVecEnv, then VecNormalize
    # VecNormalize is CRITICAL: It scales rewards and observations to help the optimizer.
    env_maker = lambda: AdvancedTradingEnv(train_df)
    env = DummyVecEnv([env_maker])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Define Model (PPO)
    # Using specific net_arch for deeper understanding
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0002,      # Slightly lower LR for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # Encourages exploration
        policy_kwargs=policy_kwargs,
        device="auto"
    )

    # 4. Train
    print("🚀 Starting Training (150,000 steps)...")
    model.learn(total_timesteps=150000)

    # 5. Save Model AND Normalization Stats
    save_path = f"ppo_{ticker.replace('.TW', '')}_advanced"
    model.save(save_path)
    env.save(f"{save_path}_vecnorm.pkl") # MUST save normalization stats
    print(f"✅ Model saved to {save_path}.zip")
    print(f"✅ Normalization stats saved to {save_path}_vecnorm.pkl")

    return model, env, test_df

# ==========================================
# 4. Model Accuracy Evaluation (回測準確度檢查)
# ==========================================
def evaluate_model_accuracy(ticker='2330.TW', model_path=None, test_days=60):
    """
    評估模型在歷史數據上的準確度
    - 計算信號準確率 (買入信號後是否上漲)
    - 計算勝率、平均報酬、夏普比率
    """
    import yfinance as yf

    print(f"\n{'='*60}")
    print(f"📊 模型準確度評估 - {ticker}")
    print(f"{'='*60}")

    # 下載數據 (需要更多歷史數據來評估)
    df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'High': 'high', 'Low': 'low', 'Open': 'open'})

    # 保存原始價格用於計算實際報酬
    df['close_raw'] = df['close'].copy()
    df_with_date = df.copy()

    # 處理特徵
    df_processed = add_technical_indicators(df)

    if len(df_processed) < test_days + 15:
        print("❌ 數據不足，無法評估")
        return None

    # 載入模型
    if model_path is None:
        model_path = f"ppo_{ticker.replace('.TW', '')}_advanced"

    try:
        env = DummyVecEnv([lambda: AdvancedTradingEnv(df_processed)])
        env = VecNormalize.load(f"{model_path}_vecnorm.pkl", env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ 載入模型失敗: {e}")
        return None

    # 回測評估
    window_size = 10
    results = []

    # 從 test_days 天前開始評估
    start_idx = len(df_processed) - test_days

    print(f"\n🔄 回測 {test_days} 天的交易信號...")

    for i in range(start_idx, len(df_processed) - 1):  # -1 因為需要下一天的價格
        # 構建觀察窗口
        obs_data = df_processed.iloc[i - window_size : i]
        raw_obs = obs_data[[
            'log_ret', 'sma_10', 'sma_30', 'rsi',
            'macd', 'bb_position', 'vol_change', 'volatility'
        ]].values.astype(np.float32)

        # 標準化並預測
        norm_obs = env.normalize_obs(raw_obs)
        action, _ = model.predict(norm_obs, deterministic=True)
        action_val = float(action[0][0]) if isinstance(action[0], np.ndarray) else float(action[0])

        # 獲取當天和下一天的價格
        current_price = df_processed.iloc[i]['close']
        next_price = df_processed.iloc[i + 1]['close']
        actual_return = (next_price - current_price) / current_price * 100

        # 判斷信號
        if action_val > 0.3:
            signal = "STRONG_BUY"
        elif action_val > 0.05:
            signal = "WEAK_BUY"
        elif action_val < -0.3:
            signal = "STRONG_SELL"
        elif action_val < -0.05:
            signal = "WEAK_SELL"
        else:
            signal = "HOLD"

        # 判斷是否正確
        if signal in ["STRONG_BUY", "WEAK_BUY"]:
            correct = actual_return > 0
        elif signal in ["STRONG_SELL", "WEAK_SELL"]:
            correct = actual_return < 0
        else:
            correct = abs(actual_return) < 1  # HOLD 時波動小於1%算正確

        results.append({
            'date': i,
            'signal': signal,
            'action_val': action_val,
            'actual_return': actual_return,
            'correct': correct
        })

    # 計算統計數據
    df_results = pd.DataFrame(results)

    # 整體準確率
    overall_accuracy = df_results['correct'].mean() * 100

    # 買入信號準確率
    buy_signals = df_results[df_results['signal'].isin(['STRONG_BUY', 'WEAK_BUY'])]
    buy_accuracy = buy_signals['correct'].mean() * 100 if len(buy_signals) > 0 else 0
    buy_avg_return = buy_signals['actual_return'].mean() if len(buy_signals) > 0 else 0

    # 賣出信號準確率
    sell_signals = df_results[df_results['signal'].isin(['STRONG_SELL', 'WEAK_SELL'])]
    sell_accuracy = sell_signals['correct'].mean() * 100 if len(sell_signals) > 0 else 0
    sell_avg_return = sell_signals['actual_return'].mean() if len(sell_signals) > 0 else 0

    # 強信號準確率
    strong_signals = df_results[df_results['signal'].isin(['STRONG_BUY', 'STRONG_SELL'])]
    strong_accuracy = strong_signals['correct'].mean() * 100 if len(strong_signals) > 0 else 0

    # 持有信號
    hold_signals = df_results[df_results['signal'] == 'HOLD']

    # 計算模擬交易報酬
    simulated_returns = []
    for _, row in df_results.iterrows():
        if row['signal'] in ['STRONG_BUY', 'WEAK_BUY']:
            simulated_returns.append(row['actual_return'])
        elif row['signal'] in ['STRONG_SELL', 'WEAK_SELL']:
            simulated_returns.append(-row['actual_return'])  # 做空獲利
        else:
            simulated_returns.append(0)

    total_simulated_return = sum(simulated_returns)
    avg_daily_return = np.mean(simulated_returns) if simulated_returns else 0
    sharpe_ratio = (avg_daily_return / np.std(simulated_returns) * np.sqrt(252)) if np.std(simulated_returns) > 0 else 0

    # 輸出結果
    print(f"\n{'='*60}")
    print(f"📈 評估結果 (過去 {test_days} 天)")
    print(f"{'='*60}")
    print(f"\n📊 信號分布:")
    print(f"   買入信號: {len(buy_signals)} 次 ({len(buy_signals)/len(df_results)*100:.1f}%)")
    print(f"   賣出信號: {len(sell_signals)} 次 ({len(sell_signals)/len(df_results)*100:.1f}%)")
    print(f"   持有信號: {len(hold_signals)} 次 ({len(hold_signals)/len(df_results)*100:.1f}%)")
    print(f"   強信號:   {len(strong_signals)} 次")

    print(f"\n🎯 準確率:")
    print(f"   整體準確率:     {overall_accuracy:.1f}%")
    print(f"   買入信號準確率: {buy_accuracy:.1f}% (平均報酬: {buy_avg_return:+.2f}%)")
    print(f"   賣出信號準確率: {sell_accuracy:.1f}% (平均報酬: {sell_avg_return:+.2f}%)")
    print(f"   強信號準確率:   {strong_accuracy:.1f}%")

    print(f"\n💰 模擬交易績效:")
    print(f"   總報酬率:       {total_simulated_return:+.2f}%")
    print(f"   平均日報酬:     {avg_daily_return:+.4f}%")
    print(f"   夏普比率:       {sharpe_ratio:.2f}")

    # 評級
    print(f"\n🏆 模型評級:")
    if overall_accuracy >= 60 and sharpe_ratio > 1:
        print(f"   ⭐⭐⭐ 優秀 - 可以信賴此模型的信號")
    elif overall_accuracy >= 50 and sharpe_ratio > 0.5:
        print(f"   ⭐⭐ 良好 - 信號有參考價值，但需結合其他分析")
    elif overall_accuracy >= 45:
        print(f"   ⭐ 一般 - 建議謹慎使用，多做確認")
    else:
        print(f"   ⚠️  需改進 - 模型可能需要重新訓練")

    print(f"{'='*60}\n")

    return {
        'overall_accuracy': overall_accuracy,
        'buy_accuracy': buy_accuracy,
        'sell_accuracy': sell_accuracy,
        'strong_accuracy': strong_accuracy,
        'total_return': total_simulated_return,
        'sharpe_ratio': sharpe_ratio,
        'buy_count': len(buy_signals),
        'sell_count': len(sell_signals),
        'hold_count': len(hold_signals)
    }


# ==========================================
# 5. Inference / Signal Generation
# ==========================================
def predict_signal(ticker='2330.TW', model_path=None):
    """
    Loads model + normalization stats to predict today's action.
    """
    import yfinance as yf

    print(f"\n🔮 Generating Signal for {ticker}...")

    # Load Data (Need enough history for window_size=10 + indicators)
    df = yf.download(ticker, period="3mo", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'High': 'high', 'Low': 'low', 'Open': 'open'})

    # Process
    df_processed = add_technical_indicators(df)

    if len(df_processed) < 15:
        print("❌ Not enough data.")
        return

    # Load Model
    if model_path is None:
        model_path = f"ppo_{ticker.replace('.TW', '')}_advanced"

    # IMPORTANT: We must load the training environment stats to normalize
    # the new data exactly how the model expects it.
    try:
        env = DummyVecEnv([lambda: AdvancedTradingEnv(df_processed)])
        env = VecNormalize.load(f"{model_path}_vecnorm.pkl", env)
        env.training = False # Do not update stats during inference
        env.norm_reward = False

        model = PPO.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model/stats: {e}")
        print("Did you train first? Run the script to train.")
        return

    # Get last observation window
    # The environment logic needs to be manually triggered or we just format data manually
    # Here we manually construct the window for the LAST step
    last_idx = len(df_processed) - 1
    window_size = 10

    obs_data = df_processed.iloc[last_idx - window_size : last_idx]
    raw_obs = obs_data[[
            'log_ret', 'sma_10', 'sma_30', 'rsi',
            'macd', 'bb_position', 'vol_change', 'volatility'
        ]].values.astype(np.float32)

    # Normalize observation using the loaded environment stats
    norm_obs = env.normalize_obs(raw_obs)

    # Predict
    action, _states = model.predict(norm_obs, deterministic=True)
    action_val = action[0]  # If using DummyVecEnv
    if isinstance(action_val, np.ndarray): action_val = action_val[0]

    # Output
    current_price = df['close'].iloc[-1]
    print(f"📅 Date: {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"💲 Price: {current_price:.2f}")
    print(f"🤖 AI Strength: {action_val:.4f} (Range: -1.0 to 1.0)")

    if action_val > 0.3:
        print(f"🟢 SIGNAL: STRONG BUY")
    elif action_val > 0.05:
        print(f"🟢 SIGNAL: WEAK BUY")
    elif action_val < -0.3:
        print(f"🔴 SIGNAL: STRONG SELL")
    elif action_val < -0.05:
        print(f"🔴 SIGNAL: WEAK SELL")
    else:
        print(f"🟡 SIGNAL: HOLD")

if __name__ == "__main__":
    # Example usage:
    # 1. Train the model (Uncomment to train)
    # train_and_save('2330.TW')

    # 2. Predict (Make sure model files exist)
    # predict_signal('2330.TW')

    # 3. Evaluate model accuracy (模型準確度檢查)
    # evaluate_model_accuracy('2330.TW', test_days=60)

    # To run specifically for Elite (2331) as requested in your prompt:
    print("--- Training for 2331.TW (Elite Material) ---")
    train_and_save('2331.TW')

    print("\n--- 評估模型準確度 ---")
    accuracy_result = evaluate_model_accuracy('2331.TW', test_days=60)

    print("\n--- 生成今日交易信號 ---")
    predict_signal('2331.TW')