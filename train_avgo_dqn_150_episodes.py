"""
AVGO (Broadcom) DQN Trading AI - 150 Episodes
==============================================
Deep Q-Network implementation with:
- Enhanced feature engineering (13 technical indicators)
- Transaction cost modeling (0.1% per trade)
- Improved reward shaping
- Dropout regularization for anti-overfitting
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. DQN Agent with Dropout Regularization
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size  # 0=HOLD, 1=BUY, 2=SELL

        # Hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        # Build model
        self.model = self._build_model()

    def _build_model(self):
        """Build DQN neural network with dropout"""
        model = keras.Sequential([
            keras.layers.Dense(128, input_dim=self.state_size, activation='relu'),
            keras.layers.Dropout(0.2),  # Anti-overfitting
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self):
        """Train on batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])

            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ==========================================
# 2. Trading Environment
# ==========================================
class TradingEnvironment:
    def __init__(self, df, initial_balance=10000):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = 0.001  # 0.1% per trade
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        return self._get_state()

    def _get_state(self):
        """Get current state (features)"""
        row = self.df.iloc[self.current_step]

        # 13 features
        state = np.array([
            row['close'],
            row['sma_5'],
            row['sma_10'],
            row['sma_20'],
            row['sma_50'],
            row['rsi'],
            row['macd'],
            row['macd_signal'],
            row['bb_upper'],
            row['bb_lower'],
            row['ema_12'],
            row['roc'],
            row['volatility']
        ], dtype=np.float32)

        return state.reshape(1, -1)

    def step(self, action):
        """Execute action and return next state"""
        current_price = float(self.df.iloc[self.current_step]['close'])

        # Execute action
        if action == 1:  # BUY
            max_shares = int(self.balance / (current_price * (1 + self.transaction_cost)))
            if max_shares > 0:
                cost = max_shares * current_price * (1 + self.transaction_cost)
                self.balance -= cost
                self.shares_held += max_shares
                self.total_trades += 1

        elif action == 2:  # SELL
            if self.shares_held > 0:
                revenue = self.shares_held * current_price * (1 - self.transaction_cost)
                self.balance += revenue
                self.shares_held = 0
                self.total_trades += 1

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Calculate reward
        new_portfolio_value = self.balance + self.shares_held * current_price
        portfolio_change = float(new_portfolio_value - self.initial_balance)

        # Enhanced reward shaping
        reward = portfolio_change / self.initial_balance

        # Bonus for profitable trades
        if portfolio_change > 0:
            reward += 0.01

        next_state = self._get_state() if not done else None

        return next_state, float(reward), done, float(new_portfolio_value)

# ==========================================
# 3. Technical Indicators
# ==========================================
def add_technical_indicators(df):
    """Add 13 technical indicators"""
    print("Adding technical indicators...")

    # Simple Moving Averages
    df['sma_5'] = df['close'].rolling(5).mean()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # Rate of Change
    df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    # Volatility
    df['volatility'] = df['close'].rolling(10).std()

    # Fill NaN
    df = df.fillna(method='bfill').fillna(method='ffill')

    print(f"✅ Added 13 technical indicators")
    return df

# ==========================================
# 4. Download Data
# ==========================================
def download_data(ticker='AVGO', start_date='2015-01-01', end_date='2024-01-01'):
    """Download stock data"""
    print("=" * 70)
    print(f"Downloading {ticker} stock data...")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 70)

    try:
        import yfinance as yf

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            raise ValueError(f"Failed to download {ticker} data")

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()

        print(f"✅ Downloaded {len(df)} days of data")
        print(f"   Price range: ${float(df['close'].min()):.2f} - ${float(df['close'].max()):.2f}")

        return df

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

# ==========================================
# 5. Main Training Loop
# ==========================================
if __name__ == "__main__":
    print("🚀 AVGO DQN Trading AI - 150 Episodes Training")
    print("=" * 70)

    # Configuration
    TICKER = 'AVGO'
    START_DATE = '2015-01-01'
    END_DATE = '2024-01-01'
    EPISODES = 150  # 🔥 Changed from 20 to 150
    TRAIN_TEST_SPLIT = 0.8

    print(f"Ticker: {TICKER}")
    print(f"Data range: {START_DATE} - {END_DATE}")
    print(f"Training episodes: {EPISODES}")
    print("=" * 70)

    # 1. Download data
    df = download_data(TICKER, START_DATE, END_DATE)

    if df is None:
        print("\n❌ Data download failed")
        exit(1)

    # 2. Add technical indicators
    df = add_technical_indicators(df)

    # 3. Split data
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nData split:")
    print(f"  Training set: {len(train_df)} days")
    print(f"  Test set: {len(test_df)} days")

    # 4. Initialize agent and environment
    state_size = 13  # Number of features
    action_size = 3  # HOLD, BUY, SELL

    agent = DQNAgent(state_size, action_size)
    env = TradingEnvironment(train_df)

    # 5. Training loop
    print("\n" + "=" * 70)
    print("Starting DQN Training...")
    print("=" * 70)

    episode_rewards = []
    episode_profits = []

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, portfolio_value = env.step(action)

            if next_state is not None:
                agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                profit_pct = (portfolio_value - env.initial_balance) / env.initial_balance * 100
                episode_rewards.append(total_reward)
                episode_profits.append(profit_pct)

                print(f"Episode {episode+1}/{EPISODES} | "
                      f"Reward: {total_reward:.4f} | "
                      f"Profit: {profit_pct:+.2f}% | "
                      f"Epsilon: {agent.epsilon:.3f} | "
                      f"Trades: {env.total_trades}")
                break

        # Train agent
        agent.replay()

    # 6. Save model
    model_path = f"dqn_{TICKER.lower()}_150ep.h5"
    agent.model.save(model_path)
    print(f"\n✅ Model saved: {model_path}")

    # 7. Plot training progress
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episode_profits)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Training Profit % per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Profit %')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'dqn_{TICKER.lower()}_training_150ep.png')
    print(f"✅ Training plot saved: dqn_{TICKER.lower()}_training_150ep.png")

    # 8. Test on test set
    print("\n" + "=" * 70)
    print("Testing on test set...")
    print("=" * 70)

    test_env = TradingEnvironment(test_df)
    state = test_env.reset()
    agent.epsilon = 0  # No exploration during testing

    portfolio_values = []

    while True:
        action = agent.act(state)
        next_state, reward, done, portfolio_value = test_env.step(action)
        portfolio_values.append(portfolio_value)

        state = next_state

        if done:
            break

    final_profit_pct = (portfolio_value - test_env.initial_balance) / test_env.initial_balance * 100

    print(f"\n✅ Test Results:")
    print(f"   Initial balance: ${test_env.initial_balance:,.2f}")
    print(f"   Final portfolio: ${portfolio_value:,.2f}")
    print(f"   Profit: {final_profit_pct:+.2f}%")
    print(f"   Total trades: {test_env.total_trades}")

    # Plot test results
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value', color='green')
    plt.axhline(y=test_env.initial_balance, color='r', linestyle='--', label='Initial Balance')
    plt.title(f'{TICKER} DQN Test Results - 150 Episodes')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'dqn_{TICKER.lower()}_test_150ep.png')
    print(f"✅ Test plot saved: dqn_{TICKER.lower()}_test_150ep.png")

    print("\n" + "=" * 70)
    print("Training Summary (150 Episodes):")
    print("=" * 70)
    print(f"  Average training reward: {np.mean(episode_rewards):.4f}")
    print(f"  Average training profit: {np.mean(episode_profits):+.2f}%")
    print(f"  Best training profit: {np.max(episode_profits):+.2f}%")
    print(f"  Test profit: {final_profit_pct:+.2f}%")
    print("=" * 70)
