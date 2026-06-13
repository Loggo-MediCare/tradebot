# -*- coding: utf-8 -*-
"""
Improved RL Trading Strategy - A2C Version (Policy Gradient)
Features:
1. Dual-Branch Network: LSTM for market trends, Dense for account status.
2. On-Policy Learning: Update every few steps with advantages.
3. Anti-Overfitting: Transaction costs, dropout, entropy regularization.
"""
import sys
import io
import os
# Fix encoding for Windows consoles
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import yfinance as yf
import pandas as pd
import numpy as np
# Fix matplotlib backend
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F
import warnings
from roi_control import print_roi
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
# Font settings for charts to support Chinese characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
print("🚀 Starting A2C-Enhanced RL Trading Strategy...")

# =========================================================
# 1. Data Acquisition (unchanged)
# =========================================================
def download_data(ticker, days=730): # Download 2 years
    """Downloads stock data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    print(f"📥 Downloading {ticker} from {start_date.date()} to {end_date.date()}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass
        df = df.rename(columns={"Close": "Close", "Open": "Open", "High": "High", "Low": "Low", "Volume": "Volume"})
        if df.empty:
            raise ValueError("Downloaded DataFrame is empty.")
        return df
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return None
ticker = 'NVDA'
df = download_data(ticker, days=730)
if df is None:
    sys.exit(1)
print(f"✓ Data acquired: {len(df)} trading days")

# =========================================================
# 2. Enhanced Feature Engineering (unchanged)
# =========================================================
def create_features(df):
    """Enhanced technical indicators"""
    features = df.copy()
    # Moving Averages
    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()
    features['SMA_50'] = features['Close'].rolling(window=50).mean()
    # Momentum
    features['ROC_5'] = features['Close'].pct_change(periods=5) * 100
    features['ROC_10'] = features['Close'].pct_change(periods=10) * 100
    # RSI
    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))
    # Volatility
    features['Volatility_10'] = features['Close'].pct_change().rolling(window=10).std()
    features['Volatility_20'] = features['Close'].pct_change().rolling(window=20).std()
    # MACD
    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    sma_20 = features['Close'].rolling(window=20).mean()
    std_20 = features['Close'].rolling(window=20).std()
    features['BB_upper'] = sma_20 + (std_20 * 2)
    features['BB_lower'] = sma_20 - (std_20 * 2)
    features['BB_position'] = (features['Close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])
    # Volume
    features['Volume_SMA'] = features['Volume'].rolling(window=20).mean()
    features['Volume_ratio'] = features['Volume'] / features['Volume_SMA']
    features = features.dropna()
    feature_cols = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'ROC_5', 'ROC_10',
                   'RSI', 'Volatility_10', 'Volatility_20', 'MACD', 'Signal_Line',
                   'BB_position', 'Volume_ratio']
    scaler = MinMaxScaler()
    features[feature_cols] = scaler.fit_transform(features[feature_cols])
    return features, feature_cols
df_features, feature_cols = create_features(df)
print(f"✓ Features created: {len(feature_cols)} features per step")

# =========================================================
# 3. Improved Trading Environment (unchanged)
# =========================================================
class ImprovedTradingEnvironment:
    def __init__(self, df, feature_cols, initial_balance=10000, window_size=10, transaction_cost=0.001):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.transaction_cost = transaction_cost
        self.trades_history = []
        self.portfolio_value_history = []
    def reset(self):
        self.balance = self.initial_balance
        self.total_shares = 0
        self.current_step = self.window_size
        self.trades_history = []
        self.portfolio_value_history = []
        return self._get_state()
    def _get_state(self):
        # 1. Market Data (Sequence)
        start = self.current_step - self.window_size
        market_state = self.df[self.feature_cols].iloc[start:self.current_step].values.flatten()
        # 2. Account Data (Static Snapshot)
        current_price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + (self.total_shares * current_price)
        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 100,
            portfolio_value / self.initial_balance,
            (portfolio_value - self.initial_balance) / self.initial_balance,
            len(self.trades_history) / 100,
        ])
        return np.concatenate((market_state, account_state))
    def step(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])
        prev_portfolio_value = self.balance + (self.total_shares * current_price)
        cost_penalty = 0
        if action == 1: # BUY
            invest_amount = self.balance * 0.5
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                transaction_fee = cost * self.transaction_cost
                total_cost = cost + transaction_fee
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.total_shares += shares_to_buy
                    self.trades_history.append(('BUY', self.current_step, current_price, shares_to_buy))
                    cost_penalty = transaction_fee
        elif action == 2: # SELL
            shares_to_sell = int(self.total_shares * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price
                transaction_fee = revenue * self.transaction_cost
                net_revenue = revenue - transaction_fee
                self.balance += net_revenue
                self.total_shares -= shares_to_sell
                self.trades_history.append(('SELL', self.current_step, current_price, shares_to_sell))
                cost_penalty = transaction_fee
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = float(self.df['Close'].iloc[self.current_step])
        new_portfolio_value = self.balance + (self.total_shares * next_price)
        self.portfolio_value_history.append(new_portfolio_value)
        # Rewards
        portfolio_change = new_portfolio_value - prev_portfolio_value
        reward = (portfolio_change / self.initial_balance) * 100
        reward -= (cost_penalty / self.initial_balance) * 100
        if action in [1, 2]:
            if portfolio_change > 0:
                reward += 0.5
            else:
                reward -= 0.5
        if done:
            self.balance += self.total_shares * next_price
            self.total_shares = 0
            final_profit_pct = (self.balance - self.initial_balance) / self.initial_balance
            if final_profit_pct > 0:
                reward += final_profit_pct * 20
            else:
                reward += final_profit_pct * 10
        return self._get_state(), reward, done, new_portfolio_value
    def get_final_profit(self):
        return self.balance - self.initial_balance

# =========================================================
# 4. A2C Agent (PyTorch)
# =========================================================
class ActorCritic(nn.Module):
    def __init__(self, window_size, num_features, action_size=3, account_dim=5):
        super(ActorCritic, self).__init__()
        self.window_size = window_size
        self.num_features = num_features
        self.market_data_len = window_size * num_features
        self.account_dim = account_dim
        self.action_size = action_size

        # Branch 1: Market Data (LSTM)
        self.lstm = nn.LSTM(num_features, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)

        # Branch 2: Account Data (Dense)
        self.fc_account = nn.Linear(self.account_dim, 32)

        # Merge
        self.fc_shared = nn.Linear(64 + 32, 64)
        self.fc_shared2 = nn.Linear(64, 32)

        # Actor and Critic
        self.fc_pi = nn.Linear(32, action_size)
        self.fc_v = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch, state_size)
        batch_size = x.size(0)
        market_flat = x[:, :self.market_data_len]
        market_reshaped = market_flat.view(batch_size, self.window_size, self.num_features)
        lstm_out, _ = self.lstm(market_reshaped)
        lstm_out = lstm_out[:, -1, :]  # Last time step
        lstm_out = self.dropout(lstm_out)

        account_flat = x[:, self.market_data_len:]
        account_out = F.relu(self.fc_account(account_flat))

        merged = torch.cat([lstm_out, account_out], dim=1)
        x = F.relu(self.fc_shared(merged))
        x = self.dropout(x)
        x = F.relu(self.fc_shared2(x))
        return x

    def pi(self, x):
        x = self.forward(x)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=1)
        return prob

    def v(self, x):
        x = self.forward(x)
        v = self.fc_v(x)
        return v

class A2CAgent:
    def __init__(self, window_size, num_features):
        self.model = ActorCritic(window_size, num_features)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.gamma = 0.95
        self.entropy_coef = 0.01

    def act(self, state, deterministic=False):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        prob = self.model.pi(state_tensor)
        if deterministic:
            a = prob.argmax(1).item()
            log_prob = torch.log(prob[0, a])
        else:
            m = Categorical(prob)
            a = m.sample().item()
            log_prob = m.log_prob(torch.tensor(a))
        value = self.model.v(state_tensor)[0]
        return a, log_prob, value, prob[0]

    def update(self, s_lst, a_lst, r_lst, log_prob_lst, value_lst, mask_lst, next_state, done):
        self.optimizer.zero_grad()

        s_tensor = torch.tensor(s_lst).float()
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        with torch.no_grad():
            next_value = self.model.v(next_state_tensor)[0].item() if not done else 0

        returns = []
        G = next_value
        for r, m in zip(r_lst[::-1], mask_lst[::-1]):
            G = r + self.gamma * G * m
            returns.append(G)
        returns = torch.tensor(returns[::-1]).float()

        values = torch.cat(value_lst)
        adv = returns - values

        log_probs = torch.stack(log_prob_lst)
        actor_loss = -(log_probs * adv.detach()).mean()

        critic_loss = F.smooth_l1_loss(values, returns)

        probs = torch.stack([p for p, _, _, _ in zip(*[self.act(s) for s in s_lst])])  # Recompute probs for entropy
        no, wait, better: collect prob in act
        Wait, to fix, in training, collect prob as well.

        # Assuming we have prob_lst
        prob_tensor = torch.stack(prob_lst)
        entropy = - (prob_tensor * torch.log(prob_tensor + 1e-8)).sum(1).mean()

        loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        loss.backward()
        self.optimizer.step()

# =========================================================
# 5. Training
# =========================================================
WINDOW_SIZE = 10
env = ImprovedTradingEnvironment(df_features, feature_cols, window_size=WINDOW_SIZE)
state_size = len(env.reset())
num_features = len(feature_cols)
agent = A2CAgent(WINDOW_SIZE, num_features)
EPISODES = 30
BATCH_SIZE = 32  # Not used, use update_interval
update_interval = 10
print(f"\n🎯 Training A2C Agent for {EPISODES} episodes...")
print(f" Window: {WINDOW_SIZE} days | Update Interval: {update_interval}")
best_profit = -float('inf')
best_model_state = None
best_episode = 0
training_profits = []
for e in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0
    s_lst, a_lst, r_lst, log_prob_lst, value_lst, prob_lst, mask_lst = [], [], [], [], [], [], []
    while not done:
        a, log_prob, value, prob = agent.act(state)
        next_state, reward, done, _ = env.step(a)
        s_lst.append(state)
        a_lst.append(a)
        r_lst.append(reward / 100)  # Scale reward
        log_prob_lst.append(log_prob)
        value_lst.append(value)
        prob_lst.append(prob)
        mask_lst.append(1 - int(done))
        state = next_state
        total_reward += reward
        if len(s_lst) == update_interval or done:
            agent.update(s_lst, a_lst, r_lst, log_prob_lst, value_lst, mask_lst, next_state, done)
            s_lst, a_lst, r_lst, log_prob_lst, value_lst, prob_lst, mask_lst = [], [], [], [], [], [], []
    episode_profit = env.get_final_profit()
    training_profits.append(episode_profit)
    if episode_profit > best_profit and episode_profit > 0:
        best_profit = episode_profit
        best_episode = e + 1
        best_model_state = agent.model.state_dict()
        save_msg = "⭐ NEW BEST!"
    else:
        save_msg = ""
    print(f"Episode {e+1:2d}/{EPISODES} | Profit: ${episode_profit:>9,.2f} | R: {total_reward:>6.1f} {save_msg}")
print(f"\n✓ Training Complete - Best: Episode {best_episode} with ${best_profit:,.2f}")
if best_model_state:
    agent.model.load_state_dict(best_model_state)
    print("💾 Loaded state from best episode.")
else:
    print("⚠️ Warning: No profitable episode found. Using final state.")
# Save the model
model_filename = f'{ticker}_a2c_model.pth'
torch.save(agent.model.state_dict(), model_filename)
print(f"💾 Model saved as: {model_filename}")

# =========================================================
# 6. Backtesting
# =========================================================
print("\n📊 Backtesting on training data (deterministic mode)...")
state = env.reset()
action_counts = {0: 0, 1: 0, 2: 0}
while True:
    a, _, _, _ = agent.act(state, deterministic=True)
    action_counts[a] += 1
    next_state, _, done, _ = env.step(a)
    state = next_state
    if done:
        break
final_profit = env.get_final_profit()
roi = (final_profit / 10000) * 100
print("\n" + "="*60)
print(f"BACKTEST RESULTS FOR {ticker} (A2C Version)")
print("="*60)
print(f"Initial Balance: ${10000:,.2f}")
print(f"Final Balance: ${env.balance:,.2f}")
print(f"Total Profit: ${final_profit:,.2f}")
print_roi(f"ROI: {roi:.2f}%")
print(f"Trades Executed: {len(env.trades_history)}")
print(f"Action Counts: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print("="*60)

# =========================================================
# 7. Visualization (unchanged)
# =========================================================
plt.figure(figsize=(14, 10))
# Portfolio Value
plt.subplot(3, 1, 1)
plt.plot(env.portfolio_value_history, label='Portfolio Value', color='blue', linewidth=2)
plt.axhline(y=10000, color='red', linestyle='--', alpha=0.5, label='Initial Balance')
plt.title(f'{ticker} - A2C Strategy Performance', fontsize=14, fontweight='bold')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True, alpha=0.3)
# Price and Trades
plt.subplot(3, 1, 2)
price_data = env.df['Close'].iloc[env.window_size:].reset_index(drop=True)
plt.plot(price_data, label='Stock Price', color='black', alpha=0.6, linewidth=1.5)
for trade in env.trades_history:
    action, step, price, shares = trade
    idx = step - env.window_size
    if 0 <= idx < len(price_data):
        if action == 'BUY':
            plt.scatter(idx, price, color='green', marker='^', s=120, zorder=5)
        elif action == 'SELL':
            plt.scatter(idx, price, color='red', marker='v', s=120, zorder=5)
plt.title('Trade Signals', fontsize=12)
plt.ylabel('Price ($)')
plt.legend(['Stock Price', 'Buy', 'Sell'])
plt.grid(True, alpha=0.3)
# Training Progress
plt.subplot(3, 1, 3)
plt.plot(training_profits, marker='o', linewidth=2, markersize=6, color='purple')
plt.axhline(y=best_profit, color='red', linestyle='--', alpha=0.5, label=f'Best: ${best_profit:,.0f}')
plt.title('Training Progress - Profit per Episode', fontsize=12)
plt.xlabel('Episode')
plt.ylabel('Profit ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{ticker}_A2C_RL_results.png', dpi=150, bbox_inches='tight')
print(f"\n📊 Chart saved as {ticker}_A2C_RL_results.png")
print("\n✅ Strategy Complete!")