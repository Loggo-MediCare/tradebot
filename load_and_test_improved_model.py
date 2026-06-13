# -*- coding: utf-8 -*-
"""
加載並測試改進版防過擬合模型
Load and test the improved anti-overfitting model
"""

import sys
import io
import os

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')
from roi_control import print_roi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🔄 載入改進版防過擬合模型...")

# =========================================================
# 1. 加載模型
# =========================================================
ticker = 'NVDA'
model_filename = f'{ticker}_improved_anti_overfit_model.keras'

try:
    model = tf.keras.models.load_model(model_filename)
    print(f"✓ 模型已加載: {model_filename}")
    print(f"  模型架構: {model.summary()}")
except Exception as e:
    print(f"❌ 無法加載模型: {e}")
    print(f"   請先運行 rl_trading_improved_anti_overfit.py 來訓練和保存模型")
    sys.exit(1)

# =========================================================
# 2. 下載新數據進行測試（使用不同的時間段）
# =========================================================
def download_data(ticker, days=180):
    """下載股票數據"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"\n📥 下載測試數據: {ticker} ({start_date.date()} 到 {end_date.date()})...")

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

        if df.empty:
            raise ValueError("數據為空")

        return df
    except Exception as e:
        print(f"❌ 數據下載失敗: {e}")
        return None

df = download_data(ticker, days=180)

if df is None:
    sys.exit(1)

print(f"✓ 數據已下載: {len(df)} 個交易日")

# =========================================================
# 3. 創建相同的特徵（必須與訓練時一致！）
# =========================================================
def create_features(df):
    """必須與訓練時使用的特徵完全相同"""
    features = df.copy()

    features['SMA_5'] = features['Close'].rolling(window=5).mean()
    features['SMA_10'] = features['Close'].rolling(window=10).mean()
    features['SMA_20'] = features['Close'].rolling(window=20).mean()
    features['SMA_50'] = features['Close'].rolling(window=50).mean()

    features['ROC_5'] = features['Close'].pct_change(periods=5) * 100
    features['ROC_10'] = features['Close'].pct_change(periods=10) * 100

    delta = features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    features['RSI'] = 100 - (100 / (1 + rs))

    features['Volatility_10'] = features['Close'].pct_change().rolling(window=10).std()
    features['Volatility_20'] = features['Close'].pct_change().rolling(window=20).std()

    ema_12 = features['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = features['Close'].ewm(span=26, adjust=False).mean()
    features['MACD'] = ema_12 - ema_26
    features['Signal_Line'] = features['MACD'].ewm(span=9, adjust=False).mean()

    sma_20 = features['Close'].rolling(window=20).mean()
    std_20 = features['Close'].rolling(window=20).std()
    features['BB_upper'] = sma_20 + (std_20 * 2)
    features['BB_lower'] = sma_20 - (std_20 * 2)
    features['BB_position'] = (features['Close'] - features['BB_lower']) / (features['BB_upper'] - features['BB_lower'])

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
print(f"✓ 特徵已創建: {len(feature_cols)} 個特徵, {len(df_features)} 個有效數據點")

# =========================================================
# 4. 使用模型進行交易模擬
# =========================================================
class SimpleTradingSimulator:
    def __init__(self, df, feature_cols, initial_balance=10000, window_size=5):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.window_size = window_size
        self.current_step = window_size
        self.total_shares = 0
        self.trades = []

    def get_state(self):
        start = self.current_step - self.window_size
        market_state = self.df[self.feature_cols].iloc[start:self.current_step].values.flatten()

        current_price = float(self.df['Close'].iloc[self.current_step])
        portfolio_value = self.balance + (self.total_shares * current_price)

        account_state = np.array([
            self.balance / self.initial_balance,
            self.total_shares / 100,
            portfolio_value / self.initial_balance,
            (portfolio_value - self.initial_balance) / self.initial_balance,
            len(self.trades) / 100,
        ])

        return np.concatenate((market_state, account_state))

    def execute_trade(self, action):
        current_price = float(self.df['Close'].iloc[self.current_step])

        if action == 1:  # BUY
            invest_amount = self.balance * 0.5
            shares_to_buy = int(invest_amount / current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.001  # 0.1% 交易成本
                if self.balance >= cost:
                    self.balance -= cost
                    self.total_shares += shares_to_buy
                    self.trades.append(('BUY', self.current_step, current_price, shares_to_buy))

        elif action == 2:  # SELL
            shares_to_sell = int(self.total_shares * 0.5)
            if shares_to_sell > 0:
                revenue = shares_to_sell * current_price * 0.999  # 0.1% 交易成本
                self.balance += revenue
                self.total_shares -= shares_to_sell
                self.trades.append(('SELL', self.current_step, current_price, shares_to_sell))

        self.current_step += 1

    def run(self, model):
        action_counts = {0: 0, 1: 0, 2: 0}
        portfolio_history = []

        while self.current_step < len(self.df) - 1:
            state = self.get_state()
            q_values = model.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])

            action_counts[action] += 1
            self.execute_trade(action)

            current_price = float(self.df['Close'].iloc[self.current_step])
            portfolio_value = self.balance + (self.total_shares * current_price)
            portfolio_history.append(portfolio_value)

        # 最終清算
        final_price = float(self.df['Close'].iloc[-1])
        self.balance += self.total_shares * final_price
        self.total_shares = 0

        return action_counts, portfolio_history

# =========================================================
# 5. 運行測試
# =========================================================
print("\n📊 開始測試模型...")
simulator = SimpleTradingSimulator(df_features, feature_cols)
action_counts, portfolio_history = simulator.run(model)

final_profit = simulator.balance - simulator.initial_balance
roi = (final_profit / simulator.initial_balance) * 100

# =========================================================
# 6. 顯示結果
# =========================================================
print("\n" + "="*60)
print("測試結果")
print("="*60)
print(f"初始資金:    ${simulator.initial_balance:,.2f}")
print(f"最終資金:    ${simulator.balance:,.2f}")
print(f"總利潤:      ${final_profit:,.2f}")
print_roi(f"投資回報率:  {roi:.2f}%")
print(f"執行交易:    {len(simulator.trades)} 筆")
print(f"動作分布:    Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")
print("="*60)

if final_profit >= 1280.40:
    print("✅ 成功！超過目標 $1,280.40")
else:
    print(f"⚠️  距離目標還差 ${1280.40 - final_profit:.2f}")

# =========================================================
# 7. 生成圖表
# =========================================================
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(portfolio_history, color='green', linewidth=2)
plt.axhline(y=simulator.initial_balance, color='red', linestyle='--', alpha=0.5)
plt.title(f'{ticker} - 模型測試表現', fontsize=14, fontweight='bold')
plt.ylabel('投資組合價值 ($)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
price_data = df_features['Close'].iloc[simulator.window_size:].reset_index(drop=True)
plt.plot(price_data, color='black', alpha=0.6, linewidth=1.5)

for trade in simulator.trades:
    action, step, price, shares = trade
    idx = step - simulator.window_size
    if 0 <= idx < len(price_data):
        if action == 'BUY':
            plt.scatter(idx, price, color='green', marker='^', s=100, zorder=5)
        elif action == 'SELL':
            plt.scatter(idx, price, color='red', marker='v', s=100, zorder=5)

plt.title('交易信號', fontsize=12)
plt.ylabel('股價 ($)')
plt.xlabel('交易日')
plt.grid(True, alpha=0.3)

plt.tight_layout()
chart_name = f'{ticker}_model_test_results.png'
plt.savefig(chart_name, dpi=150, bbox_inches='tight')
print(f"\n📊 圖表已保存: {chart_name}")

print("\n✅ 測試完成！")
