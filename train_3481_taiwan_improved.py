"""
改进版 台股 3481 交易 AI 训练
=====================================
基于最佳模型 (AAPL Improved PPO) 客制化
改进点:
1. ✅ 更多训练数据 (2015-2024, 9年数据)
2. ✅ 改进的奖励函数 (鼓励交易)
3. ✅ 更长训练时间 (100,000 步)
4. ✅ 连续动作空间 (更灵活的买卖比例)
"""
import os
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# ==========================================
# 1. 改进的交易环境
# ==========================================
class ImprovedTradingEnv(gym.Env):
    """
    改进的交易环境
    - 连续动作空间 (更灵活)
    - 改进的奖励函数 (鼓励交易)
    """
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        # 🔥 改进 4: 连续动作空间
        # action: -1.0 到 1.0
        # -1.0 = 全部卖出, 0 = 持有, 1.0 = 全部买入
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        # 观察空间 (增加了更多特征)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(15,),  # 从10增加到15个特征
            dtype=np.float32
        )
        self.reset()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0  # 追踪交易次数
        self.last_action = 0  # 上一次的动作
        return self._get_observation(), {}
    def _get_observation(self):
        """增强的观察空间"""
        row = self.df.iloc[self.current_step]
        # 计算当前持仓比例
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1
        obs = np.array([
            float(self.shares_held),          # 0. 持股数量
            float(self.balance),              # 1. 现金余额
            float(row['close']),              # 2. 当前价格
            float(row.get('sma_10', 0)),      # 3. SMA 10
            float(row.get('sma_30', 0)),      # 4. SMA 30
            float(row.get('sma_50', 0)),      # 5. SMA 50
            float(row.get('rsi', 50)),        # 6. RSI
            float(row.get('macd', 0)),        # 7. MACD
            float(row.get('macd_signal', 0)), # 8. MACD Signal
            float(row.get('bb_upper', 0)),    # 9. Bollinger Upper
            float(row.get('bb_lower', 0)),    # 10. Bollinger Lower
            float(row.get('volume', 0)),      # 11. 成交量
            float(self.total_profit),         # 12. 总收益
            float(stock_ratio),               # 13. 持股比例
            float(cash_ratio),                # 14. 现金比例
        ], dtype=np.float32)
        return obs
    def step(self, action):
        """
        执行动作 (连续动作空间)
        action: -1.0 到 1.0 的浮点数
        """
        # 确保 action 是标量
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)
        # 限制 action 范围
        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price
        # 🔥 改进: 连续动作执行
        if action < -0.1:  # 卖出 (action 在 -1.0 到 -0.1)
            # 卖出比例 = abs(action)
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:  # 买入 (action 在 0.1 到 1.0)
            # 买入比例 = action
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
        # else: -0.1 <= action <= 0.1 -> 持有,不做任何操作
        # 计算新的总价值和收益
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        # 🔥 改进 2: 改进的奖励函数
        # 基础奖励: 收益率
        profit_reward = self.total_profit / self.initial_balance
        # 交易激励: 鼓励适度交易
        trade_incentive = 0.0
        if abs(action) > 0.1:  # 如果进行了交易
            trade_incentive = 0.01  # 小额奖励
        # 惩罚过度持有现金
        cash_penalty = 0.0
        if self.balance > old_total_value * 0.9:  # 如果现金超过90%
            cash_penalty = -0.005
        # 综合奖励
        reward = profit_reward + trade_incentive + cash_penalty
        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()
        return obs, float(reward), done, False, {}
# ==========================================
# 2. 下载和处理数据
# ==========================================
def download_and_prepare_data(ticker='3481.TW', start_date='2015-01-01', end_date='2025-01-01'):
    """
    🔥 改进 1: 下载更多年份的数据
    台股代码格式: 3481.TW
    """
    print("=" * 70)
    print(f"下载 {ticker} 股票数据...")
    print(f"日期范围: {start_date} 至 {end_date}")
    print("=" * 70)
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        # 处理MultiIndex列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if df.empty:
            raise ValueError(f"无法下载 {ticker} 的数据")
        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()
        print(f"✅ 成功下载 {len(df)} 天的数据")
        print(f"   价格范围: NT${float(df['close'].min()):.2f} - NT${float(df['close'].max()):.2f}")
        return df
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None
def add_technical_indicators(df):
    """添加技术指标"""
    print("\n添加技术指标...")
    # SMA
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    # EMA
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
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)
    # KD指标 (Stochastic Oscillator)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()
    # 移动平均线
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()
    # 波动性指标
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    # 价格变化率
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    # MA50斜率
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100
    # 未来涨跌方向 (用于特征重要性分析)
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
    # 填充缺失值
    df = df.bfill().ffill()
    print(f"✅ 添加了多个技术指标")
    return df
# ==========================================
# 3. 特征重要性分析
# ==========================================
def analyze_feature_importance(df, ticker):
    """分析技术指标的重要性"""
    print("\n特徵重要性分析 (ML Model)")
    print("=" * 70)
    # 選擇特徵 (所有技術指標)
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
        'MA_20', 'MA_50', 'MA_200',
        'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
    ]
    # 排除 NaN 值
    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) == 0:
        print("數據不足，無法執行特徵重要性分析。請檢查NaN值是否過多。")
        return None
    X = ml_data[features]
    y = ml_data['future_direction']
    print(f"用於分析的數據點總數: {len(X)}")
    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    # 訓練隨機森林分類器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    # 獲取特徵重要性分數
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    # 預測準確度
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在測試集上的預測準確率: {accuracy:.4f}")
    print("特徵重要性計算完成")
    print("=" * 70)
    print("技術指標重要性排名")
    print("=" * 70)
    print(feature_importance_df.to_string(index=False))
    # 視覺化特徵重要性
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498DB')
    plt.xlabel("特徵重要性分數 (Feature Importance Score)", fontweight='bold')
    plt.ylabel("技術指標 (Technical Indicator)", fontweight='bold')
    plt.title(f"{ticker} 基於隨機森林模型的技術指標重要性", fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    filename = f'{ticker.replace(".", "_")}_feature_importance.png'
    plt.savefig(filename, dpi=300)
    print(f"特徵重要性圖表已儲存: {filename}")
    plt.close()

    # 保存为 JSON 文件供交易信号使用
    import json
    from datetime import datetime
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'feature_importance': {
            row['Feature']: float(row['Importance'])
            for _, row in feature_importance_df.iterrows()
        }
    }
    json_filename = f'{ticker.replace(".", "_")}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"特徵重要性數據已保存: {json_filename}")

    return feature_importance_df
def train_improved_model(df, ticker, total_timesteps=100000):
    """
    🔥 改进 3: 更长的训练时间
    """
    print("\n" + "=" * 70)
    print(f"开始训练改进版 {ticker} 交易模型")
    print("=" * 70)
    env = DummyVecEnv([lambda: ImprovedTradingEnv(df)])
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # 熵系数,鼓励探索
    )
    print(f"\n训练配置:")
    print(f"  总训练步数: {total_timesteps:,}")
    print(f"  训练数据点: {len(df)}")
    print(f"  学习率: 0.0003")
    print(f"  动作空间: 连续 [-1.0, 1.0]")
    print(f"  奖励机制: 收益 + 交易激励 + 现金惩罚")
    print("\n开始训练...")
    model.learn(total_timesteps=total_timesteps)
    model_path = f"ppo_{ticker.lower().replace('.', '_')}_improved"
    model.save(model_path)
    print(f"\n✅ 模型已保存: {model_path}.zip")
    return model
# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 改进版台股 3481 交易 AI 训练系统")
    print("=" * 70)
    # 配置
    TICKER = '3481.TW'  # 台股 3481
    START_DATE = '2015-01-01'  # 🔥 从2015年开始 (9年数据)
    END_DATE = '2025-07-31'
    TRAIN_TEST_SPLIT = 0.8
    TOTAL_TIMESTEPS = 100000  # 🔥 训练100,000步
    print(f"目标股票: {TICKER}")
    print(f"数据范围: {START_DATE} - {END_DATE}")
    print(f"训练步数: {TOTAL_TIMESTEPS:,}")
    print("=" * 70)
    # 1. 下载数据
    df = download_and_prepare_data(TICKER, START_DATE, END_DATE)
    if df is None:
        print("\n❌ 数据下载失败")
        exit(1)
    # 2. 添加技术指标
    df = add_technical_indicators(df)

    # 2.5 分析特征重要性
    analyze_feature_importance(df, TICKER)

    # 3. 分割数据
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)} 天")
    print(f"  测试集: {len(test_df)} 天")
    # 4. 训练模型
    model = train_improved_model(train_df, TICKER, total_timesteps=TOTAL_TIMESTEPS)
    print("\n✅ 训练完成!")
    print(f"模型文件: ppo_{TICKER.lower().replace('.', '_')}_improved.zip")
    print("\n改进点总结:")
    print("  ✅ 使用 9 年数据 (2015-2024)")
    print("  ✅ 连续动作空间 (更灵活)")
    print("  ✅ 改进的奖励函数 (鼓励交易)")
    print("  ✅ 训练 100,000 步 (更充分)")
    print("  ✅ 特征重要性分析 (优化指标)")
