"""
高信心股票 AI 交易模型训练系统
=====================================
基于 train_2317_taiwan_improved.py 架构

优先训练股票（按信心度排序）:
1. ✅ 8499.TW 鼎炫-KY - 测试准确率 74.4%, 过拟合差距低
2. ✅ 2408.TW 南亚科 - 测试准确率 65.1%, 过拟合差距中低
3. ✅ 2368.TW 金像电 - 测试准确率 55.8%, 过拟合差距中等
4. ✅ 2383.TW 台光电 - 测试准确率 55.8%, 过拟合差距中等
5. ✅ 4722.TW 国精化 - 测试准确率 55.8%, 过拟合差距中等

改进点:
1. ✅ 10年历史数据 (2015-2025)
2. ✅ 连续动作空间 (灵活买卖)
3. ✅ 改进奖励函数 (鼓励交易)
4. ✅ 长时间训练 (150,000步)
5. ✅ 特征重要性分析
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
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
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
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

        # 连续动作空间: -1.0 到 1.0
        # -1.0 = 全部卖出, 0 = 持有, 1.0 = 全部买入
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 观察空间 (15个特征)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(15,),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        self.last_action = 0
        return self._get_observation(), {}

    def _get_observation(self):
        """增强的观察空间"""
        row = self.df.iloc[self.current_step]
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
        """执行动作 (连续动作空间)"""
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])
        old_total_value = self.balance + self.shares_held * current_price

        # 连续动作执行
        if action < -0.1:  # 卖出
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:  # 买入
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1

        # 计算新的总价值和收益
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance

        # 改进的奖励函数
        profit_reward = self.total_profit / self.initial_balance
        trade_incentive = 0.01 if abs(action) > 0.1 else 0.0
        cash_penalty = -0.005 if self.balance > old_total_value * 0.9 else 0.0
        reward = profit_reward + trade_incentive + cash_penalty

        # 移动到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()
        return obs, float(reward), done, False, {}

# ==========================================
# 2. 下载和处理数据
# ==========================================
def download_and_prepare_data(ticker, start_date='2015-01-01', end_date='2025-01-12'):
    """下载股票数据"""
    print("=" * 70)
    print(f"下载 {ticker} 股票数据...")
    print(f"日期范围: {start_date} 至 {end_date}")
    print("=" * 70)

    try:
        import yfinance as yf
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

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
        print(f"   价格范围: ${float(df['close'].min()):.2f} - ${float(df['close'].max()):.2f}")
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

    # KD指标
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()

    # 移动平均线
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()

    # 波动性指标
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 价格变化率
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_20d'] = df['close'].pct_change(20).mean() * 100

    # MA50斜率
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100

    # 未来涨跌方向
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
    print("\n特征重要性分析 (ML Model)")
    print("=" * 70)

    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
        'MA_20', 'MA_50', 'MA_200',
        'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
    ]

    ml_data = df.dropna(subset=features + ['future_direction'])
    if len(ml_data) == 0:
        print("数据不足，无法执行特征重要性分析")
        return None

    X = ml_data[features]
    y = ml_data['future_direction']
    print(f"用于分析的数据点总数: {len(X)}")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    # 训练随机森林分类器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # 获取特征重要性分数
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 预测准确度
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在测试集上的预测准确率: {accuracy:.4f}")
    print("=" * 70)
    print("技术指标重要性排名")
    print("=" * 70)
    print(feature_importance_df.to_string(index=False))

    # 可视化特征重要性
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498DB')
    plt.xlabel("Feature Importance Score", fontweight='bold')
    plt.ylabel("Technical Indicator", fontweight='bold')
    plt.title(f"{ticker} Feature Importance (Random Forest)", fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    filename = f'{ticker.replace(".", "_")}_feature_importance.png'
    plt.savefig(filename, dpi=300)
    print(f"特征重要性图表已保存: {filename}")
    plt.close()

    # 保存为 JSON 文件
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
    print(f"特征重要性数据已保存: {json_filename}")

    return feature_importance_df

def train_improved_model(df, ticker, total_timesteps=150000):
    """训练改进版模型"""
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
        ent_coef=0.01,
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
    print("🚀 高信心股票 AI 交易模型训练系统")
    print("=" * 70)

    # 高信心股票列表（按准确率排序）
    HIGH_CONFIDENCE_STOCKS = [
        {'ticker': '8499.TW', 'name': '鼎炫-KY', 'accuracy': 0.744, 'confidence': '最高'},
        {'ticker': '2408.TW', 'name': '南亚科', 'accuracy': 0.651, 'confidence': '高'},
        {'ticker': '2368.TW', 'name': '金像电', 'accuracy': 0.558, 'confidence': '中高'},
        {'ticker': '2383.TW', 'name': '台光电', 'accuracy': 0.558, 'confidence': '中高'},
        {'ticker': '4722.TW', 'name': '国精化', 'accuracy': 0.558, 'confidence': '中高'},
    ]

    # 配置
    START_DATE = '2015-01-01'
    END_DATE = '2025-01-12'
    TRAIN_TEST_SPLIT = 0.8
    TOTAL_TIMESTEPS = 150000

    print(f"数据范围: {START_DATE} - {END_DATE}")
    print(f"训练步数: {TOTAL_TIMESTEPS:,}")
    print("=" * 70)

    # 逐个训练高信心股票
    for stock_info in HIGH_CONFIDENCE_STOCKS:
        ticker = stock_info['ticker']
        name = stock_info['name']
        accuracy = stock_info['accuracy']
        confidence = stock_info['confidence']

        print(f"\n{'='*70}")
        print(f"🎯 训练目标: {ticker} ({name})")
        print(f"   预测准确率: {accuracy:.1%}")
        print(f"   信心度: {confidence}")
        print(f"{'='*70}")

        # 1. 下载数据
        df = download_and_prepare_data(ticker, START_DATE, END_DATE)
        if df is None:
            print(f"\n❌ {ticker} 数据下载失败，跳过")
            continue

        # 2. 添加技术指标
        df = add_technical_indicators(df)

        # 3. 分析特征重要性
        analyze_feature_importance(df, ticker)

        # 4. 分割数据
        split_idx = int(len(df) * TRAIN_TEST_SPLIT)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        print(f"\n数据分割:")
        print(f"  训练集: {len(train_df)} 天")
        print(f"  测试集: {len(test_df)} 天")

        # 5. 训练模型
        model = train_improved_model(train_df, ticker, total_timesteps=TOTAL_TIMESTEPS)

        print(f"\n✅ {ticker} ({name}) 训练完成!")
        print(f"   模型文件: ppo_{ticker.lower().replace('.', '_')}_improved.zip")
        print(f"   特征重要性: {ticker.replace('.', '_')}_feature_importance.json")

    print("\n" + "=" * 70)
    print("🎉 所有高信心股票训练完成!")
    print("=" * 70)
    print("\n训练总结:")
    for stock_info in HIGH_CONFIDENCE_STOCKS:
        ticker = stock_info['ticker']
        name = stock_info['name']
        print(f"  ✅ {ticker} ({name}) - 模型文件: ppo_{ticker.lower().replace('.', '_')}_improved.zip")

    print("\n改进点:")
    print("  ✅ 使用 10 年数据 (2015-2025)")
    print("  ✅ 连续动作空间 (更灵活)")
    print("  ✅ 改进的奖励函数 (鼓励交易)")
    print("  ✅ 训练 150,000 步 (更充分)")
    print("  ✅ 特征重要性分析 (优化指标)")
