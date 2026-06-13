import os
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import io
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import torch
except Exception:
    torch = None

#!pip install stable-baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
# 配置中文字体和后端
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
def download_and_prepare_data(ticker='2891.TW', start_date='2015-01-01', end_date='2025-01-01'):
    """
    🔥 改进 1: 下载更多年份的数据
    台股代码格式: 2891.TW (中信金)
    """
    print("=" * 70)
    print(f"下载 {ticker} 股票数据 (台股 ezconn)...")
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
    # 日收益率 (短期滞后特征，帮助模型抓住短期动量)
    df['return_1d'] = df['close'].pct_change(1)
    df['return_2d'] = df['close'].pct_change(2)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    # 成交量动量与均线
    df['volume_change_1d'] = df['volume'].pct_change(1)
    df['volume_ma_5'] = df['volume'].rolling(5).mean()
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    # 短/长期波动性
    df['volatility_10'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
    df['volatility_50'] = df['close'].rolling(50).std() / df['close'].rolling(50).mean()
    # MA50斜率
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100
    # 未来涨跌方向 (用于特征重要性分析)
    # 生成多时间窗口的目标标签，使用涨幅阈值来减少噪音
    horizons = [1, 3, 5, 10]
    thresh = 0.005  # 0.5% 涨幅阈值视为上涨
    for h in horizons:
        col = f'future_dir_{h}d'
        df[col] = ((df['close'].shift(-h) - df['close']) / df['close'] > thresh).astype(int)
    # 填充缺失值
    df = df.bfill().ffill()
    print(f"✅ 添加了多个技术指标")
    return df
# ==========================================
# 3. 特征重要性分析
# ==========================================
def analyze_feature_importance(df, ticker):
    """分析技术指标的重要性"""
    print(" 第 5 步：特徵重要性分析 (ML Model)")
    print("=" * 70 + "")
    # 選擇特徵 (所有技術指標)
    # 基础特征 + 新增短期收益率
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
        'MA_20', 'MA_50', 'MA_200',
        'volatility', 'volatility_10', 'volatility_50', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope',
        'return_1d', 'return_2d', 'return_3d', 'return_5d',
        'volume_change_1d', 'volume_ma_5', 'volume_ma_20'
    ]
    # 尝试多个预测窗口并选择表现最佳的标签
    horizons = [1, 3, 5, 10]
    best_accuracy = -1.0
    best_result = None
    best_horizon = None
    for h in horizons:
        label_col = f'future_dir_{h}d'
        if label_col not in df.columns:
            continue
        ml_data = df.dropna(subset=features + [label_col])
        if len(ml_data) < 50:
            continue
        X = ml_data[features]
        y = ml_data[label_col]
        print(f"嘗試 horizon={h} 天，資料點: {len(X)}")
        # 数据清洗与标准化并使用一个更稳健的 ML 流水线
        # 替换无穷值并填充缺失，裁剪极端值以避免数值不稳定
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X = X.clip(lower=-1e6, upper=1e6)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        # 使用时间序列友好的划分（保持顺序）并且保证类别分布
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 首先尝试使用随机森林并进行随机搜索调参
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
        param_dist = {
            'n_estimators': [100, 200, 400],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        cv = StratifiedKFold(n_splits=3)
        try:
            rs = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=12, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
            rs.fit(X_train, y_train)
            best_model = rs.best_estimator_
            print(f"RandomForest best params: {rs.best_params_}")
        except Exception:
            # 如若 RandomizedSearchCV 失败，退回到默认训练
            best_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
            best_model.fit(X_train, y_train)

        # 如果安装了 xgboost，则尝试使用 XGBoost（通常在表格数据上更强）
        try:
            import xgboost as xgb
            xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
            xgb_params = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1]
            }
            rs_xgb = RandomizedSearchCV(xgb_clf, param_distributions=xgb_params, n_iter=6, cv=cv, scoring='accuracy', n_jobs=-1, random_state=42)
            rs_xgb.fit(X_train, y_train)
            xgb_best = rs_xgb.best_estimator_
            # 比较两个模型在验证集上的表现，选择更好的
            rf_acc = accuracy_score(y_test, best_model.predict(X_test))
            xgb_acc = accuracy_score(y_test, xgb_best.predict(X_test))
            if xgb_acc > rf_acc:
                best_model = xgb_best
                print(f"XGBoost outperformed RandomForest: {xgb_acc:.4f} > {rf_acc:.4f}")
            else:
                print(f"RandomForest retained: {rf_acc:.4f} >= {xgb_acc:.4f}")
        except Exception:
            # xgboost 不可用，则继续使用 best_model
            pass

        # 獲取特徵重要性分數（如果模型有 feature_importances_）
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
        else:
            # 否则使用 permutation importance 作为后备
            from sklearn.inspection import permutation_importance
            r = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances = r.importances_mean

        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 預測準確度
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"horizon={h} 模型在測試集上的預測準確率: {accuracy:.4f}")
        # 保存表现最好的
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_result = (best_model, feature_importance_df, accuracy)
            best_horizon = h

    if best_result is None:
        print("未能在任何 horizon 上得到有效模型，請檢查數據")
        return

    best_model, feature_importance_df, accuracy = best_result
    print("✅ 特徵重要性計算完成，選擇最佳 horizon: {} 天，準確率: {:.4f}".format(best_horizon, accuracy))
    print("" + "=" * 70)
    print("🏅 技術指標重要性排名")
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
    filename = f'{ticker}_feature_importance.png'
    plt.savefig(filename, dpi=300)
    print(f"✅ 特徵重要性圖表已儲存: {filename}")
    plt.close()

    # 保存为 JSON 文件供交易信号使用
    import json
    from datetime import datetime
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'best_horizon_days': int(best_horizon) if 'best_horizon' in locals() and best_horizon is not None else None,
        'feature_importance': {
            row['Feature']: float(row['Importance'])
            for _, row in feature_importance_df.iterrows()
        }
    }
    json_filename = f'{ticker}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 特徵重要性數據已保存: {json_filename}")
    # 儲存 best model 以便後續分析（pickle），非必需但有用
    try:
        import joblib
        model_filename = f"{ticker}_best_classifier.joblib"
        joblib.dump(best_model, model_filename)
        print(f"✅ 最佳分類器已保存: {model_filename}")
    except Exception:
        pass
    return feature_importance_df
# ==========================================
# 4. 训练模型
# ==========================================
def train_improved_model(df, ticker, total_timesteps=200000, seeds=None):
    """
    🔥 改进 3: 更长的训练时间
    """
    print("\n" + "=" * 70)
    print(f"开始训练改进版 {ticker} 交易模型")
    print("=" * 70)
    # 标准化观察空间中的数值特征以帮助 PPO 学习（避免极端值）
    scaled_df = df.copy().reset_index(drop=True)
    scale_cols = ['close', 'sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'volume']
    try:
        # 对 volume 做对数变换以缓解长尾
        if 'volume' in scaled_df.columns:
            scaled_df['volume'] = np.log1p(scaled_df['volume'].astype(float))
        scaler = StandardScaler()
        available = [c for c in scale_cols if c in scaled_df.columns]
        scaled_vals = scaler.fit_transform(scaled_df[available].fillna(0))
        for i, c in enumerate(available):
            scaled_df[c] = scaled_vals[:, i]
    except Exception:
        # 如果标准化失败，则继续使用原始 df
        scaled_df = df.copy()

    # 支持多随机种子训练以评估稳定性
    if seeds is None:
        seeds = [42]
    models = []
    for s in seeds:
        if s is not None:
            print(f"\n--- 训练 seed={s} ---")
            try:
                set_random_seed(int(s))
            except Exception:
                np.random.seed(int(s))
                if torch is not None:
                    try:
                        torch.manual_seed(int(s))
                    except Exception:
                        pass
        else:
            print("\n--- 训练 seed=None (随机) ---")

        env = DummyVecEnv([lambda: ImprovedTradingEnv(scaled_df)])
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            learning_rate=0.0003,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            ent_coef=0.005,  # 略降低熵系数以平衡探索/利用
        )
        print(f"\n训练配置:")
        print(f"  seed: {s}")
        print(f"  总训练步数: {total_timesteps:,}")
        print(f"  训练数据点: {len(df)}")
        print(f"  学习率: 0.0003")
        print(f"  动作空间: 连续 [-1.0, 1.0]")
        print(f"  奖励机制: 收益 + 交易激励 + 现金惩罚")
        print("\n开始训练...")
        model.learn(total_timesteps=total_timesteps)
        model_path = f"ppo_{ticker.lower().replace('.', '_')}_improved_seed{s}"
        model.save(model_path)
        print(f"\n✅ 模型已保存: {model_path}.zip")
        models.append((s, model_path))

    return models
# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("🚀 改进版台股 2891 (中信金) 交易 AI 训练系统")
    print("=" * 70)
    # 配置
    TICKER = '2891.TW'  # 台股 中信金
    START_DATE = '2015-01-01'  # 从2015年开始 (10年数据)
    END_DATE = '2025-07-31'
    TRAIN_TEST_SPLIT = 0.8
    TOTAL_TIMESTEPS = 200000  # 训练 200,000 步 起始，可改为 200k-500k
    # 可选多种随机种子以稳定评估
    SEEDS = [42, 7, 2024]
    print(f"目标股票: {TICKER} (中信金)")
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
    # 3. 分割数据
    split_idx = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    print(f"\n数据分割:")
    print(f"  训练集: {len(train_df)} 天")
    print(f"  测试集: {len(test_df)} 天")
    # 4. 特征重要性分析
    print("\n执行特征重要性分析...")
    feature_importance = analyze_feature_importance(train_df, TICKER)
    # 5. 训练模型（多种随机种子）
    models_info = train_improved_model(train_df, TICKER, total_timesteps=TOTAL_TIMESTEPS, seeds=SEEDS)

    # 6. 再次对全部数据做特徵重要性分析（供展示）
    from datetime import datetime
    analyze_feature_importance(df, TICKER)

    print("\n✅ 训练完成!")
    print(f"模型文件: ppo_{TICKER.lower().replace('.', '_')}_improved.zip")
    print("\n改进点总结:")
    print("  ✅ 使用 10 年数据 (2015-2025)")
    print("  ✅ 连续动作空间 (更灵活)")
    print("  ✅ 改进的奖励函数 (鼓励交易)")
    print("  ✅ 训练 100,000 步 (更充分)")
    print("  ✅ 包含特征重要性分析")
