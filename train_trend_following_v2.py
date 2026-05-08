"""
趨勢跟蹤 AI 交易模型 V2
========================
核心改進：
1. 不使用 RSI 數值作為賣出依據
2. 使用趨勢破壞信號來決定賣出
3. 獎勵函數鼓勵順勢操作

關鍵指標：
- MA 趨勢 (MA10 > MA30 > MA50 = 多頭)
- MACD 趨勢 (MACD > Signal = 多頭)
- 價格位置 (Price > MA50 = 多頭)
- 成交量趨勢 (放量上漲 = 強勢)
"""
import os
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
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==========================================
# 趨勢跟蹤交易環境
# ==========================================
class TrendFollowingEnv(gym.Env):
    """
    趨勢跟蹤交易環境
    核心理念：順勢操作，不逆勢賣出

    觀察空間 (18維)：
    - 持倉資訊 (3): 持股、現金、持股比例
    - 價格資訊 (1): 當前價格
    - 趨勢指標 (6): MA10/30/50, 價格vs各MA
    - 動量指標 (4): MACD, MACD_signal, MACD_hist, MACD_trend
    - 成交量 (2): 量比, 量價配合
    - 波動性 (2): ATR, 布林帶位置

    *** 不使用 RSI 數值 ***
    """

    def __init__(self, df, initial_balance=10000):
        super(TrendFollowingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        # 連續動作空間: -1.0(全賣) 到 1.0(全買)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # 觀察空間: 18 維特徵 (不含 RSI)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,),
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
        self.consecutive_hold = 0  # 連續持有天數
        return self._get_observation(), {}

    def _get_observation(self):
        """趨勢跟蹤觀察空間 - 不使用 RSI"""
        row = self.df.iloc[self.current_step]

        # 計算持倉比例
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0

        # 趨勢判斷
        ma10 = float(row.get('sma_10', current_price))
        ma30 = float(row.get('sma_30', current_price))
        ma50 = float(row.get('sma_50', current_price))

        # 價格相對於均線的位置 (標準化)
        price_vs_ma10 = (current_price - ma10) / ma10 * 100 if ma10 > 0 else 0
        price_vs_ma30 = (current_price - ma30) / ma30 * 100 if ma30 > 0 else 0
        price_vs_ma50 = (current_price - ma50) / ma50 * 100 if ma50 > 0 else 0

        # 均線排列 (1=多頭, -1=空頭)
        ma_trend = 1.0 if ma10 > ma30 > ma50 else (-1.0 if ma10 < ma30 < ma50 else 0.0)

        # MACD 趨勢
        macd = float(row.get('macd', 0))
        macd_signal = float(row.get('macd_signal', 0))
        macd_hist = float(row.get('macd_hist', 0))
        macd_trend = 1.0 if macd > macd_signal else -1.0

        # 成交量分析
        volume_ratio = float(row.get('volume_ratio', 1.0))
        # 量價配合: 上漲放量=1, 下跌放量=-1, 其他=0
        price_change = float(row.get('price_change_1d', 0))
        volume_price_sync = 1.0 if (price_change > 0 and volume_ratio > 1.2) else (
            -1.0 if (price_change < 0 and volume_ratio > 1.2) else 0.0
        )

        # 波動性
        atr_pct = float(row.get('atr_pct', 2.0))
        bb_position = float(row.get('bb_position', 50))

        # MA50 斜率 (趨勢方向)
        ma50_slope = float(row.get('ma50_slope', 0))

        obs = np.array([
            # 持倉資訊 (3)
            float(self.shares_held) / 1000,  # 標準化
            float(self.balance) / self.initial_balance,
            float(stock_ratio),

            # 價格資訊 (1)
            float(current_price) / 1000,  # 標準化

            # 趨勢指標 (6)
            float(price_vs_ma10),
            float(price_vs_ma30),
            float(price_vs_ma50),
            float(ma_trend),
            float(ma50_slope),
            float(row.get('price_momentum', 0)),  # 5日動量

            # MACD 動量 (4)
            float(macd) * 10,  # 放大便於學習
            float(macd_signal) * 10,
            float(macd_hist) * 10,
            float(macd_trend),

            # 成交量 (2)
            float(volume_ratio),
            float(volume_price_sync),

            # 波動性 (2)
            float(atr_pct),
            float(bb_position) / 100,  # 標準化到 0-1

        ], dtype=np.float32)

        return obs

    def step(self, action):
        """
        趨勢跟蹤策略執行
        核心: 順勢操作，不逆勢賣出
        """
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)

        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        old_total_value = self.balance + self.shares_held * current_price

        # 獲取趨勢狀態
        ma10 = float(row.get('sma_10', current_price))
        ma30 = float(row.get('sma_30', current_price))
        ma50 = float(row.get('sma_50', current_price))
        macd = float(row.get('macd', 0))
        macd_signal = float(row.get('macd_signal', 0))

        # 判斷是否為強勢趨勢
        is_uptrend = (ma10 > ma30) and (macd > macd_signal) and (current_price > ma50)
        is_downtrend = (ma10 < ma30) and (macd < macd_signal) and (current_price < ma50)

        # ==========================================
        # 趨勢跟蹤交易邏輯
        # ==========================================

        if action < -0.1:  # 賣出意圖
            if is_uptrend:
                # 🔥 核心改進: 上升趨勢中不賣出！
                action = 0  # 強制持有
                self.consecutive_hold += 1
            else:
                # 下降趨勢或震盪，允許賣出
                sell_ratio = abs(action)
                shares_to_sell = int(self.shares_held * sell_ratio)
                if shares_to_sell > 0:
                    self.balance += shares_to_sell * current_price
                    self.shares_held -= shares_to_sell
                    self.total_trades += 1
                    self.consecutive_hold = 0

        elif action > 0.1:  # 買入意圖
            if is_downtrend:
                # 下降趨勢中減少買入
                action = action * 0.3  # 買入量減少 70%

            buy_ratio = abs(action)
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1
                self.consecutive_hold = 0
        else:
            self.consecutive_hold += 1

        # 計算新價值
        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance

        # ==========================================
        # 趨勢跟蹤獎勵函數
        # ==========================================

        # 基礎收益獎勵
        profit_reward = self.total_profit / self.initial_balance

        # 趨勢順應獎勵
        trend_reward = 0.0
        if is_uptrend and self.shares_held > 0:
            # 上升趨勢持有股票 = 獎勵
            trend_reward = 0.02
        elif is_downtrend and self.shares_held == 0:
            # 下降趨勢空倉 = 獎勵
            trend_reward = 0.01
        elif is_uptrend and self.shares_held == 0:
            # 上升趨勢空倉 = 懲罰 (錯過行情)
            trend_reward = -0.02
        elif is_downtrend and self.shares_held > 0:
            # 下降趨勢持股 = 懲罰
            trend_reward = -0.01

        # 長期持有獎勵 (鼓勵不頻繁交易)
        hold_bonus = min(self.consecutive_hold * 0.001, 0.01)

        # 綜合獎勵
        reward = profit_reward + trend_reward + hold_bonus

        # 移動到下一步
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, float(reward), done, False, {}


# ==========================================
# 數據處理
# ==========================================
def download_and_prepare_data(ticker, start_date, end_date):
    """下載股票數據"""
    print("=" * 70)
    print(f"下載 {ticker} 股票數據...")
    print(f"日期範圍: {start_date} 至 {end_date}")
    print("=" * 70)

    try:
        import yfinance as yf
        yf.set_tz_cache_location(r"C:\Users\Silvi\Projects\trading-bot\TMP")

        df = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if df.empty:
            print("指定區間無數據，使用全量數據...")
            df = yf.download(ticker, period="max", progress=False)
            if df.empty:
                raise ValueError(f"無法下載 {ticker} 的數據")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()

        print(f"成功下載 {len(df)} 天的數據")
        return df

    except Exception as e:
        print(f"下載失敗: {e}")
        return None


def add_trend_indicators(df):
    """添加趨勢跟蹤指標 - 不使用 RSI 作為主要信號"""
    print("\n添加趨勢跟蹤指標...")

    # 均線
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()

    # EMA
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # MACD (主要動量指標)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # 布林帶
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)

    # 成交量分析
    df['volume_ma20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # 價格動量
    df['price_change_1d'] = df['close'].pct_change(1) * 100
    df['price_momentum'] = df['close'].pct_change(5) * 100

    # MA50 斜率 (趨勢方向)
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    # ATR (波動性)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 填充缺失值
    df = df.fillna(method='bfill').fillna(method='ffill')

    print(f"添加了趨勢跟蹤指標 (不含 RSI)")
    return df


# ==========================================
# 訓練模型
# ==========================================
def train_trend_model(df, ticker, total_timesteps=100000):
    """訓練趨勢跟蹤模型"""
    print("\n" + "=" * 70)
    print("訓練趨勢跟蹤 PPO 模型")
    print("=" * 70)

    # 創建環境
    env = DummyVecEnv([lambda: TrendFollowingEnv(df)])

    # PPO 參數 (趨勢跟蹤優化)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,        # 高 gamma = 重視長期收益
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,     # 低熵 = 更確定的策略
        verbose=1,
        tensorboard_log=f"./tensorboard_logs/{ticker}_trend/"
    )

    # 訓練
    print(f"\n開始訓練 {total_timesteps:,} 步...")
    model.learn(total_timesteps=total_timesteps)

    # 保存
    model_path = f"ppo_{ticker.lower().replace('.', '_')}_trend_v2"
    model.save(model_path)
    print(f"\n模型已保存: {model_path}.zip")

    return model


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    print("=" * 70)
    print("趨勢跟蹤 AI 交易模型 V2")
    print("核心改進: 不依賴 RSI 數值，使用趨勢破壞信號")
    print("=" * 70)

    # 配置
    TICKER = '2330.TW'  # 可更改為其他股票
    START_DATE = '2015-01-01'
    END_DATE = '2026-12-31'
    TOTAL_TIMESTEPS = 100000

    print(f"\n目標股票: {TICKER}")
    print(f"數據範圍: {START_DATE} - {END_DATE}")
    print(f"訓練步數: {TOTAL_TIMESTEPS:,}")

    # 1. 下載數據
    df = download_and_prepare_data(TICKER, START_DATE, END_DATE)
    if df is None:
        print("\n數據下載失敗")
        exit(1)

    # 2. 添加指標
    df = add_trend_indicators(df)

    # 3. 訓練
    split_idx = int(len(df) * 0.8)
    train_df = df[:split_idx].copy()

    print(f"\n訓練集: {len(train_df)} 天")

    model = train_trend_model(train_df, TICKER, total_timesteps=TOTAL_TIMESTEPS)

    print("\n" + "=" * 70)
    print("訓練完成!")
    print("=" * 70)
    print("\n核心改進:")
    print("  1. 不使用 RSI 數值作為賣出依據")
    print("  2. 上升趨勢中強制持有 (不逆勢賣出)")
    print("  3. 獎勵函數鼓勵順勢操作")
    print("  4. 使用 MACD + MA 趨勢判斷")
