"""
台股 2331 (精英) AI 交易訊號生成器 (含 VecNormalize 標準化修正)
======================================
使用訓練好的 PPO 模型生成今日交易策略
輸出: 買入/賣出/持有 信號 + 建議價格
"""

import os
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'  # Fix Tcl/Tk error on Windows

import sys
import io
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from datetime import datetime
import warnings
import yfinance as yf

# 修正編碼問題
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# 嘗試導入自定義模組 (使用 try-except 避免完全失敗)
try:
    from dynamic_signal_weights import DynamicWeightCalculator
except ImportError:
    print("⚠️  警告: 無法載入 dynamic_signal_weights，使用預設權重")
    # 提供一個簡單的替代方案
    class DynamicWeightCalculator:
        def __init__(self, symbol):
            self.symbol = symbol
        
        def get_buy_weights(self):
            # 預設權重
            return {
                'rsi_oversold': 15,
                'macd_cross': 20,
                'ma_bullish': 15,
                'bb_lower': 10,
                'volume_surge': 12,
                'ai_strong': 18,
                'target_above': 10
            }
        
        def get_sell_weights(self):
            # 預設權重
            return {
                'rsi_overbought': 15,
                'macd_bearish': 20,
                'ma_bearish': 15,
                'bb_upper': 10,
                'target_below': 20,
                'target_near': 10,
                'volume_divergence': 10
            }

try:
    from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output
except ImportError:
    print("⚠️  警告: 無法載入 finbert_enhanced_scoring，使用簡化評分")
    
    def calculate_enhanced_buy_score_with_sentiment(rsi, macd, macd_signal, sma_10, sma_30, current_price, bb_upper, bb_lower, volume_ratio, ai_action, buy_weights, symbol):
        # 簡化評分邏輯
        score = 50  # 基礎分
        
        # RSI 評分
        if rsi < 30:
            score += buy_weights.get('rsi_oversold', 15)
        elif rsi < 40:
            score += buy_weights.get('rsi_oversold', 15) * 0.5
        
        # MACD 評分
        if macd > macd_signal:
            score += buy_weights.get('macd_cross', 20)
        
        # 均線評分
        if sma_10 > sma_30:
            score += buy_weights.get('ma_bullish', 15)
        
        # AI 動作評分
        if ai_action > 0.3:
            score += buy_weights.get('ai_strong', 18)
        elif ai_action > 0.1:
            score += buy_weights.get('ai_strong', 18) * 0.5
        
        # 量能評分
        if volume_ratio > 1.2:
            score += buy_weights.get('volume_surge', 12)
        
        # 布林帶評分
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5
        if bb_position < 0.3:
            score += buy_weights.get('bb_lower', 10)
        
        # 限制分數範圍
        score = max(0, min(100, score))
        
        return score, None, ["簡化評分系統"], [], {'score': score}, {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}

    def format_sentiment_output(sentiment_result):
        return "簡化情緒分析系統\n   情緒評分: 0.0 (中性)"

try:
    from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
except ImportError:
    print("⚠️  警告: 無法載入 candlestick_patterns")
    
    def analyze_candlestick_patterns(df, days=5):
        return {'patterns': []}
    
    def format_pattern_output(patterns):
        return "   蠟燭圖型態分析不可用"
    
    def get_pattern_score_adjustment(patterns):
        return 0

try:
    from ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment
except ImportError:
    print("⚠️  警告: 無法載入 ma50_slope_analysis")
    
    def calculate_ma50_slope(prices, window=50, slope_period=5):
        # 簡化實現
        if len(prices) < window:
            return {'ma50_current': prices.iloc[-1], 'slope': 0, 'slope_pct': 0, 'trend': '未知', 'color': '灰色', 'signal': '無', 'description': '數據不足'}
        
        ma50 = prices.rolling(window=window).mean()
        current_ma50 = ma50.iloc[-1]
        prev_ma50 = ma50.iloc[-slope_period]
        slope = (current_ma50 - prev_ma50) / slope_period
        slope_pct = (slope / prev_ma50) * 100 if prev_ma50 != 0 else 0
        
        if slope_pct > 0.1:
            trend = "上升"
            color = "綠色"
            signal = "偏多"
        elif slope_pct < -0.1:
            trend = "下降"
            color = "紅色"
            signal = "偏空"
        else:
            trend = "盤整"
            color = "灰色"
            signal = "中性"
        
        return {
            'ma50_current': current_ma50,
            'slope': slope,
            'slope_pct': slope_pct,
            'trend': trend,
            'color': color,
            'signal': signal,
            'description': f'MA50 {trend}趨勢，斜率{slope_pct:+.4f}%'
        }
    
    def format_ma50_slope_output(slope_info):
        return f"   當前MA50: {slope_info['ma50_current']:.2f}\n   MA50斜率: {slope_info['slope']:+.6f}\n   斜率百分比: {slope_info['slope_pct']:+.4f}%\n   趨勢判斷: {slope_info['color']} {slope_info['trend']}\n   交易信號: {slope_info['signal']}\n   💡 說明: {slope_info['description']}"
    
    def get_ma50_slope_score_adjustment(slope_info):
        slope_pct = slope_info.get('slope_pct', 0)
        if slope_pct > 0.2:
            return 5
        elif slope_pct > 0.1:
            return 3
        elif slope_pct < -0.2:
            return -5
        elif slope_pct < -0.1:
            return -3
        return 0

try:
    from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display
except ImportError:
    print("⚠️  警告: 無法載入 model_accuracy_tracker")
    
    def get_model_accuracy_display(symbol):
        return "準確度追蹤不可用"

try:
    from structure_pattern_analysis import detect_structure_patterns, format_structure_pattern_output, get_structure_score_adjustment
except ImportError:
    print("⚠️  警告: 無法載入 structure_pattern_analysis")
    
    def detect_structure_patterns(df, window=60):
        return {'patterns': []}
    
    def format_structure_pattern_output(patterns):
        return "   結構型態分析不可用"
    
    def get_structure_score_adjustment(patterns):
        return 0

# ==========================================
# 交易環境 (必須與訓練時完全一致)
# 注意: 根據錯誤信息，模型訓練時觀察空間為 (10, 8)
# ==========================================
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0
        
        # 關鍵修復: 根據錯誤信息，模型期望的觀察空間是 (10, 8)
        # 這意味著是10個時間步，每個時間步8個特徵
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 修改為二維觀察空間 (10個時間步，8個特徵)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10, 8),  # 修正為訓練時的形狀
            dtype=np.float32
        )
        
        # 確保有足夠的歷史數據
        self.history_window = 10
        self.reset()
    
    def _get_features(self, row):
        """提取8個特徵，與 Colab 訓練時完全一致"""
        features = np.array([
            float(row.get('log_ret', 0)),        # 對數報酬率
            float(row.get('sma_10', 0)),         # SMA10 比率
            float(row.get('sma_30', 0)),         # SMA30 比率
            float(row.get('rsi', 0.5)),          # RSI (0-1)
            float(row.get('macd', 0)),           # MACD 標準化
            float(row.get('bb_position', 0.5)),  # 布林帶位置 (0-1)
            float(row.get('vol_change', 0)),     # 成交量變化
            float(row.get('volatility', 0))      # 波動率
        ], dtype=np.float32)
        return features
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_window - 1  # 確保有足夠的歷史數據
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.total_trades = 0
        return self._get_observation(), {}
    
    def _get_observation(self):
        """返回最近10個時間步的8個特徵"""
        obs_matrix = []
        
        # 獲取從 current_step-9 到 current_step 的10個時間步
        for i in range(self.history_window):
            idx = max(0, min(self.current_step - (self.history_window - 1) + i, len(self.df) - 1))
            row = self.df.iloc[idx]
            features = self._get_features(row)
            obs_matrix.append(features)
        
        obs = np.array(obs_matrix, dtype=np.float32)
        return obs
    
    def step(self, action):
        # 預測模式下通常不執行完整的 step 邏輯，僅需提供結構
        return self._get_observation(), 0.0, False, False, {}

# ==========================================
# 輔助功能：決策矩陣與指標計算
# ==========================================
def peg_volume_ma_decision(peg, volume_ratio, ma50_slope_pct, market_cap, current_signal):
    if market_cap is not None and market_cap > 3e11:
        is_large_cap = True
    else: 
        is_large_cap = False
    
    if peg is not None and peg < 0.7 and ma50_slope_pct > 0:
        return "持有 (HOLD)", "PEG<0.7 且 MA50 上升 → 價值順風，不應賣出"
    
    if current_signal.startswith("卖出") and is_large_cap:
        return "持有 (HOLD)", "大型權值股（>300億），SELL 信號降級"
    
    return current_signal, None

def add_technical_indicators(df):
    df = df.copy()

    # === 原始指標 (用於顯示) ===
    df['sma_10_raw'] = df['close'].rolling(10).mean()
    df['sma_30_raw'] = df['close'].rolling(30).mean()
    df['sma_50_raw'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_raw'] = 100 - (100 / (1 + rs))
    df['macd_raw'] = df['ema_12'] - df['ema_26']
    df['macd_signal_raw'] = df['macd_raw'].ewm(span=9, adjust=False).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    # === 標準化指標 (用於 AI 模型，與 Colab 訓練一致) ===
    # 1. 對數報酬率
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    # 2. 標準化 SMA (價格相對於均線的比率)
    df['sma_10'] = df['close'] / df['close'].rolling(10).mean() - 1
    df['sma_30'] = df['close'] / df['close'].rolling(30).mean() - 1
    df['sma_50'] = df['close'] / df['close'].rolling(50).mean() - 1

    # 3. RSI 縮放到 0-1
    df['rsi'] = df['rsi_raw'] / 100.0
    df['rsi'] = df['rsi'].fillna(0.5)

    # 4. MACD 標準化 (除以價格)
    df['macd'] = (df['ema_12'] - df['ema_26']) / df['close']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    # 5. 布林帶位置 (0-1)
    df['bb_position'] = (df['close'] - (df['bb_middle'] - 2 * df['bb_std'])) / (4 * df['bb_std'])
    df['bb_position'] = df['bb_position'].fillna(0.5)
    df['bb_position'] = np.where(df['bb_std'] == 0, 0.5, df['bb_position'])

    # 6. 成交量變化 (對數)
    df['vol_change'] = np.log(df['volume'] / df['volume'].shift(1) + 1e-5)

    # 7. 波動率
    df['volatility'] = (df['high'] - df['low']) / df['close']

    # 處理 NaN 和 Inf
    for col in ['log_ret', 'sma_10', 'sma_30', 'sma_50', 'rsi', 'macd', 'macd_signal', 'bb_position', 'vol_change', 'volatility']:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(0.0)

    return df.fillna(method='bfill').fillna(method='ffill')

# ==========================================
# 核心功能：生成訊號
# ==========================================
def get_trading_signal():
    print("=" * 80)
    print("🤖 台股 2331 (精英) AI 交易訊號生成器")
    print("=" * 80)
    
    # 0. 下載數據
    print("\n📊 下載最新市場數據...")
    try:
        df = yf.download('2331.TW', period='90d', progress=False, auto_adjust=True)
        if df.empty: 
            print("❌ 無法下載數據")
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df.rename(columns={
            'Close': 'close', 
            'Volume': 'volume', 
            'Open': 'open', 
            'High': 'high', 
            'Low': 'low'
        }).reset_index()
        
        df = add_technical_indicators(df)
        latest_date = df['Date'].iloc[-1].strftime('%Y-%m-%d') if 'Date' in df.columns else '未知日期'
        
    except Exception as e:
        print(f"❌ 數據下載失敗: {e}")
        return None
    
    # 1. 建立環境 (與訓練時結構一致)
    raw_env = ImprovedTradingEnv(df)
    venv = DummyVecEnv([lambda: raw_env])
    
    # 2. 載入「數據眼鏡」 (VecNormalize 統計數據)
    stats_path = "ppo_2331_advanced_vecnorm.pkl"
    if os.path.exists(stats_path):
        print(f"✅ 載入標準化統計數據: {stats_path}")
        try:
            venv = VecNormalize.load(stats_path, venv)
            # 重要：測試模式下不更新平均值與標準差
            venv.training = False 
            venv.norm_reward = False
        except Exception as e:
            print(f"⚠️  載入標準化數據失敗: {e}")
            print("   嘗試創建新的 VecNormalize 環境...")
            venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
    else:
        print(f"⚠️  警告: 找不到 {stats_path}")
        print("   創建新的 VecNormalize 環境...")
        venv = VecNormalize(venv, training=False, norm_obs=True, norm_reward=False)
    
    # 3. 載入 AI 模型 (大腦)
    model_path = "ppo_2331_tw_improved"
    #model_path = "ppo_2331_advanced"
    try:
        model = PPO.load(model_path, env=venv)
        print(f"✅ AI 模型載入成功: {model_path}")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        
        # 嘗試使用 device='cpu' 參數
        try:
            print("   嘗試使用 CPU 載入模型...")
            model = PPO.load(model_path, env=venv, device='cpu')
            print(f"✅ AI 模型載入成功 (使用CPU): {model_path}")
        except Exception as e2:
            print(f"❌ 模型載入完全失敗: {e2}")
            return None
    
    # 4. 取得最新觀察值並預測
    try:
        # 使用環境的 reset 方法獲取初始觀察值
        obs = venv.reset()
        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0])
        print(f"\n🧠 AI 預測動作值: {action_value:+.4f}")
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        action_value = 0.0
    
    # 5. 獲取技術指標數據 (使用原始值用於顯示)
    latest_data = df.iloc[-1]
    current_price = float(latest_data['close'])
    rsi = float(latest_data['rsi_raw'])
    macd = float(latest_data['macd_raw'])
    macd_signal = float(latest_data['macd_signal_raw'])
    sma_10 = float(latest_data['sma_10_raw'])
    sma_30 = float(latest_data['sma_30_raw'])
    sma_50 = float(latest_data['sma_50_raw'])
    bb_upper = float(latest_data['bb_upper'])
    bb_lower = float(latest_data['bb_lower'])
    current_volume = float(latest_data['volume'])
    
    # 計算平均成交量（過去20天）
    avg_volume_20 = float(df['volume'].tail(20).mean()) if len(df) >= 20 else current_volume
    
    # 計算成交量比率
    volume_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0
    
    print("\n" + "=" * 80)
    print("📊 技術指標分析")
    print("=" * 80)
    print(f"當前價格:       NT${current_price:.2f}")
    print(f"RSI (14):        {rsi:.2f}  {'[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'}")
    print(f"MACD:            {macd:.4f}")
    print(f"MACD Signal:     {macd_signal:.4f}  {'[金叉]' if macd > macd_signal else '[死叉]'}")
    print(f"SMA 10:          NT${sma_10:.2f}")
    print(f"SMA 30:          NT${sma_30:.2f}  {'[多頭]' if sma_10 > sma_30 else '[空頭]'}")
    print(f"SMA 50:          NT${sma_50:.2f}")
    print(f"布林帶上軌:      NT${bb_upper:.2f}")
    print(f"布林帶下軌:      NT${bb_lower:.2f}")
    
    # 計算布林帶位置
    if (bb_upper - bb_lower) > 0:
        bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower) * 100)
        print(f"當前價格位置:    {bb_position:.1f}% (布林帶內)")
    else:
        print(f"當前價格位置:    無法計算 (布林帶範圍為0)")
    
    print(f"成交量:          {int(current_volume):,}")
    print(f"20日平均量:      {int(avg_volume_20):,}  {'[放量]' if volume_ratio > 1.5 else '[縮量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"量比:            {volume_ratio:.2f}x")
    
    # 6. 計算MA50斜率
    print("\n" + "=" * 80)
    print("📈 MA50趨勢分析")
    print("=" * 80)
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5)
    print(f"當前MA50:        NT${ma50_slope_info['ma50_current']:.2f}")
    print(f"MA50斜率:        {ma50_slope_info['slope']:+.6f}")
    print(f"斜率百分比:      {ma50_slope_info['slope_pct']:+.4f}%")
    print(f"趨勢判斷:        {ma50_slope_info['color']} {ma50_slope_info['trend']}")
    print(f"交易信號:        {ma50_slope_info['signal']}")
    print(f"\n💡 說明: {ma50_slope_info['description']}")
    
    # 7. 初始化動態權重計算器
    weight_calc = DynamicWeightCalculator('2331.TW')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()
    
    # 8. 獲取市場情緒分析
    print("\n" + "=" * 80)
    print("📰 市場情緒分析")
    print("=" * 80)
    
    try:
        from finbert_enhanced_scoring import calculate_sentiment_score
        sentiment_result = calculate_sentiment_score('2331.TW', verbose=False)
        if sentiment_result and sentiment_result['news_count'] > 0:
            print(format_sentiment_output(sentiment_result))
        else:
            print("⚠️  未找到相關新聞，情緒分析不可用")
            sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}
    except:
        print("簡化情緒分析系統")
        print("   情緒評分: 0.0 (中性)")
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}
    
    # 9. 蠟燭圖型態分析
    print("\n" + "=" * 80)
    print("📊 蠟燭圖型態分析")
    print("=" * 80)
    
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        pattern_adjustment = get_pattern_score_adjustment(patterns)
        print(f"\n型態評分調整: {pattern_adjustment:+.1f} 分")
    except Exception as e:
        print(f"   ⚠️  型態分析失敗: {e}")
        pattern_adjustment = 0
    
    # 10. 結構型態分析
    print("\n" + "=" * 80)
    print("🏗️ 結構型態分析 (60天週期)")
    print("=" * 80)
    
    try:
        structure_patterns = detect_structure_patterns(df, window=60)
        print(format_structure_pattern_output(structure_patterns))
        structure_score_bonus = get_structure_score_adjustment(structure_patterns)
    except Exception as e:
        print(f"   ⚠️  結構型態分析失敗: {e}")
        structure_score_bonus = 0
    
    # 11. 獲取基本面數據 (PEG和市值)
    try:
        ticker = yf.Ticker('2331.TW')
        info = ticker.info
        market_cap = info.get('marketCap', None)
        
        # 嘗試獲取PEG比率
        peg_ratio = info.get('pegRatio', None)
        if peg_ratio is None:
            # 簡化計算PEG
            pe_ratio = info.get('trailingPE', info.get('forwardPE', None))
            growth_rate = info.get('earningsGrowth', info.get('revenueGrowth', 0.1))
            if pe_ratio and growth_rate:
                peg_ratio = pe_ratio / (growth_rate * 100) if growth_rate != 0 else None
    except:
        market_cap = None
        peg_ratio = None
    
    # 12. 獲取分析師目標價 (嘗試)
    target_price = None
    target_high = None
    recommendation_mean = None
    
    try:
        if 'targetMeanPrice' in info:
            target_price = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice', target_price * 1.1)
            recommendation_mean = info.get('recommendationMean', 3.0)
    except:
        pass
    
    # 13. 生成交易建議
    print("\n" + "=" * 80)
    print("🎯 AI 交易信號")
    print("=" * 80)
    print(f"模型輸出動作值: {action_value:+.4f}")
    
    # 初始化變量
    signal = "持有 (HOLD)"
    signal_emoji = "🟡"
    strength = 0
    suggested_price_low = current_price
    suggested_price_high = current_price
    reasons = []
    warnings = []
    adjusted_strength = 0
    suggested_ratio = 0
    
    # 買入邏輯
    if action_value > 0.1:
        signal = "買入 (BUY)"
        signal_emoji = "🟢"
        strength = action_value
        suggested_price_low = current_price * 0.995
        suggested_price_high = current_price * 1.000
        
        # 增強版買入信號評分系統
        try:
            buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = calculate_enhanced_buy_score_with_sentiment(
                rsi=rsi,
                macd=macd,
                macd_signal=macd_signal,
                sma_10=sma_10,
                sma_30=sma_30,
                current_price=current_price,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                volume_ratio=volume_ratio,
                ai_action=action_value,
                buy_weights=buy_weights,
                symbol='2331.TW'
            )
            
            # 加入MA50斜率評分調整
            ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
            buy_score += ma50_slope_adjustment
            
            if ma50_slope_adjustment > 0:
                buy_reasons.append(f"MA50趨勢向上 (+{ma50_slope_adjustment}分)")
            elif ma50_slope_adjustment < 0:
                buy_warnings.append(f"MA50趨勢向下 ({ma50_slope_adjustment}分)")
            
            # 加入結構型態評分調整
            if structure_score_bonus > 0:
                buy_score += structure_score_bonus
                buy_reasons.append(f"結構型態加分 (+{structure_score_bonus}分)")
            
            # 加入蠟燭圖型態評分
            buy_score += pattern_adjustment
            if pattern_adjustment > 0:
                buy_reasons.append(f"蠟燭圖型態加分 (+{pattern_adjustment}分)")
            elif pattern_adjustment < 0:
                buy_warnings.append(f"蠟燭圖型態減分 ({pattern_adjustment}分)")
            
            buy_score = max(0, min(100, buy_score))
            
            reasons = buy_reasons
            warnings = buy_warnings
            
            # 調整買入強度
            adjusted_strength = max(min((buy_score / 100) * strength, 1.0), 0)
            suggested_ratio = int(adjusted_strength * 100)
            
            # 如果評分過低，改為觀望
            if buy_score < 20:
                signal = "觀望 (WAIT)"
                signal_emoji = "🟡"
                
        except Exception as e:
            print(f"⚠️  買入評分系統錯誤: {e}")
            buy_score = 50
            reasons = ["簡化評分系統"]
            adjusted_strength = strength * 0.5
            suggested_ratio = int(adjusted_strength * 100)
    
    # 賣出邏輯
    elif action_value < -0.1:
        signal = "賣出 (SELL)"
        signal_emoji = "🔴"
        strength = abs(action_value)
        suggested_price_low = current_price * 1.000
        suggested_price_high = current_price * 1.005
        
        # 改進的賣出判斷邏輯
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        is_macd_bearish = macd < macd_signal
        is_trending_down = sma_10 < sma_30
        
        # 賣出信號評分系統
        sell_score = 0
        reasons = []
        
        # 加入MA50斜率評分調整
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        if ma50_slope_adjustment < 0:
            sell_score += abs(ma50_slope_adjustment)
            reasons.append(f"MA50趨勢向下 ({ma50_slope_info['slope_pct']:.2f}%)")
        elif ma50_slope_adjustment > 0:
            sell_score -= abs(ma50_slope_adjustment) * 0.5
            if sell_score < 0:
                sell_score = 0
        
        # 分析師目標價判斷
        if target_price is not None and current_price > 0:
            target_upside_mean = ((target_price - current_price) / current_price) * 100
            target_upside_high = ((target_high - current_price) / current_price) * 100 if target_high else None
            
            # 狀況1: 股價衝破平均目標價，但評級還是 Buy
            if current_price > target_price and recommendation_mean and recommendation_mean <= 2.5:
                if target_upside_high and target_upside_high > 0:
                    reasons.append(f"🔥 動能突破! 股價已超越平均目標價，但分析師仍看好")
                    reasons.append(f"   最高目標價 NT${target_high:.2f} (上漲空間 {target_upside_high:.1f}%)")
                else:
                    reasons.append(f"🔥 動能突破! 分析師目標價滯後 (評級: {recommendation_mean:.1f}/5)")
            
            # 狀況2: 股價高於目標價，且評級差
            elif current_price > target_price and recommendation_mean and recommendation_mean > 3.0:
                sell_score += sell_weights.get('target_below', 20)
                reasons.append(f"⚠️ 股價過高! 超越目標價且評級差 ({recommendation_mean:.1f}/5)")
            
            # 狀況3: 股價低於目標價，正常的低估值判斷
            elif target_upside_mean < -10:
                sell_score += sell_weights.get('target_below', 20)
                reasons.append(f"📉 目標價低於現價 {target_upside_mean:.1f}%")
            elif target_upside_mean < 5:
                sell_score += sell_weights.get('target_near', 10)
                reasons.append(f"⚠️ 上漲空間有限 (僅{target_upside_mean:.1f}%)")
        
        # MACD 死叉
        if is_macd_bearish:
            sell_score += sell_weights.get('macd_bearish', 20)
            reasons.append("MACD 死叉,趨勢轉弱")
        
        # 均線排列
        if is_trending_down:
            sell_score += sell_weights.get('ma_bearish', 15)
            reasons.append("短期均線下穿長期均線")
        
        # 布林帶位置
        if bb_position > 90:
            sell_score += sell_weights.get('bb_upper', 10)
            reasons.append(f"價格接近布林帶上軌 ({bb_position:.1f}%)")
        elif bb_position > 80:
            sell_score += sell_weights.get('bb_high', 8)
            reasons.append(f"價格偏高,接近布林帶上軌")
        
        # 成交量分析
        is_strong_trend = (not is_macd_bearish) and (not is_trending_down) and (volume_ratio > 1.2)
        
        # 情景A: 極度放量 + RSI<80 + 多頭趨勢 = 超強勢股
        if volume_ratio > 2.5 and rsi < 80 and is_strong_trend:
            sell_score = 0
            reasons.clear()
            reasons.append(f"🚀 超強勢突破信號!")
            reasons.append(f"極度放量({volume_ratio:.1f}x) + MACD金叉 + 均線多頭")
            reasons.append(f"RSI {rsi:.1f} (未達極端超買)")
            reasons.append(f"回測數據: 此情景66.7%繼續大漲")
            reasons.append(f"💡 建議: 繼續持有100%, 設置追蹤止損")
        
        # 情景B: 普通強勢股
        elif is_strong_trend and rsi > 70:
            sell_score = int(sell_score * 0.2)
            reasons.clear()
            reasons.append(f"⚠️  RSI超買但符合強勢股特徵")
            reasons.append(f"MACD金叉 + 均線多頭 + 放量({volume_ratio:.1f}x)")
            reasons.append(f"回測: 繼續漲39.3% vs 回調28.6%")
            reasons.append(f"💡 建議: 持有或小幅減倉10-20%")
        
        # 情景C: 超級放量 + RSI>80 = 可能出貨
        elif volume_ratio > 3.0 and rsi > 80:
            sell_score += 30
            reasons.append(f"⚠️  超級放量({volume_ratio:.1f}x) + RSI嚴重超買({rsi:.1f})")
            reasons.append(f"可能是主力出貨高峰")
        
        # 情景D: 高位放量但趨勢轉弱
        elif volume_ratio > 2.0 and rsi > 70 and (not is_strong_trend):
            sell_score += 20
            reasons.append(f"高位放量({volume_ratio:.1f}x)但趨勢轉弱")
            reasons.append(f"疑似出貨信號")
        
        # 情景E: 價漲量縮
        elif volume_ratio < 0.5 and current_price > sma_10:
            sell_score += 15
            reasons.append(f"價漲量縮(量比{volume_ratio:.1f}x)，上漲乏力")
        
        # 超跌保護
        if bb_position < 35 and sell_score > 0:
            fundamentals_good_sell = False
            fundamentals_bad_sell = False
            profit_margin_sell = 0
            
            try:
                ticker_obj = yf.Ticker('2331.TW')
                info = ticker_obj.info
                net_income_sell = info.get('netIncome', None)
                profit_margin_sell = info.get('profitMargins', None)
                
                if profit_margin_sell is not None and profit_margin_sell > 0.10:
                    fundamentals_good_sell = True
                elif profit_margin_sell is not None and profit_margin_sell < 0:
                    fundamentals_bad_sell = True
                elif net_income_sell is not None and net_income_sell > 0:
                    fundamentals_good_sell = True
                elif net_income_sell is not None and net_income_sell < 0:
                    fundamentals_bad_sell = True
            except:
                pass
            
            original_score = sell_score
            
            if fundamentals_good_sell:
                sell_score = 0
                reasons.clear()
                reasons.append(f"🎯 優質公司超跌! 布林帶{bb_position:.1f}%")
                if profit_margin_sell is not None and profit_margin_sell > 0:
                    reasons.append(f"淨利率{profit_margin_sell*100:.1f}% (健康)")
                else:
                    reasons.append(f"公司盈利良好")
                reasons.append(f"💎 建議: 不要追殺,這是價值投資機會!")
                reasons.append(f"可考慮加倉或持有,等待反彈")
            elif fundamentals_bad_sell:
                reasons.clear()
                reasons.append(f"⚠️ 公司虧損,超跌合理")
                reasons.append(f"淨利為負,下跌可能持續")
                reasons.append(f"💡 建議: 可以賣出止損")
            else:
                sell_score = int(sell_score * 0.3)
                reasons.clear()
                reasons.append(f"⚠️  股價接近支撐位(布林帶{bb_position:.1f}%)")
                reasons.append(f"雖然技術指標轉弱(原評分{original_score}),但已超跌")
                reasons.append(f"💡 建議: 暫不追殺,等待反彈或進一步確認")
        
        # 調整賣出強度和建議比例
        adjusted_strength = min(sell_score / 100, 1.0) if sell_score > 0 else 0
        suggested_ratio = int(adjusted_strength * 100)
        
        # 如果評分=0，覆蓋為持有信號
        if sell_score == 0:
            signal = "持有 (HOLD - 強勢突破)"
            signal_emoji = "🟢"
    
    # =========================================================
    # 🧠 PEG × 量 × MA Trend 決策矩陣覆蓋
    # =========================================================
    signal_override, override_reason = peg_volume_ma_decision(
        peg=peg_ratio,
        volume_ratio=volume_ratio,
        ma50_slope_pct=ma50_slope_info['slope_pct'],
        market_cap=market_cap,
        current_signal=signal
    )
    
    if signal_override != signal:
        signal = signal_override
        signal_emoji = "🟡"
        if override_reason:
            reasons = [override_reason]
    
    # 輸出交易信號
    print(f"\n{signal_emoji} 信號: {signal}")
    
    if signal == "買入 (BUY)" or signal == "賣出 (SELL)":
        print(f"   AI 模型強度: {strength:.2f} / 1.00")
        if signal == "買入 (BUY)":
            if 'buy_score' in locals():
                print(f"   技術指標評分: {buy_score} / 100")
        elif signal == "賣出 (SELL)":
            print(f"   技術指標評分: {sell_score if 'sell_score' in locals() else 0} / 100")
        
        print(f"   綜合建議強度: {adjusted_strength:.2f}")
        
        if suggested_ratio > 0:
            if signal == "買入 (BUY)":
                print(f"   建議買入比例: {suggested_ratio}%")
                print(f"   建議買入價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
            elif signal == "賣出 (SELL)":
                print(f"   建議賣出比例: {suggested_ratio}%")
                print(f"   建議賣出價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
    
    if warnings:
        print(f"\n   ⚠️  警告:")
        for warning in warnings:
            print(f"      • {warning}")
    
    if reasons:
        if signal == "買入 (BUY)":
            print(f"\n   📌 買入理由:")
        elif signal == "賣出 (SELL)":
            print(f"\n   📌 賣出理由:")
        elif signal.startswith("持有"):
            print(f"\n   📌 持有理由:")
        
        for i, reason in enumerate(reasons, 1):
            print(f"      {i}. {reason}")
    
    print(f"\n   💡 操作建議:")
    if signal == "買入 (BUY)":
        if 'buy_score' in locals() and buy_score < 20:
            print(f"      • AI建議買入,但技術面支持度不足")
            print(f"      • 建議觀望,等待成交量放大")
            print(f"      • 關注支撐位: NT${bb_lower:.2f}")
        elif 'buy_score' in locals() and buy_score >= 60:
            print(f"      • 多個買入信號確認,可以買入")
            print(f"      • 分批買入,建議買入 {suggested_ratio}%")
            print(f"      • 設置止損: NT${current_price * 0.95:.2f} (-5%)")
        elif 'buy_score' in locals():
            print(f"      • 謹慎買入 {suggested_ratio}%")
            print(f"      • 等待量能確認後再加倉")
            print(f"      • 設置止損: NT${current_price * 0.95:.2f} (-5%)")
    elif signal == "賣出 (SELL)":
        if 'sell_score' in locals():
            if sell_score == 0:
                pass
            elif sell_score >= 70:
                print(f"      • ⚠️  多個賣出信號確認,建議盡快賣出")
                print(f"      • 可分2-3批賣出,保留少量倉位")
                print(f"      • 設置止損: NT${current_price * 0.97:.2f} (-3%)")
            elif sell_score >= 50:
                print(f"      • 適度賣出,建議賣出 {suggested_ratio}% 倉位")
                print(f"      • 保留部分倉位觀察後續走勢")
                print(f"      • 如果 RSI 繼續上升,再賣出剩餘倉位")
            else:
                print(f"      • AI 建議賣出,但技術指標支持度較弱")
                print(f"      • 可考慮小幅減倉 10-20%")
                print(f"      • 密切關注 MACD 和 RSI 變化")
    else:
        print(f"      • 繼續觀察市場走勢")
        print(f"      • 關注支撐位: NT${bb_lower:.2f}")
        print(f"      • 關注壓力位: NT${bb_upper:.2f}")
    
    # 14. 風險提示
    print("\n" + "=" * 80)
    print("⚠️  風險提示")
    print("=" * 80)
    print("   • 本信號由 AI 模型生成,僅供參考,不構成投資建議")
    print("   • 股市有風險,投資需謹慎")
    print("   • 請根據自身風險承受能力做出投資決策")
    print("   • 建議結合其他分析方法綜合判斷")
    print("=" * 80)
    
    return {
        'date': latest_date,
        'symbol': '2331.TW',
        'current_price': current_price,
        'signal': signal,
        'action_value': action_value,
        'strength': abs(action_value) if abs(action_value) > 0.1 else 0,
        'rsi': rsi,
        'macd': macd,
        'sma_10': sma_10,
        'sma_30': sma_30,
        'suggested_price_low': suggested_price_low if abs(action_value) > 0.1 else current_price,
        'suggested_price_high': suggested_price_high if abs(action_value) > 0.1 else current_price,
    }

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    result = get_trading_signal()
    
    if result:
        print(f"\n✅ 信號生成成功!")
        print(f"\n📱 快速摘要:")
        print(f"   股票: {result['symbol']} (精英)")
        print(f"   日期: {result['date']}")
        print(f"   價格: NT${result['current_price']:.2f}")
        print(f"   信號: {result['signal']}")
        if result['strength'] > 0:
            print(f"   強度: {result['strength']:.2f}")
        
        # 顯示AI模型準確度摘要
        try:
            print(f"   {get_model_accuracy_display('2331.TW')}")
        except:
            print(f"   準確度追蹤: 功能不可用")
    else:
        print("\n❌ 信號生成失敗")