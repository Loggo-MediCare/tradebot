"""
台股 2409 AI 交易信号生成器
======================================
使用训练好的 PPO 模型生成今日交易策略
输出: 买入/卖出/持有 信号 + 建议价格
"""

import os
# 抑制 TensorFlow 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'  # Fix Tcl/Tk error on Windows

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入动态权重计算器
from dynamic_signal_weights import DynamicWeightCalculator
# 导入增强评分模块
# 导入增强评分模块（含FinBERT情绪分析）
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
# 导入MA50斜率分析模块
from ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment
# 导入模型准确度追踪器
from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display
# 導入結構型態分析模組
from structure_pattern_analysis import detect_structure_patterns, format_structure_pattern_output, get_structure_score_adjustment
from triangle_pattern import detect_triangle, triangle_breakout
from breakout_detector import get_breakout_signal, get_advanced_breakout_signal, format_advanced_breakout_output
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
from breakout_long_red import get_breakout_long_red_signal
from chart_visualizer import plot_candlestick


# ==========================================
# 交易环境 (必须与训练时一致)
# ==========================================
class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(ImprovedTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
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
        row = self.df.iloc[self.current_step]
        current_price = float(row['close'])
        total_value = self.balance + self.shares_held * current_price
        stock_ratio = (self.shares_held * current_price) / total_value if total_value > 0 else 0
        cash_ratio = self.balance / total_value if total_value > 0 else 1

        obs = np.array([
            float(self.shares_held),
            float(self.balance),
            float(row['close']),
            float(row.get('sma_10', 0)),
            float(row.get('sma_30', 0)),
            float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)),
            float(row.get('macd', 0)),
            float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)),
            float(row.get('bb_lower', 0)),
            float(row.get('volume', 0)),
            float(self.total_profit),
            float(stock_ratio),
            float(cash_ratio),
        ], dtype=np.float32)
        return obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = float(action[0])
        else:
            action = float(action)

        action = np.clip(action, -1.0, 1.0)
        current_price = float(self.df.iloc[self.current_step]['close'])

        if action < -0.1:
            sell_ratio = abs(action)
            shares_to_sell = int(self.shares_held * sell_ratio)
            if shares_to_sell > 0:
                self.balance += shares_to_sell * current_price
                self.shares_held -= shares_to_sell
                self.total_trades += 1
        elif action > 0.1:
            buy_ratio = action
            max_can_buy = int(self.balance // current_price)
            shares_to_buy = int(max_can_buy * buy_ratio)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                self.balance -= cost
                self.shares_held += shares_to_buy
                self.total_trades += 1

        new_total_value = self.balance + self.shares_held * current_price
        self.total_profit = new_total_value - self.initial_balance
        reward = self.total_profit / self.initial_balance

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        obs = self._get_observation()

        return obs, float(reward), done, False, {}

# ==========================================
# 技术指标计算
# ==========================================
def add_technical_indicators(df):
    """添加技术指标"""
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

# ==========================================
# 交易信号生成
# ==========================================
def get_trading_signal():
    """生成今日交易信号"""
    print("=" * 80)
    print("🤖 台股 2409 AI 交易信号生成器")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 顯示AI模型準確度
    accuracy_display = get_model_accuracy_display('2409.TW')
    print(f"模型準確度: {accuracy_display}")
    print("=" * 80)

    # 1. 加载模型
    model_path = r"C:\Users\Silvi\Projects\trading-bot\ppo_2409_tw_improved"
    print(f"\n📦 加载 AI 模型: {model_path}")

    try:
        model = PPO.load(model_path)
        print("✅ 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

    # 2. 下载最新数据 (使用 period 方式获取更新的数据)
    print("\n📊 下载最新市场数据...")
    try:
        import yfinance as yf

        # 使用 period 方式下载，auto_adjust=True 确保价格是最新的
        df = yf.download('2409.TW', period='90d', progress=False, auto_adjust=True)

        if df.empty:
            print("❌ 无法获取数据")
            return None

        # 处理可能的 MultiIndex 列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()

        print(f"✅ 成功下载 {len(df)} 天数据")

        # 獲取分析師目標價和評級
        target_price = None
        target_high = None
        recommendation_mean = None
        try:
            ticker_info = yf.Ticker('2409.TW').info
            target_price = ticker_info.get('targetMeanPrice')
            target_high = ticker_info.get('targetHighPrice')
            target_low = ticker_info.get('targetLowPrice')
            num_analysts = ticker_info.get('numberOfAnalystOpinions', 0)
            recommendation_mean = ticker_info.get('recommendationMean')  # 1=Strong Buy, 5=Sell
            recommendation_key = ticker_info.get('recommendationKey', '')

            if target_price and num_analysts > 0:
                print(f"   📊 分析師目標價: NT${target_price:.2f} (平均) / NT${target_high:.2f} (最高)")
                print(f"   📊 分析師評級: {recommendation_key} ({recommendation_mean:.1f}/5, {num_analysts}位分析師)")
        except:
            pass

    except Exception as e:
        print(f"❌ 数据下载失败: {e}")
        return None

    # 3. 添加技术指标
    print("\n🔧 计算技术指标...")
    df = add_technical_indicators(df)

    # 获取今日数据 (最后一行)
    latest_data = df.iloc[-1]
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    print(f"✅ 最新数据日期: {latest_date}")
    print(f"   当前价格: NT${float(latest_data['close']):.2f}")
    print(f"   今日成交量: {int(latest_data['volume']):,}")

    # 4. 创建环境并获取观察值
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1  # 移到最后一天
    obs = env._get_observation()

    # 5. 使用模型预测
    print("\n🧠 AI 模型分析中...")
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)

    # 6. 解析交易信号
    current_price = float(latest_data['close'])
    rsi = float(latest_data['rsi'])
    macd = float(latest_data['macd'])
    macd_signal = float(latest_data['macd_signal'])
    sma_10 = float(latest_data['sma_10'])
    sma_30 = float(latest_data['sma_30'])
    bb_upper = float(latest_data['bb_upper'])
    bb_lower = float(latest_data['bb_lower'])
    current_volume = float(latest_data['volume'])

    # 计算平均成交量（过去20天）
    avg_volume_20 = float(df['volume'].tail(20).mean())

    # 计算成交量比率
    volume_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0

    print("\n" + "=" * 80)
    print("📊 技术指标分析")
    print("=" * 80)
    print(f"RSI (14):        {rsi:.2f}  {'[超买]' if rsi > 70 else '[超卖]' if rsi < 30 else '[中性]'}")
    print(f"MACD:            {macd:.4f}")
    print(f"MACD Signal:     {macd_signal:.4f}  {'[金叉]' if macd > macd_signal else '[死叉]'}")
    print(f"SMA 10:          NT${sma_10:.2f}")
    print(f"SMA 30:          NT${sma_30:.2f}  {'[多头]' if sma_10 > sma_30 else '[空头]'}")
    print(f"SMA 50:          NT${sma_30:.2f}")
    print(f"布林带上轨:      NT${bb_upper:.2f}")
    print(f"布林带下轨:      NT${bb_lower:.2f}")
    print(f"当前价格位置:    {((current_price - bb_lower) / (bb_upper - bb_lower) * 100):.1f}% (布林带内)")
    print(f"成交量:          {int(current_volume):,}")
    print(f"20日平均量:      {int(avg_volume_20):,}  {'[放量]' if volume_ratio > 1.5 else '[缩量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"量比:            {volume_ratio:.2f}x")

    # 7.1 計算MA50斜率
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

    # 7. 初始化动态权重计算器
    weight_calc = DynamicWeightCalculator('2409.TW')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

    # 7.5 获取市场情绪分析（FinBERT + VADER）
    print("\n" + "=" * 80)
    print("📰 市场情绪分析 (FinBERT NLP Engine)")
    print("=" * 80)

    from finbert_enhanced_scoring import calculate_sentiment_score, format_sentiment_output
    sentiment_result = calculate_sentiment_score('2409.TW', verbose=False)

    if sentiment_result and sentiment_result['news_count'] > 0:
        print(format_sentiment_output(sentiment_result))
    else:
        print("⚠️  未找到相关新闻，情绪分析不可用")
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}




    # 蠟燭圖型態分析
    print("\n" + "=" * 80)
    print("📊 蠟燭圖型態分析")
    print("=" * 80)

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))

        # 獲取型態評分調整
        pattern_adjustment = get_pattern_score_adjustment(patterns)
        print(f"\n型態評分調整: {pattern_adjustment:+.1f} 分")
    except Exception as e:
        print(f"   ⚠️  型態分析失敗: {e}")
        pattern_adjustment = 0

    # 7.8 結構型態分析 (W底/鍋底)
    print("\n" + "=" * 80)
    print("📐 結構型態分析 (W底/鍋底)")
    print("=" * 80)

    try:
        structure_patterns = detect_structure_patterns(df, window=60)
        print(format_structure_pattern_output(structure_patterns))
        structure_score_bonus = get_structure_score_adjustment(structure_patterns)
    except Exception as e:
        print(f"   ⚠️  結構型態分析失敗: {e}")
        structure_score_bonus = 0

    # 7.9 進階突破型態檢測 (W底/杯柄/旗形/三角)
    print("\n" + "=" * 80)
    print("🚀 進階突破型態檢測")
    print("=" * 80)

    advanced_breakout_score = 0
    advanced_breakout_patterns = []
    try:
        advanced_signal = get_advanced_breakout_signal(df)
        if advanced_signal['detected']:
            print(f"✅ 檢測到 {len(advanced_signal['patterns'])} 個突破型態:")
            print(format_advanced_breakout_output(advanced_signal))
            advanced_breakout_score = advanced_signal['total_score']
            advanced_breakout_patterns = advanced_signal['patterns']
        else:
            print("   未檢測到進階突破型態")
    except Exception as e:
        print(f"   ⚠️  進階突破檢測失敗: {e}")

    # 8. 生成交易建议
    print("\n" + "=" * 80)
    print("🎯 AI 交易信号")
    print("=" * 80)
    print(f"模型输出动作值: {action_value:+.4f}")

    if action_value > 0.1:
        signal = "买入 (BUY)"
        signal_emoji = "🟢"
        strength = action_value
        suggested_price_low = current_price * 0.995
        suggested_price_high = current_price * 1.000

        # 🔥 买入信号评分系统（加入成交量判断）
        # 🔥 增强版买入信号评分系统
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
            symbol='2409.TW'
        )

        # 加入MA50斜率評分調整
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        buy_score += ma50_slope_adjustment

        if ma50_slope_adjustment > 0:
            buy_reasons.append(f"MA50趨勢向上 (+{ma50_slope_adjustment}分)")
        elif ma50_slope_adjustment < 0:
            buy_warnings.append(f"MA50趨勢向下 ({ma50_slope_adjustment}分)")

        # 三角收斂型態檢測
        if detect_triangle(df):
            status = triangle_breakout(df)
            if status == "BREAK_UP":
                buy_score += 10
                buy_reasons.append("三角收斂向上突破")
            elif status == "BREAK_DOWN":
                buy_warnings.append("跌破三角收斂")

        # 真假突破檢測
        breakout_signal = get_breakout_signal(df)
        if breakout_signal['detected']:
            if breakout_signal['type'] == 'TRUE_BREAKOUT':
                buy_score += 15
                buy_reasons.append(breakout_signal['signal_text'])
            elif breakout_signal['type'] == 'FALSE_BREAKOUT':
                buy_score -= 10
                buy_warnings.append(breakout_signal['signal_text'])

        # 圖表型態識別
        pattern_signal = get_pattern_signal(df)
        if pattern_signal['patterns']:
            if pattern_signal['score_adjustment'] > 0:
                buy_score += pattern_signal['score_adjustment']
                buy_reasons.append(f"型態: {pattern_signal['signal_text']}")
            elif pattern_signal['score_adjustment'] < 0:
                buy_warnings.append(f"型態警示: {pattern_signal['signal_text']}")

        # 爆量信號檢測 (法人上車)
        volume_signal = get_volume_signal(df)
        if volume_signal['surge'] and volume_signal['surge']['detected']:
            if volume_signal['surge']['type'] == 'SURGE_UP':
                buy_score += 15
                buy_reasons.append(volume_signal['surge']['signal_text'])
            elif volume_signal['surge']['type'] == 'SURGE_DOWN':
                buy_warnings.append(volume_signal['surge']['signal_text'])

        # 突破長紅檢測
        blr_signal = get_breakout_long_red_signal(df)
        if blr_signal['detected']:
            buy_score += blr_signal['score_adjustment']
            buy_reasons.append(blr_signal['signal_text'])
        elif blr_signal['score_adjustment'] > 0:
            buy_score += blr_signal['score_adjustment']
            buy_reasons.append(blr_signal['signal_text'])





        # 加入結構型態評分調整 (W底/鍋底)
        if structure_score_bonus > 0:
            buy_score += structure_score_bonus
            buy_reasons.append(f"結構型態加分 (+{structure_score_bonus}分)")

        # 加入進階突破型態評分 (W底/杯柄/旗形/三角)
        if advanced_breakout_score > 0:
            buy_score += advanced_breakout_score
            for pattern in advanced_breakout_patterns:
                signal_text = pattern.get('signal_text', '')
                if signal_text:
                    buy_reasons.append(f"🚀 {signal_text}")

        buy_score = max(0, min(100, buy_score))  # 限制在0-100之間


        # 使用增强评分结果
        reasons = buy_reasons
        warnings = buy_warnings

        # 🔥 调整买入强度
        adjusted_buy_strength = max(min((buy_score / 100) * strength, 1.0), 0)
        suggested_buy_ratio = int(adjusted_buy_strength * 100)

        # 如果评分过低，改为观望
        if buy_score < 20:
            signal = "观望 (WAIT)"
            signal_emoji = "🟡"

        print(f"\n{signal_emoji} 信号: {signal}")
        print(f"   AI 模型强度: {strength:.2f} / 1.00")
        print(f"   技术指标评分: {buy_score} / 100")
        print(f"   综合建议强度: {adjusted_buy_strength:.2f}")

        if buy_score >= 20:
            print(f"   建议买入比例: {suggested_buy_ratio}%")
            print(f"   建议买入价格区间: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")

        if warnings:
            print(f"\n   ⚠️  警告:")
            for warning in warnings:
                print(f"      • {warning}")

        if reasons:
            print(f"\n   📌 买入理由:")
            for i, reason in enumerate(reasons, 1):
                print(f"      {i}. {reason}")

        print(f"\n   💡 操作建议:")
        if buy_score < 20:
            print(f"      • AI建议买入,但技术面支持度不足")
            print(f"      • 建议观望,等待成交量放大")
            print(f"      • 关注支撑位: NT${bb_lower:.2f}")
        elif buy_score >= 60:
            print(f"      • 多个买入信号确认,可以买入")
            print(f"      • 分批买入,建议买入 {suggested_buy_ratio}%")
            print(f"      • 设置止损: NT${current_price * 0.95:.2f} (-5%)")
        else:
            print(f"      • 谨慎买入 {suggested_buy_ratio}%")
            print(f"      • 等待量能确认后再加仓")
            print(f"      • 设置止损: NT${current_price * 0.95:.2f} (-5%)")

    elif action_value < -0.1:
        signal = "卖出 (SELL)"
        signal_emoji = "🔴"
        strength = abs(action_value)
        suggested_price_low = current_price * 1.000
        suggested_price_high = current_price * 1.005

        # 🔥 改进的卖出判断逻辑
        # 计算更多技术指标
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        is_macd_bearish = macd < macd_signal
        is_trending_down = sma_10 < sma_30

        # 卖出信号评分系统（0-100分）
        sell_score = 0
        reasons = []

        # 加入MA50斜率評分調整 (負斜率增加賣出分數)
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        if ma50_slope_adjustment < 0:
            sell_score += abs(ma50_slope_adjustment)
            reasons.append(f"MA50趨勢向下 ({ma50_slope_info['slope_pct']:.2f}%)")
        elif ma50_slope_adjustment > 0:
            # MA50向上趨勢，降低賣出評分
            sell_score -= abs(ma50_slope_adjustment) * 0.5  # 使用0.5係數減少影響
            if sell_score < 0:
                sell_score = 0

        # 1. 分析師目標價判斷 (使用动态权重)
        # 投資官邏輯：修復數據盲點
        if target_price is not None and current_price > 0:
            target_upside_mean = ((target_price - current_price) / current_price) * 100
            target_upside_high = ((target_high - current_price) / current_price) * 100 if target_high else None

            # 狀況1: 股價衝破平均目標價，但評級還是 Buy (1.0-2.5) = 動能突破
            if current_price > target_price and recommendation_mean and recommendation_mean <= 2.5:
                # 這是動能股！分析師目標價滯後，不應賣出
                if target_upside_high and target_upside_high > 0:
                    reasons.append(f"🔥 動能突破! 股價已超越平均目標價，但分析師仍看好")
                    reasons.append(f"   最高目標價 NT${target_high:.2f} (上漲空間 {target_upside_high:.1f}%)")
                else:
                    reasons.append(f"🔥 動能突破! 分析師目標價滯後 (評級: {recommendation_mean:.1f}/5)")

            # 狀況2: 股價高於目標價，且評級差 (>3.0) = 危險
            elif current_price > target_price and recommendation_mean and recommendation_mean > 3.0:
                sell_score += sell_weights['target_below']
                reasons.append(f"⚠️ 股價過高! 超越目標價且評級差 ({recommendation_mean:.1f}/5)")

            # 狀況3: 股價低於目標價，正常的低估值判斷
            elif target_upside_mean < -10:
                sell_score += sell_weights['target_below']
                reasons.append(f"📉 目標價低於現價 {target_upside_mean:.1f}%")
            elif target_upside_mean < 5:
                sell_score += sell_weights['target_near']
                reasons.append(f"⚠️ 上漲空間有限 (僅{target_upside_mean:.1f}%)")

        # 2. MACD 死叉 (使用动态权重)
        if is_macd_bearish:
            sell_score += sell_weights['macd_bearish']
            reasons.append("MACD 死叉,趋势转弱")

        # 3. 均线排列 (使用动态权重)
        if is_trending_down:
            sell_score += sell_weights['ma_bearish']
            reasons.append("短期均线下穿长期均线")

        # 4. 布林带位置 (使用动态权重)
        if bb_position > 90:
            sell_score += sell_weights['bb_upper']
            reasons.append(f"价格接近布林带上轨 ({bb_position:.1f}%)")
        elif bb_position > 80:
            sell_score += sell_weights['bb_high']
            reasons.append(f"价格偏高,接近布林带上轨")

        # 6. 成交量分析（量价配合）- 基于回测数据优化
        is_strong_trend = (not is_macd_bearish) and (not is_trending_down) and (volume_ratio > 1.2)

        # 🔥 情景A: 极度放量 + RSI<80 + 多头趋势 = 超强势股
        # 回测2317: 66.7%继续涨, 0%回调 → 应该100%持有!
        if volume_ratio > 2.5 and rsi < 80 and is_strong_trend:
            sell_score = 0  # 评分归零 = 不卖出
            reasons.clear()
            reasons.append(f"🚀 超强势突破信号!")
            reasons.append(f"极度放量({volume_ratio:.1f}x) + MACD金叉 + 均线多头")
            reasons.append(f"RSI {rsi:.1f} (未达极端超买)")
            reasons.append(f"回测数据: 此情景66.7%继续大涨")
            reasons.append(f"💡 建议: 继续持有100%, 设置追踪止损")

        # 🔥 情景B: 普通强势股 (量比1.2-2.5) - 不应卖出
        elif is_strong_trend and rsi > 70:
            sell_score = int(sell_score * 0.2)  # 评分打2折
            signal = "持有 (HOLD - 强势股)"  # 覆盖为持有
            signal_emoji = "🟡"
            reasons.clear()
            reasons.append(f"📈 RSI超买但符合强势股特征")
            reasons.append(f"MACD金叉 + 均线多头 + 放量({volume_ratio:.1f}x)")
            reasons.append(f"💡 建议: 持有或小幅减仓10-20%")

        # 🔥 情景C: 超级放量(>3.0x) + RSI>80 = 可能出货
        elif volume_ratio > 3.0 and rsi > 80:
            sell_score += 30
            reasons.append(f"⚠️  超级放量({volume_ratio:.1f}x) + RSI严重超买({rsi:.1f})")
            reasons.append(f"可能是主力出货高峰")

        # 🔥 情景D: 高位放量但趋势转弱
        elif volume_ratio > 2.0 and rsi > 70 and (not is_strong_trend):
            sell_score += 20
            reasons.append(f"高位放量({volume_ratio:.1f}x)但趋势转弱")
            reasons.append(f"疑似出货信号")

        # 🔥 情景E: 价涨量缩
        elif volume_ratio < 0.5 and current_price > sma_10:
            sell_score += 15
            reasons.append(f"价涨量缩(量比{volume_ratio:.1f}x)，上涨乏力")

        # 🛡️ 超跌保护（接近布林带下轨 = 不应追杀！）- 加入基本面判断
        # 使用 < 35 来捕捉超跌股票（布林带下1/3区域）
        if bb_position < 35 and sell_score > 0:
            # 获取公司基本面（超跌时才检查）
            fundamentals_good_sell = False
            fundamentals_bad_sell = False
            profit_margin_sell = 0

            try:
                ticker_obj = yf.Ticker('2409.TW')
                info = ticker_obj.info
                net_income_sell = info.get('netIncome', None)
                profit_margin_sell = info.get('profitMargins', None)

                # 优先使用利润率判断（更可靠）
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
                # 优质公司超跌 = 不要卖！可能是买入机会
                sell_score = 0  # 评分归零
                reasons.clear()
                reasons.append(f"🎯 优质公司超跌! 布林带{bb_position:.1f}%")
                if profit_margin_sell is not None and profit_margin_sell > 0:
                    reasons.append(f"淨利率{profit_margin_sell*100:.1f}% (健康)")
                else:
                    reasons.append(f"公司盈利良好")
                reasons.append(f"💎 建议: 不要追杀,这是价值投资机会!")
                reasons.append(f"可考虑加仓或持有,等待反弹")
            elif fundamentals_bad_sell:
                # 亏损公司超跌 = 合理下跌,可以卖出
                sell_score = original_score  # 保持原评分
                reasons.clear()
                reasons.append(f"⚠️ 公司亏损,超跌合理")
                reasons.append(f"淨利为负,下跌可能持续")
                reasons.append(f"💡 建议: 可以卖出止损")
            else:
                # 无法获取基本面,使用原保护逻辑
                sell_score = int(sell_score * 0.3)  # 评分打3折
                reasons.clear()
                reasons.append(f"⚠️  股价接近支撑位(布林带{bb_position:.1f}%)")
                reasons.append(f"虽然技术指标转弱(原评分{original_score}),但已超跌")
                reasons.append(f"💡 建议: 暂不追杀,等待反弹或进一步确认")

        # 调整卖出强度和建议比例
        adjusted_strength = min(sell_score / 100, 1.0)  # 根据评分调整强度
        suggested_sell_ratio = int(adjusted_strength * 100)

        # 🔥 如果评分=0，覆盖为持有信号
        if sell_score == 0:
            signal = "持有 (HOLD - 强势突破)"
            signal_emoji = "🟢"

        print(f"\n{signal_emoji} 信号: {signal}")
        print(f"   AI 模型强度: {strength:.2f} / 1.00")
        print(f"   技术指标评分: {sell_score} / 100")
        print(f"   综合建议强度: {adjusted_strength:.2f}")

        if sell_score > 0:
            print(f"   建议卖出比例: {suggested_sell_ratio}%")
            print(f"   建议卖出价格区间: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")

        if reasons:
            if signal.startswith("持有"):
                print(f"\n   📌 持有理由:")
            else:
                print(f"\n   📌 卖出理由:")
            for i, reason in enumerate(reasons, 1):
                print(f"      {i}. {reason}")

        # 根据评分给出不同的操作建议
        print(f"\n   💡 操作建议:")
        if sell_score == 0:
            # 已经在reasons里输出了
            pass
        elif sell_score >= 70:
            print(f"      • ⚠️  多个卖出信号确认,建议尽快卖出")
            print(f"      • 可分2-3批卖出,保留少量仓位")
            print(f"      • 设置止损: NT${current_price * 0.97:.2f} (-3%)")
        elif sell_score >= 50:
            print(f"      • 适度卖出,建议卖出 {suggested_sell_ratio}% 仓位")
            print(f"      • 保留部分仓位观察后续走势")
            print(f"      • 如果 RSI 继续上升,再卖出剩余仓位")
        else:
            print(f"      • AI 建议卖出,但技术指标支持度较弱")
            print(f"      • 可考虑小幅减仓 10-20%")
            print(f"      • 密切关注 MACD 和 RSI 变化")

    else:
        signal = "持有 (HOLD)"
        signal_emoji = "🟡"

        print(f"\n{signal_emoji} 信号: {signal}")
        print(f"   市场观望,暂不操作")
        print(f"\n   💡 操作建议:")
        print(f"      • 继续观察市场走势")
        print(f"      • 关注支撑位: NT${bb_lower:.2f}")
        print(f"      • 关注压力位: NT${bb_upper:.2f}")

    # 8. 风险提示
    print("\n" + "=" * 80)
    print("⚠️  风险提示")
    print("=" * 80)
    print("   • 本信号由 AI 模型生成,仅供参考,不构成投资建议")
    print("   • 股市有风险,投资需谨慎")
    print("   • 请根据自身风险承受能力做出投资决策")
    print("   • 建议结合其他分析方法综合判断")
    print("=" * 80)

    return {
        'date': latest_date,
        'symbol': '2409.TW',
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
        print(f"\n✅ 信号生成成功!")

        # 生成 K 線圖
        try:
            import yfinance as yf
            chart_df = yf.Ticker("2409.TW").history(period="6mo")
            chart_df.columns = [c.lower() for c in chart_df.columns]
            chart_path = "2409_chart.png"
            plot_candlestick(chart_df, "2409.TW", save_path=chart_path)
        except Exception as e:
            print(f"   圖表生成失敗: {e}")

        print(f"\n📱 快速摘要:")
        print(f"   股票: {result['symbol']}")
        print(f"   日期: {result['date']}")
        print(f"   价格: NT${result['current_price']:.2f}")
        print(f"   信号: {result['signal']}")
        if result['strength'] > 0:
            print(f"   强度: {result['strength']:.2f}")


        # 顯示AI模型準確度摘要
        print(f"   {get_model_accuracy_display('2409.TW')}")
    else:
        print("\n❌ 信号生成失败")
