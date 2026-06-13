
"""
美股 8064.TWO (東捷科技) AI 交易信号生成器 V2
台股 8064.TWO (東捷科技) AI 交易信号生成器 V2
========================================
=======

🔥 加碼條件判斷系統:
   ✅ MA50 上升趨勢
   ✅ 結構型態完成 (W底/鍋底)
   ✅ 量能健康 (量比 > 1.0)

📌 信號解讀:
   - 買入強度 0-100 分
   - 加碼信號: 三條件全部滿足
   - 觀望信號: 條件不足

================================================================================
"""

import os
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

# 导入模块
from dynamic_signal_weights import DynamicWeightCalculator
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output, calculate_sentiment_score
from tavily_news import print_tavily_news
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
from ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment
from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display
from structure_pattern_analysis import detect_structure_patterns, format_structure_pattern_output, get_structure_score_adjustment
from triangle_pattern import detect_triangle, triangle_breakout
from breakout_detector import get_breakout_signal, get_advanced_breakout_signal, format_advanced_breakout_output
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
from breakout_long_red import get_breakout_long_red_signal
from chart_visualizer import plot_candlestick
from backtest_utils import calculate_ppo_backtest_roi, print_ppo_action_line
class SignalFormatter:
    """統一的信號輸出格式化器"""
    
    @staticmethod
    def print_header(title, width=80):
        print("\n" + "═" * width)
        print(f"  {title}")
        print("═" * width)
    
    @staticmethod
    def print_section(title, width=80):
        print("\n" + "─" * width)
        print(f"📊 {title}")
        print("─" * width)
    
    @staticmethod
    def print_metric(label, value, status="", width=20):
        status_str = f"  {status}" if status else ""
        print(f"   {label:<{width}} {value}{status_str}")
    
    @staticmethod
    def print_signal_box(signal_type, strength, score, recommendations):
        """打印信號框"""
        emoji_map = {
            "BUY": "🟢",
            "SELL": "🔴", 
            "HOLD": "🟡",
            "ADD": "🔥",  # 加碼
            "WAIT": "⏸️"
        }
        
        emoji = emoji_map.get(signal_type, "⚪")
        
        print("\n" + "╔" + "═" * 58 + "╗")
        print(f"║  {emoji} 交易信號: {signal_type:<20} 強度: {strength:.2f}  ║")
        print(f"║  📈 技術評分: {score}/100                              ║")
        print("╠" + "═" * 58 + "╣")
        
        for rec in recommendations[:5]:  # 最多顯示5條建議
            rec_display = rec[:52] if len(rec) > 52 else rec
            print(f"║  • {rec_display:<54} ║")
        
        print("╚" + "═" * 58 + "╝")


# ==========================================
# 加碼條件檢查器
# ==========================================
class AddPositionChecker:
    """
    加碼條件判斷系統
    三大條件: MA50上升 + 結構型態完成 + 量能健康
    """
    
    def __init__(self):
        self.conditions = {
            'ma50_rising': False,
            'structure_complete': False,
            'volume_healthy': False
        }
        self.details = {}
    
    def check_ma50_rising(self, ma50_slope_info):
        """檢查MA50是否上升"""
        slope_pct = ma50_slope_info.get('slope_pct', 0)
        trend = ma50_slope_info.get('trend', '')
        
        # MA50斜率 > 0 且趨勢為上升
        is_rising = slope_pct > 0 and '上升' in trend
        
        self.conditions['ma50_rising'] = is_rising
        self.details['ma50'] = {
            'value': slope_pct,
            'status': '✅ 上升' if is_rising else '❌ 下降/持平',
            'description': f"斜率 {slope_pct:+.4f}%"
        }
        
        return is_rising
    
    def check_structure_complete(self, structure_patterns):
        """檢查結構型態是否完成"""
        pattern_detected = structure_patterns.get('pattern_detected', False)
        pattern_type = structure_patterns.get('pattern_type', '')
        confidence = structure_patterns.get('confidence', 0)
        
        # 檢測到型態且信心度 > 60%
        is_complete = pattern_detected and confidence >= 60
        
        self.conditions['structure_complete'] = is_complete
        self.details['structure'] = {
            'value': confidence,
            'status': f'✅ {pattern_type}' if is_complete else '❌ 未完成',
            'description': f"信心度 {confidence}%" if pattern_detected else "無明顯型態"
        }
        
        return is_complete
    
    def check_volume_healthy(self, volume_ratio, avg_volume_trend=None):
        """檢查量能是否健康"""
        # 量比 > 0.8 視為健康（不需要放量，但不能太縮量）
        is_healthy = volume_ratio >= 0.8
        
        if volume_ratio >= 1.5:
            status = '✅ 放量 (強勢)'
        elif volume_ratio >= 1.0:
            status = '✅ 正常'
        elif volume_ratio >= 0.8:
            status = '✅ 略縮 (可接受)'
        else:
            status = '❌ 嚴重縮量'
        
        self.conditions['volume_healthy'] = is_healthy
        self.details['volume'] = {
            'value': volume_ratio,
            'status': status,
            'description': f"量比 {volume_ratio:.2f}x"
        }
        
        return is_healthy
    
    def can_add_position(self):
        """判斷是否可以加碼"""
        return all(self.conditions.values())
    
    def get_conditions_met(self):
        """獲取滿足的條件數量"""
        return sum(self.conditions.values())
    
    def print_summary(self):
        """打印加碼條件摘要"""
        print("\n" + "═" * 60)
        print("  🔥 加碼條件檢查")
        print("═" * 60)
        
        condition_labels = {
            'ma50_rising': 'MA50 上升趨勢',
            'structure_complete': '結構型態完成',
            'volume_healthy': '量能健康'
        }
        
        for key, label in condition_labels.items():
            detail = self.details.get(key.replace('_rising', '').replace('_complete', '').replace('_healthy', ''), {})
            status = detail.get('status', '❓ 未檢查')
            desc = detail.get('description', '')
            print(f"   {label:<20} {status:<15} ({desc})")
        
        met = self.get_conditions_met()
        can_add = self.can_add_position()
        
        print("─" * 60)
        if can_add:
            print(f"   🎯 結論: ✅ 三條件全部滿足，可以加碼!")
        else:
            print(f"   🎯 結論: ⚠️ 僅滿足 {met}/3 條件，建議觀望")
        print("═" * 60)
        
        return can_add


# ==========================================
# 交易环境
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
# 主要交易信號生成
# ==========================================
    fmt.print_header("🤖 美股 8064.TWO (東捷科技) AI 交易信号生成器 V2")
def get_trading_signal():
    """生成今日交易信号 (V2 重寫版)"""
    
    fmt = SignalFormatter()
    add_checker = AddPositionChecker()
    
    # ========== 標題區 ==========
    fmt.print_header("🤖 台股 8064.TWO (東捷科技) AI " \
    "交易信号生成器 V2")

    fmt.print_header("🤖 美股 8064.TWO (東捷科技) AI 交易信号生成器 V2")
    print(f"   📅 生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   📊 模型準確度: {get_model_accuracy_display('8064.TWO')}")

    # ========== 載入模型 ==========
    model_path = r"C:\Users\Silvi\Projects\trading-bot\ppo_8064_two_improved"
    
    try:
        model = PPO.load(model_path)
        print(f"   ✅ AI 模型載入成功")
    except Exception as e:
        print(f"   ❌ 模型載入失敗: {e}")
        return None

    # ========== 下載數據 ==========
    try:
        import yfinance as yf
        df = yf.download('8064.TWO', period='90d', progress=False, auto_adjust=True)
        
        if df.empty:
            print("   ❌ 無法獲取數據")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df = df.rename(columns={
            'Close': 'close', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low'
        })
        df = df.reset_index()
        
        print(f"   ✅ 成功下載 {len(df)} 天數據")

        # 獲取分析師資訊
        target_price, target_high, target_low = None, None, None
        recommendation_mean, recommendation_key = None, ''
        num_analysts = 0
        
        try:
            ticker_info = yf.Ticker('8064.TWO').info
            target_price = ticker_info.get('targetMeanPrice')
            target_high = ticker_info.get('targetHighPrice')
            target_low = ticker_info.get('targetLowPrice')
            num_analysts = ticker_info.get('numberOfAnalystOpinions', 0)
            recommendation_mean = ticker_info.get('recommendationMean')
            recommendation_key = ticker_info.get('recommendationKey', '')
        except:
            pass

    except Exception as e:
        print(f"   ❌ 數據下載失敗: {e}")
        return None

    # ========== 計算技術指標 ==========
    df = add_technical_indicators(df)
    latest_data = df.iloc[-1]
    
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price = float(latest_data['close'])
    rsi = float(latest_data['rsi'])
    macd = float(latest_data['macd'])
    macd_signal = float(latest_data['macd_signal'])
    sma_10 = float(latest_data['sma_10'])
    sma_30 = float(latest_data['sma_30'])
    sma_50 = float(latest_data['sma_50'])
    bb_upper = float(latest_data['bb_upper'])
    bb_lower = float(latest_data['bb_lower'])
    current_volume = float(latest_data['volume'])
    avg_volume_20 = float(df['volume'].tail(20).mean())
    volume_ratio = (current_volume / avg_volume_20) if avg_volume_20 > 0 else 1.0

    # ========== 技術指標摘要 ==========
    fmt.print_section("技術指標摘要")
    
    rsi_status = '[超買]' if rsi > 70 else '[超賣]' if rsi < 30 else '[中性]'
    macd_status = '[金叉]' if macd > macd_signal else '[死叉]'
    ma_status = '[多頭]' if sma_10 > sma_30 else '[空頭]'
    vol_status = '[放量]' if volume_ratio > 1.5 else '[縮量]' if volume_ratio < 0.7 else '[正常]'
    
    fmt.print_metric("股票代碼", "8064.TWO (東捷科技)")
    fmt.print_metric("當前價格", f"NT${current_price:.2f}")
    fmt.print_metric("股票代碼", "8064.TWO (東捷科技)")
    fmt.print_metric("當前價格", f"NT${current_price:.2f}")
    fmt.print_metric("數據日期", latest_date)
    print()
    fmt.print_section("MA50 趨勢分析")
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5)
    
    fmt.print_metric("當前 MA50", f"${ma50_slope_info['ma50_current']:.2f}")

    fmt.print_metric("RSI (14)", f"{rsi:.2f}", rsi_status)
    fmt.print_metric("MACD", f"{macd:.4f}", macd_status)
    fmt.print_metric("SMA 10/30", f"NT${sma_10:.2f} / NT${sma_30:.2f}", ma_status)
    fmt.print_metric("布林帶", f"NT${bb_lower:.2f} - NT${bb_upper:.2f}")
    fmt.print_metric("SMA 10/30", f"NT${sma_10:.2f} / NT${sma_30:.2f}", ma_status)
    fmt.print_metric("布林帶", f"NT${bb_lower:.2f} - NT${bb_upper:.2f}")
    fmt.print_metric("量比", f"{volume_ratio:.2f}x", vol_status)

    if target_price and num_analysts > 0:
        upside = ((target_price - current_price) / current_price) * 100
        print()
        fmt.print_metric("分析師目標價", f"NT${target_price:.2f} ({upside:+.1f}%)")
        fmt.print_metric("分析師目標價", f"NT${target_price:.2f} ({upside:+.1f}%)")
        fmt.print_metric("分析師評級", f"{recommendation_key} ({num_analysts}位)")

    # ========== MA50 趨勢分析 ==========
    fmt.print_section("MA50 趨勢分析")
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5)
    
    fmt.print_metric("當前 MA50", f"NT${ma50_slope_info['ma50_current']:.2f}")

    fmt.print_section("MA50 趨勢分析")
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5)
    
    fmt.print_metric("當前 MA50", f"${ma50_slope_info['ma50_current']:.2f}")
    fmt.print_metric("MA50 斜率", f"{ma50_slope_info['slope_pct']:+.4f}%")
    fmt.print_metric("趨勢判斷", f"{ma50_slope_info['color']} {ma50_slope_info['trend']}")
    
    # 檢查加碼條件 1: MA50上升
    add_checker.check_ma50_rising(ma50_slope_info)

    # ========== 結構型態分析 ==========
    fmt.print_section("結構型態分析 (W底/鍋底)")
    
    try:
        structure_patterns = detect_structure_patterns(df, window=60)
        print(format_structure_pattern_output(structure_patterns))
        structure_score_bonus = get_structure_score_adjustment(structure_patterns)
        
        # 檢查加碼條件 2: 結構型態完成
        add_checker.check_structure_complete(structure_patterns)
    except Exception as e:
        print(f"   ⚠️ 分析失敗: {e}")
        structure_patterns = {'pattern_detected': False, 'confidence': 0}
        structure_score_bonus = 0
        add_checker.check_structure_complete(structure_patterns)

    # 檢查加碼條件 3: 量能健康
    add_checker.check_volume_healthy(volume_ratio)

    # ========== 蠟燭圖型態 ==========
    fmt.print_section("蠟燭圖型態分析")
    
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        pattern_adjustment = get_pattern_score_adjustment(patterns)
    except Exception as e:
        print(f"   ⚠️ 分析失敗: {e}")
        pattern_adjustment = 0

    # ========== 市場情緒分析 ==========
    fmt.print_section("市場情緒分析 (FinBERT)")
    
    sentiment_result = calculate_sentiment_score('8064.TWO', verbose=False)
    if sentiment_result and sentiment_result['news_count'] > 0:
        print(format_sentiment_output(sentiment_result))
    else:
        print("   ⚠️ 未找到相關新聞")
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}

    # ── Tavily 即時新聞 ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🌐 8064.TWO (8064.TWO) 即時新聞  (Tavily REST API)")
    print("=" * 80)
    print_tavily_news('8064.TWO', '8064.TWO', max_results=5)


    # ========== AI 模型預測 ==========
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1
    obs = env._get_observation()
    
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    # PPO backtest ROI
    _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)

    # ========== 生成交易建議 ==========
    weight_calc = DynamicWeightCalculator('8064.TWO')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

    fmt.print_section("AI 交易信號")
    print(f"   模型輸出動作值: {action_value:+.4f}")

    # ========== 處理買入信號 ==========
    if action_value > 0.1:
        strength = action_value
        suggested_price_low = current_price * 0.995
        suggested_price_high = current_price * 1.000

        buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = \
            calculate_enhanced_buy_score_with_sentiment(
                rsi=rsi, macd=macd, macd_signal=macd_signal,
                sma_10=sma_10, sma_30=sma_30, current_price=current_price,
                bb_upper=bb_upper, bb_lower=bb_lower, volume_ratio=volume_ratio,
                ai_action=action_value, buy_weights=buy_weights, symbol='8064.TWO'
            )

        # 加入 MA50 斜率評分調整
        ma50_adj = get_ma50_slope_score_adjustment(ma50_slope_info)
        buy_score += ma50_adj
        if ma50_adj > 0:
            buy_reasons.append(f"MA50趨勢向上 (+{ma50_adj}分)")
        elif ma50_adj < 0:
            buy_warnings.append(f"MA50趨勢向下 ({ma50_adj}分)")

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





        # 加入結構型態評分調整
        if structure_score_bonus > 0:
            buy_score += structure_score_bonus
            buy_reasons.append(f"結構型態加分 (+{structure_score_bonus}分)")

        buy_score = max(0, min(100, buy_score))
        adjusted_buy_strength = max(min((buy_score / 100) * strength, 1.0), 0)
        suggested_buy_ratio = int(adjusted_buy_strength * 100)

        # 決定信號類型
        if buy_score < 20:
            signal_type = "WAIT"
            signal_text = "觀望 (WAIT)"
        elif add_checker.can_add_position() and buy_score >= 60:
            signal_type = "ADD"
            signal_text = "🔥 加碼 (ADD POSITION)"
        else:
            signal_type = "BUY"
            signal_text = "買入 (BUY)"

        # 生成建議列表
        recommendations = []
        if signal_type == "ADD":
            recommendations.append("三大加碼條件全部滿足!")
            recommendations.append(f"建議加碼比例: {suggested_buy_ratio}%")
        elif signal_type == "BUY":
            recommendations.append(f"建議買入比例: {suggested_buy_ratio}%")
        
        recommendations.append(f"建議價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
        recommendations.append(f"止損設置: NT${current_price * 0.95:.2f} (-5%)")
        recommendations.append(f"建議價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
        recommendations.append(f"止損設置: NT${current_price * 0.95:.2f} (-5%)")
        
        for reason in buy_reasons[:3]:
            recommendations.append(f"✓ {reason}")

        fmt.print_signal_box(signal_type, adjusted_buy_strength, buy_score, recommendations)

        # 顯示警告
        if buy_warnings:
            print("\n   ⚠️ 警告:")
            for w in buy_warnings[:3]:
                print(f"      • {w}")

    # ========== 處理賣出信號 ==========
    elif action_value < -0.1:
        strength = abs(action_value)
        suggested_price_low = current_price * 1.000
        suggested_price_high = current_price * 1.005

        sell_score = 0
        reasons = []

        # 賣出評分邏輯
        is_macd_bearish = macd < macd_signal
        is_trending_down = sma_10 < sma_30
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50

        ma50_adj = get_ma50_slope_score_adjustment(ma50_slope_info)
        if ma50_adj < 0:
            sell_score += abs(ma50_adj)
            reasons.append(f"MA50趨勢向下")
        elif ma50_adj > 0:
            sell_score -= abs(ma50_adj) * 0.5
            sell_score = max(0, sell_score)

        if is_macd_bearish:
            sell_score += sell_weights.get('macd_bearish', 15)
            reasons.append("MACD 死叉")

        if is_trending_down:
            sell_score += sell_weights.get('ma_bearish', 15)
            reasons.append("均線空頭排列")

        if bb_position > 90:
            sell_score += sell_weights.get('bb_upper', 10)
            reasons.append("接近布林帶上軌")

        # 強勢股保護
        is_strong_trend = (not is_macd_bearish) and (not is_trending_down) and (volume_ratio > 1.2)
        
        if volume_ratio > 2.5 and rsi < 80 and is_strong_trend:
            sell_score = 0
            reasons = ["🚀 超強勢突破信號!", "建議繼續持有"]
            signal_type = "HOLD"
        elif is_strong_trend and rsi > 70:
            sell_score = int(sell_score * 0.2)
            signal_type = "HOLD"
            reasons = ["RSI超買但強勢股特徵明顯", "建議持有或小幅減倉"]
        else:
            signal_type = "SELL"

        adjusted_strength = min(sell_score / 100, 1.0)
        suggested_sell_ratio = int(adjusted_strength * 100)

        recommendations = []
        if signal_type == "HOLD":
            recommendations.extend(reasons)
        else:
            recommendations.append(f"建議賣出比例: {suggested_sell_ratio}%")
            recommendations.append(f"建議價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
            recommendations.append(f"建議價格區間: NT${suggested_price_low:.2f} - NT${suggested_price_high:.2f}")
            for r in reasons[:3]:
                recommendations.append(f"✗ {r}")

        fmt.print_signal_box(signal_type, adjusted_strength, sell_score, recommendations)

    # ========== 處理持有信號 ==========
    else:
        signal_type = "HOLD"
        recommendations = [
            "市場觀望，暫不操作",
            f"關注支撐位: NT${bb_lower:.2f}",
            f"關注壓力位: NT${bb_upper:.2f}"
            f"關注支撐位: NT${bb_lower:.2f}",
            f"關注壓力位: NT${bb_upper:.2f}"
        ]
        fmt.print_signal_box(signal_type, 0, 50, recommendations)

    # ========== 加碼條件摘要 ==========
    add_checker.print_summary()

    # ========== 風險提示 ==========
    print("\n" + "═" * 60)
    print("  ⚠️ 風險提示")
    print("═" * 60)
    print("   • 本信號由 AI 模型生成，僅供參考")
    print("   • 股市有風險，投資需謹慎")
    print("   • 請根據自身風險承受能力做出決策")
    print("═" * 60)

    # ========== 快速摘要 ==========
    print("\n" + "╔" + "═" * 40 + "╗")
    print("║        📱 快速摘要                   ║")
    print("╠" + "═" * 40 + "╣")
    print(f"║  股票: 8064.TWO (東捷科技)                 ║")
    print(f"║  股票: 8064.TWO (東捷科技)                ║")
    print(f"║  日期: {latest_date}                   ║")
    print(f"║  價格: NT${current_price:.2f}                        ║")
    print(f"║  價格: NT${current_price:.2f}                     ║")
    
    if action_value > 0.1:
        if add_checker.can_add_position():
            print(f"║  信號: 🔥 可加碼 (三條件滿足)        ║")
        else:
            met = add_checker.get_conditions_met()
            print(f"║  信號: 買入 ({met}/3條件滿足)        ║")
    elif action_value < -0.1:
        print(f"║  信號: 賣出/持有                     ║")
    else:
        print(f"║  信號: 持有                          ║")
    
    print("╚" + "═" * 40 + "╝")

    return {
        'date': latest_date,
        'symbol': '8064.TWO',
        'current_price': current_price,
        'action_value': action_value,
        'can_add_position': add_checker.can_add_position(),
        'conditions_met': add_checker.get_conditions_met(),
        'ma50_rising': add_checker.conditions['ma50_rising'],
        'structure_complete': add_checker.conditions['structure_complete'],
        'volume_healthy': add_checker.conditions['volume_healthy'],
    }


# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    result = get_trading_signal()
    
    if result:
        print("\n✅ 信號生成成功!")

        # 生成 K 線圖
        try:
            import yfinance as yf
            chart_df = yf.Ticker("8064.TWO").history(period="6mo")
            chart_df.columns = [c.lower() for c in chart_df.columns]
            chart_path = "8064_chart.png"
            plot_candlestick(chart_df, "8064.TWO", save_path=chart_path)
        except Exception as e:
            print(f"   圖表生成失敗: {e}")

        
        if result['can_add_position']:
            print("\n🔥🔥🔥 重要: 三大加碼條件全部滿足! 🔥🔥🔥")
            print("   ✅ MA50 上升")
            print("   ✅ 結構型態完成") 
            print("   ✅ 量能健康")
    else:
        print("\n❌ 信號生成失敗")