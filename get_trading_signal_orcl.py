"""
美股 ORCL (Oracle) AI 交易信号生成器
======================================
使用训练好的 Hybrid PPO + XGBoost 模型生成今日交易策略
输出: 买入/卖出/持有 信号 + 建议价格
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces

# 导入动态权重计算器
from dynamic_signal_weights import DynamicWeightCalculator
# 导入增强评分模块
# 导入增强评分模块（含FinBERT情绪分析）
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
# 导入MA50斜率分析模块
from ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment
# 导入模型准确度追踪器
from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display, should_mute_ai_signal
from triangle_pattern import detect_triangle, triangle_breakout
from breakout_detector import get_breakout_signal
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
from shared_market_checks import evaluate_fundamentals_for_sell, calculate_obv as shared_calculate_obv, money_flow_strength as shared_money_flow_strength, calculate_growth_score_adjustment



# ==========================================
# 技术指标计算 (XGBoost 所需特征)
# ==========================================
def add_technical_indicators(df):
    """添加技术指标 - 包含 XGBoost 模型所需的所有特征"""
    # 均线
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
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
    df['macd_hist'] = df['macd'] - df['macd_signal']  # XGBoost 需要

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)  # XGBoost 需要

    # KD 指标 (XGBoost 需要)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV (能量潮指标)
    df = calculate_obv(df)

    # 波動性 (XGBoost 需要)
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # ATR (XGBoost 需要)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()

    # 價格變化率 (XGBoost 需要)
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    # MA50 斜率 (XGBoost 需要)
    df['ma50_slope'] = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100

    df = df.bfill().ffill()
    return df


# ==========================================
# 新增：资金流向分析函数
# ==========================================
def calculate_obv(df):
    """计算OBV (能量潮指标)"""
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_ma20'] = pd.Series(obv).rolling(20).mean()
    return df

def money_flow_strength(df):
    """分析资金流向强度 (进阶版: up/down volume + money flow)"""
    if len(df) < 20:
        return False, 1.0, {}

    # 量比
    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]

    # Up Volume vs Down Volume (基于 close vs open)
    recent = df.tail(30).copy()
    recent['up_volume'] = recent.apply(lambda x: x['volume'] if x['close'] > x['open'] else 0, axis=1)
    recent['down_volume'] = recent.apply(lambda x: x['volume'] if x['close'] <= x['open'] else 0, axis=1)
    up_vol = recent['up_volume'].sum()
    down_vol = recent['down_volume'].sum()
    up_down_ratio = up_vol / (down_vol + 1e-10)

    # Money Flow = volume * (close - open)
    recent['money_flow'] = recent['volume'] * (recent['close'] - recent['open'])
    net_money_flow_30d = recent['money_flow'].sum()
    net_money_flow_5d = recent['money_flow'].tail(5).sum()

    # Capital inflow: 放量 + 收阳
    latest = df.iloc[-1]
    capital_inflow = (volume_ratio > 1.5) and (latest['close'] > latest['open'])

    # OBV
    obv_now = df['obv'].iloc[-1] if 'obv' in df.columns else 0
    obv_ma = df['obv_ma20'].iloc[-1] if 'obv_ma20' in df.columns else 0
    obv_bullish = obv_now > obv_ma

    # 综合判断
    strong_money = (
        (capital_inflow or (up_down_ratio > 1.3 and volume_ratio > 1.0)) and
        obv_bullish
    )

    details = {
        'up_volume_30d': int(up_vol),
        'down_volume_30d': int(down_vol),
        'up_down_ratio': round(up_down_ratio, 2),
        'net_money_flow_30d': net_money_flow_30d,
        'net_money_flow_5d': net_money_flow_5d,
        'capital_inflow': capital_inflow,
        'obv_bullish': obv_bullish,
    }

    return strong_money, volume_ratio, details

def detect_memory_cycle_phase(df):
    """检测内存周期阶段（适用于芯片股）"""
    if len(df) < 200:
        return "NEUTRAL"

    ma50 = df['sma_50']
    ma200 = df['sma_200']
    price = df['close'].iloc[-1]

    # 週期初升段
    early_upcycle = (
        price > ma50.iloc[-1] and
        ma50.iloc[-1] > ma200.iloc[-1] and
        ma50.diff().iloc[-1] > 0
    )

    # 高檔末升段
    late_cycle = (
        price > ma50.iloc[-1] * 1.25 and
        ma50.diff().iloc[-1] < ma50.diff().iloc[-5] if len(df) >= 5 else False
    )

    if early_upcycle:
        return "EARLY_UPCYCLE"   # 🔥最會噴的階段
    elif late_cycle:
        return "LATE_CYCLE"
    else:
        return "NEUTRAL"

def trend_acceleration(df):
    """检测趋势加速"""
    if len(df) < 30:
        return False

    sma10 = df['sma_10']
    sma30 = df['sma_30']
    slope10 = sma10.diff().iloc[-1]
    slope30 = sma30.diff().iloc[-1]
    price = df['close'].iloc[-1]

    accelerating = (
        slope10 > slope30 and
        slope10 > 0 and
        price > sma10.iloc[-1]
    )
    return accelerating

def explosive_trend_filter(df):
    """爆发行情过滤器"""
    strong_money, vol_ratio, flow_details = money_flow_strength(df)
    cycle_phase = detect_memory_cycle_phase(df)
    accelerating = trend_acceleration(df)

    explosive = (
        strong_money and
        accelerating and
        cycle_phase == "EARLY_UPCYCLE"
    )

    return {
        "explosive": explosive,
        "volume_ratio": vol_ratio,
        "cycle_phase": cycle_phase,
        "money_inflow": strong_money,
        "trend_accelerating": accelerating,
        "flow_details": flow_details
    ,
        'explosion_detected': explosion['explosive'] if 'explosion' in locals() else False
    }


# 使用共享版資金流邏輯，確保所有股票一致
calculate_obv = shared_calculate_obv
money_flow_strength = shared_money_flow_strength

# ==========================================
# 交易信号生成
# ==========================================
def get_trading_signal():
    """生成今日交易信号"""
    print("=" * 80)
    print("🤖 美股 ORCL AI HYBRID PPO + XGBoost 交易信号生成器")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 顯示AI模型準確度
    accuracy_display = get_model_accuracy_display('ORCL')
    print(f"模型準確度: {accuracy_display}")
    print("=" * 80)

    # 1. 加载模型 (XGBoost + PPO Hybrid)
    print(f"\n📦 加载 Hybrid 模型: XGBoost + PPO")
    try:
        model = joblib.load("xgb_orcl_model.pkl")
        print("✅ XGBoost 模型加载成功!")
    except Exception as e:
        print(f"❌ XGBoost 模型加载失败: {e}")
        return None

    ppo_model = None
    try:
        ppo_model = PPO.load("ppo_orcl_improved")
        print("✅ PPO 模型加载成功!")
    except Exception as e:
        print(f"⚠️  PPO 模型加载失败 (将退化为纯XGB): {e}")

    # 2. 下载最新数据 (使用 period 方式获取更新的数据)
    print("\n📊 下载最新市场数据...")
    try:
        import yfinance as yf

        # 使用 period 方式下载，auto_adjust=True 确保价格是最新的
        df = yf.download('ORCL', period='300d', progress=False, auto_adjust=True)  # 改为300天以计算200日均线

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
            ticker_info = yf.Ticker('ORCL').info
            target_price = ticker_info.get('targetMeanPrice')
            target_high = ticker_info.get('targetHighPrice')
            target_low = ticker_info.get('targetLowPrice')
            num_analysts = ticker_info.get('numberOfAnalystOpinions', 0)
            recommendation_mean = ticker_info.get('recommendationMean')  # 1=Strong Buy, 5=Sell
            recommendation_key = ticker_info.get('recommendationKey', '')

            if target_price and num_analysts > 0:
                print(f"   📊 分析師目標價: ${target_price:.2f} (平均) / ${target_high:.2f} (最高)")
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
    print(f"   当前价格: ${float(latest_data['close']):.2f}")
    print(f"   今日成交量: {int(latest_data['volume']):,}")

    # 4. 准备 XGBoost 模型特征
    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'obv', 'obv_ma20',
        'sma_10', 'sma_30', 'sma_50', 'sma_200',
        'volatility', 'atr',
        'price_change_5d', 'price_change_10d', 'price_change_20d',
        'ma50_slope'
    ]

    # 5a. XGBoost 推理
    print("\n🧠 XGBoost 模型分析中...")
    latest_features = df[feature_columns].iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    prediction_proba = model.predict_proba(latest_features)[0]

    if prediction == 1:
        xgb_action = float(prediction_proba[1] * 2 - 1)
        xgb_action = max(0.1, xgb_action)
    else:
        xgb_action = -float(prediction_proba[0] * 2 - 1)
        xgb_action = min(-0.1, xgb_action)

    print(f"   XGBoost 预测: {'买入' if prediction == 1 else '不买/观望'}")
    print(f"   买入概率: {prediction_proba[1]*100:.2f}%")
    print(f"   XGB Action: {xgb_action:+.4f}")

    # 5b. PPO 推理
    ppo_action = 0.0
    if ppo_model is not None:
        print("\n🤖 PPO 模型分析中...")
        try:
            p = float(latest_data['close'])
            obs = np.array([
                0, 100000, p,
                float(latest_data.get('sma_10', p)),
                float(latest_data.get('sma_30', p)),
                float(latest_data.get('sma_50', p)),
                float(latest_data.get('rsi', 50)),
                float(latest_data.get('macd', 0)),
                float(latest_data.get('macd_signal', 0)),
                float(latest_data.get('bb_upper', p)),
                float(latest_data.get('bb_lower', p)),
                float(latest_data.get('volume', 0)),
                0, 1.0, 1.0,
            ], dtype=np.float32)
            ppo_act, _ = ppo_model.predict(obs, deterministic=True)
            ppo_action = float(ppo_act[0])
            ppo_signal = 'BUY' if ppo_action > 0.3 else ('SELL' if ppo_action < -0.3 else 'HOLD')
            from ppo_backtest_cache import format_ppo_roi_line
            print(format_ppo_roi_line('ORCL', 'ORCL', 'ppo_orcl_improved', df, ppo_action))
        except Exception as e:
            print(f"   ⚠️  PPO 推理失败: {e}")

    # 5c. Hybrid 融合 (60% XGB + 40% PPO)
    if ppo_model is not None:
        action_value = 0.6 * xgb_action + 0.4 * ppo_action
        print(f"\n🔀 Hybrid Action: {action_value:+.4f}  (XGB×0.6 + PPO×0.4)")
    else:
        action_value = xgb_action
        print(f"\n🔀 Action Value (XGB only): {action_value:+.4f}")

    ai_muted = should_mute_ai_signal('ORCL', threshold=52)
    if ai_muted:
        print("⚠️  AI模型準確度低於52%，已靜音AI交易動作（action=0）")
        action_value = 0.0

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
    print(f"RSI (10):        {rsi:.2f}  {'[超买]' if rsi > 92 else '[超卖]' if rsi < 15 else '[中性]'}")
    print(f"MACD:            {macd:.4f}")
    print(f"MACD Signal:     {macd_signal:.4f}  {'[金叉]' if macd > macd_signal else '[死叉]'}")
    print(f"SMA 10:          ${sma_10:.2f}")
    print(f"SMA 30:          ${sma_30:.2f}  {'[多头]' if sma_10 > sma_30 else '[空头]'}")
    print(f"SMA 50:          ${sma_30:.2f}")
    print(f"布林带上轨:      ${bb_upper:.2f}")
    print(f"布林带下轨:      ${bb_lower:.2f}")
    print(f"当前价格位置:    {((current_price - bb_lower) / (bb_upper - bb_lower) * 100):.1f}% {'⚡ 上軌外延' if current_price > bb_upper else ('⚠️ 下軌外延' if current_price < bb_lower else '布林带内')}")
    print(f"成交量:          {int(current_volume):,}")
    print(f"20日平均量:      {int(avg_volume_20):,}  {'[放量]' if volume_ratio > 1.5 else '[缩量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"量比:            {volume_ratio:.2f}x")

    # 基本面獲利檢查 (供 MA50 斜率模組使用)
    try:
        profit_positive = evaluate_fundamentals_for_sell(yf, 'ORCL')['good']
    except Exception:
        profit_positive = None

    # 7.1 計算MA50斜率
    print("\n" + "=" * 80)
    print("📈 MA50趨勢分析")
    print("=" * 80)
    ma50_slope_info = calculate_ma50_slope(df['close'], window=50, slope_period=5, current_rsi=rsi, profit_positive=profit_positive)
    print(f"當前MA50:        ${ma50_slope_info['ma50_current']:.2f}")
    print(f"MA50斜率:        {ma50_slope_info['slope']:+.6f}")
    print(f"斜率百分比:      {ma50_slope_info['slope_pct']:+.4f}%")
    print(f"趨勢判斷:        {ma50_slope_info['color']} {ma50_slope_info['trend']}")
    print(f"交易信號:        {ma50_slope_info['signal']}")
    # print(f"\n💡 說明: {ma50_slope_info['description']}")

    # 7. 初始化动态权重计算器
    weight_calc = DynamicWeightCalculator('ORCL')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

    # 7.5 获取市场情绪分析（FinBERT + VADER）
    # print("\n" + "=" * 80)
    # print("📰 市场情绪分析 (FinBERT NLP Engine)")
    # print("=" * 80)

    from finbert_enhanced_scoring import calculate_sentiment_score, format_sentiment_output
    sentiment_result = calculate_sentiment_score('ORCL', verbose=True)

    if sentiment_result and sentiment_result['news_count'] > 0:
        pass  # 情緒分析結果已計算，輸出已移至後續統一顯示
    else:
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}




    # 蠟燭圖型態分析
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))

        # 獲取型態評分調整
        pattern_adjustment = get_pattern_score_adjustment(patterns)
        print(f"\n型態評分調整: {pattern_adjustment:+.1f} 分")
    except Exception as e:
        print(f"   ⚠️  型態分析失敗: {e}")
        pattern_adjustment = 0

    # 爆发行情检测（主升段分析）
    print("\n" + "=" * 80)
    print("🚀 爆发行情检测 (主升段分析)")
    print("=" * 80)

    explosion = explosive_trend_filter(df)
    flow = explosion.get('flow_details', {})
    print(f"资金流入状态: {'✅ 强势' if explosion['money_inflow'] else '❌ 弱势'}")
    if flow:
        print(f"   Up Volume (30d):   {flow.get('up_volume_30d', 0):,}")
        print(f"   Down Volume (30d): {flow.get('down_volume_30d', 0):,}")
        print(f"   Up/Down Ratio:     {flow.get('up_down_ratio', 0):.2f}x  {'[多方主导]' if flow.get('up_down_ratio', 0) > 1.3 else '[空方主导]' if flow.get('up_down_ratio', 0) < 0.7 else '[均衡]'}")
        mf_30 = flow.get('net_money_flow_30d', 0)
        mf_5 = flow.get('net_money_flow_5d', 0)
        print(f"   Net Money Flow(30d): {'+'if mf_30>0 else ''}{mf_30:,.0f}  {'[净流入]' if mf_30 > 0 else '[净流出]'}")
        print(f"   Net Money Flow(5d):  {'+'if mf_5>0 else ''}{mf_5:,.0f}  {'[近期流入]' if mf_5 > 0 else '[近期流出]'}")
        print(f"   即时资金流入:  {'✅ 放量收阳' if flow.get('capital_inflow') else '❌ 未确认'}")
        print(f"   obv trend:    {'bullish' if flow.get('obv_bullish') else 'bearish'}")
    print(f"趋势加速状态: {'✅ 加速中' if explosion['trend_accelerating'] else '❌ 减速中'}")
    print(f"周期阶段: {explosion['cycle_phase']}")
    print(f"量比: {explosion['volume_ratio']:.2f}x")

    if explosion["explosive"]:
        print("\n🚀 主升段爆发行情侦测!")
        print("📌 爆发行情特征:")
        print("   • 资金强势流入 (OBV > 20日均线)")
        print("   • 趋势加速 (10日均线斜率 > 30日均线斜率)")
        print("   • 处于周期初升段 (EARLY_UPCYCLE)")
        print("   • 量能放大 (量比 > 1.3x)")

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
            symbol='ORCL'
        )

        # 加入MA50斜率評分調整
        ma50_slope_adjustment = get_ma50_slope_score_adjustment(ma50_slope_info)
        buy_score += ma50_slope_adjustment

        # ==========================================
        # 🎯 ORCL 专属策略增强 (基于相关性与机构资金)
        # ==========================================
        print("\n" + "=" * 80)
        print("🎯 ORCL Oracle 专属策略分析")
        print("=" * 80)

        # 1303相关性检查已移除 - 与台湾DRAM厂商无直接关联

        # 2. OBV机构资金确认 (权值股特有)
        obv_current = df['obv'].iloc[-1] if 'obv' in df.columns else 0
        obv_ma20 = df['obv_ma20'].iloc[-1] if 'obv_ma20' in df.columns else 0
        obv_5d_avg = df['obv'].tail(5).mean() if 'obv' in df.columns and len(df) >= 5 else 0

        institutional_inflow = False
        if obv_current > obv_ma20 and obv_5d_avg > obv_ma20:
            institutional_inflow = True
            buy_score += 20
            buy_reasons.append("💰 OBV持续向上，机构资金进驻确认")

        print(f"\nOBV 机构资金分析 (权值股关键指标):")
        print(f"  当前OBV:     {obv_current:,.0f}")
        print(f"  OBV MA20:    {obv_ma20:,.0f}")
        print(f"  5日均值:     {obv_5d_avg:,.0f}")
        print(f"  机构进驻:    {'✅ 是 (OBV持续突破MA20)' if institutional_inflow else '❌ 否 (等待OBV确认)'}")

        # 3. 目标价位区间
        target_zone_low = 55.0
        target_zone_high = 60.0
        distance_to_target = ((target_zone_low - current_price) / current_price) * 100

        print(f"\n目标价位分析:")
        print(f"  当前价格:     ${current_price:.2f}")
        print(f"  第一目标区:   ${target_zone_low:.2f} - ${target_zone_high:.2f} (分析师中值)")
        print(f"  潜在空间:     {distance_to_target:+.1f}% {'(已达标)' if distance_to_target <= 0 else '(上涨空间)'}")

        # 如果已接近或超过目标区，降低买入评分
        if current_price >= target_zone_high:
            buy_score -= 15
            buy_warnings.append(f"⚠️ 价格已达第一目标区上限 ${target_zone_high}, 建议获利了结")
        elif current_price >= target_zone_low:
            buy_score -= 5
            buy_warnings.append(f"价格已进入目标区 ${target_zone_low}-{target_zone_high}, 注意压力")

        print("\n" + "=" * 80)
        # ==========================================



        # 加入爆发行情评分调整
        if explosion["explosive"]:
            buy_score += 25  # 爆发行情额外加分
            buy_reasons.append(f"🚀 爆发行情确认: 主升段初期")
            buy_reasons.append(f"资金强势流入 (OBV > MA20)")
            buy_reasons.append(f"趋势加速 (10日斜率 > 30日斜率)")
# 加入爆发行情评分调整
        if explosion["explosive"]:
            buy_score += 25  # 爆发行情额外加分
            buy_reasons.append(f"🚀 爆发行情确认: 主升段初期")
            buy_reasons.append(f"资金强势流入 (OBV > MA20)")
            buy_reasons.append(f"趋势加速 (10日斜率 > 30日斜率)")
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


        buy_score = max(0, min(100, buy_score))  # 限制在0-100之間

        if ma50_slope_adjustment > 0:
            buy_reasons.append(f"MA50趨勢向上 (+{ma50_slope_adjustment}分)")
        elif ma50_slope_adjustment < 0:
            buy_warnings.append(f"MA50趨勢向下 ({ma50_slope_adjustment}分)")


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
            print(f"   建议买入价格区间: ${suggested_price_low:.2f} - ${suggested_price_high:.2f}")

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
            print(f"      • 关注支撑位: ${(bb_upper if current_price > bb_upper else bb_lower):.2f}{" (前上軌→支撐)" if current_price > bb_upper else ""}")
        elif buy_score >= 60:
            print(f"      • 多个买入信号确认,可以买入")
            print(f"      • 分批买入,建议买入 {suggested_buy_ratio}%")
            print(f"      • 设置止损: ${current_price * 0.95:.2f} (-5%)")
        else:
            print(f"      • 谨慎买入 {suggested_buy_ratio}%")
            print(f"      • 等待量能确认后再加仓")
            print(f"      • 设置止损: ${current_price * 0.95:.2f} (-5%)")

    elif action_value < -0.1:

        # 先检查是否为爆发行情，如果是则覆盖卖出信号
        if explosion["explosive"]:
            signal = "强势持有 (HOLD - TREND EXPLOSION)"
            signal_emoji = "🚀"
            strength = abs(action_value)
            suggested_price_low = current_price
            suggested_price_high = current_price

            print("\n🚀 主升段爆发行情侦测!")
            print(f"资金流入: {explosion['money_inflow']}")
            print(f"趋势加速: {explosion['trend_accelerating']}")
            print(f"周期位置: {explosion['cycle_phase']}")
            print(f"量比: {explosion['volume_ratio']:.2f}x")

            print("\n📌 操作策略:")
            print("   • 不卖出 (主升段爆发行情)")
            print("   • 回调不破均线继续抱")
            print("   • 使用追踪止损代替固定止损")
            print("   • 关注 OBV 资金流向指标")
            print("   • 设置移动止盈: 跌破 10 日均线减半仓")

            # 跳过卖出评分逻辑
            skip_sell_scoring = True
        else:
            skip_sell_scoring = False

        # 🔥 计算技术指标（需要在条件块外定义，以便后续使用）


        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50


        is_macd_bearish = macd < macd_signal


        is_trending_down = sma_10 < sma_30



        if not skip_sell_scoring:
            signal = "卖出 (SELL)"
            signal_emoji = "🔴"
            strength = abs(action_value)
            suggested_price_low = current_price * 1.000
            suggested_price_high = current_price * 1.005

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
                    reasons.append(f"   最高目標價 ${target_high:.2f} (上漲空間 {target_upside_high:.1f}%)")
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

        # 🔥 情景B: 普通强势股 (量比1.2-2.5)
        # 回测: 39.3%继续涨 vs 28.6%回调
        elif is_strong_trend and rsi > 70:
            sell_score = int(sell_score * 0.2)  # 评分打2折（更激进）
            reasons.clear()
            reasons.append(f"⚠️  RSI超买但符合强势股特征")
            reasons.append(f"MACD金叉 + 均线多头 + 放量({volume_ratio:.1f}x)")
            reasons.append(f"回测: 继续涨39.3% vs 回调28.6%")
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
            fundamentals = evaluate_fundamentals_for_sell(yf, 'ORCL')
            fundamentals_good_sell = fundamentals['good']
            fundamentals_bad_sell = fundamentals['bad']
            profit_margin_sell = fundamentals['profit_margin']

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
        # 🌱 基本面成長保護: Revenue Growth > 33% 或 EPS Growth > 100% 各降低賣出強度 8 分
        _growth = calculate_growth_score_adjustment(yf, 'ORCL')
        if _growth['adjustment'] > 0 and sell_score > 0:
            sell_score = max(0, sell_score - _growth['adjustment'])
            for _gr in _growth['reasons']:
                reasons.append(f'🌱 {_gr}')
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
            print(f"   建议卖出价格区间: ${suggested_price_low:.2f} - ${suggested_price_high:.2f}")

        if reasons:
            if sell_score == 0:
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
            print(f"      • 设置止损: ${current_price * 0.97:.2f} (-3%)")
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
        print(f"      • 关注支撑位: ${(bb_upper if current_price > bb_upper else bb_lower):.2f}{" (前上軌→支撐)" if current_price > bb_upper else ""}")
        print(f"      • 关注压力位: ${(current_price * 1.05 if current_price > bb_upper else bb_upper):.2f}{" ↑突破後估算" if current_price > bb_upper else ""}")
    # 8. 风险提示
    print("\n" + "=" * 80)

    return {
        'date': latest_date,
        'symbol': 'ORCL',
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
        print(f"\n📱 快速摘要:")
        print(f"   股票: {result['symbol']}")
        print(f"   日期: {result['date']}")
        print(f"   价格: ${result['current_price']:.2f}")
        print(f"   信号: {result['signal']}")
        if result['strength'] > 0:
            print(f"   强度: {result['strength']:.2f}")



        # 爆发行情警告
        if result.get('explosion_detected', False):
            print("\n" + "=" * 80)
            print("🚀 主升段爆发行情侦测")
            print("=" * 80)
            print("📌 爆发行情特征:")
            print("   • 资金强势流入 (OBV > 20日均线)")
            print("   • 趋势加速 (10日均线斜率 > 30日均线斜率)")
            print("   • 处于周期初升段 (EARLY_UPCYCLE)")
            print("   • 量能放大 (量比 > 1.3x)")
            if result.get('explosion_data'):
                exp_data = result['explosion_data']
                print(f"\n🔥 爆发行情数据:")
                print(f"   量比: {exp_data['volume_ratio']:.2f}x")
                print(f"   周期阶段: {exp_data['cycle_phase']}")
                print(f"   资金流入: {'是' if exp_data['money_inflow'] else '否'}")
                print(f"   趋势加速: {'是' if exp_data['trend_accelerating'] else '否'}")
            print("=" * 80)

        # 顯示AI模型準確度摘要
        print(f"   {get_model_accuracy_display('ORCL')}")
    else:
        print("\n❌ 信号生成失败")



