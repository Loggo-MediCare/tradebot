"""
台股 mu (Micron) AI 交易信号生成器
======================================
 信号: 卖出 (SELL) 20251222
   强度: 1.00 / 1.00
   建议卖出比例: 100%
   建议卖出价格区间: $180.00 - $180.90

   📌 卖出理由:
      1. RSI 偏高,有回调风险
      2. 价格高于短期均线,可能回调

   💡 操作建议:
      • 分批卖出,避免一次性清仓
      • 保留部分仓位应对反弹

================================================================================
⚠️  风险提示
================================================================================
   • 本信号由 AI 模型生成,仅供参考,不构成投资建议
   • 股市有风险,投资需谨慎
   • 请根据自身风险承受能力做出投资决策
   • 建议结合其他分析方法综合判断
================================================================================

✅ 信号生成成功!

📱 快速摘要:
   股票: MU (Micron)
   日期: 2025-12-22
   价格: $180.00
   信号: 卖出 (SELL)
   强度: 1.00

使用训练好的 PPO 模型生成今日交易策略
输出: 买入/卖出/持有 信号 + 建议价格
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
from stable_baselines3 import PPO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入动态权重计算器
from dynamic_signal_weights import DynamicWeightCalculator

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
    print("🤖 台股 mu (Micron) AI 交易信号生成器")
    print("=" * 80)
   # print(f("生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. 加载模型
    model_path = r"C:\Users\Silvi\Projects\trading-bot\ppo_mu_improved.zip"
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
        df = yf.download('MU', period='90d', progress=False, auto_adjust=True)

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
    print(f"SMA 10:          ${sma_10:.2f}")
    print(f"SMA 30:          ${sma_30:.2f}  {'[多头]' if sma_10 > sma_30 else '[空头]'}")
    print(f"布林带上轨:      ${bb_upper:.2f}")
    print(f"布林带下轨:      ${bb_lower:.2f}")
    print(f"当前价格位置:    {((current_price - bb_lower) / (bb_upper - bb_lower) * 100):.1f}% (布林带内)")
    print(f"成交量:          {int(current_volume):,}")
    print(f"20日平均量:      {int(avg_volume_20):,}  {'[放量]' if volume_ratio > 1.5 else '[缩量]' if volume_ratio < 0.7 else '[正常]'}")
    print(f"量比:            {volume_ratio:.2f}x")

    # 7. 初始化动态权重计算器
    weight_calc = DynamicWeightCalculator('MU')
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

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
        buy_score = 0
        warnings = []
        reasons = []

        # 🚫 MACD死叉 = 直接拒绝买入！
        if macd < macd_signal:
            buy_score = -100  # 死叉直接负分
            warnings.append(f"⚠️  MACD死叉,趋势转弱,不应买入!")
        else:
            # 技术指标评分 (使用动态权重)
            if rsi < 30:
                buy_score += buy_weights['rsi_oversold']
                reasons.append(f"RSI超卖 ({rsi:.1f} < 30)")
            elif rsi < 50:
                buy_score += buy_weights['rsi_low']
                reasons.append(f"RSI偏低 ({rsi:.1f})")

            if macd > macd_signal and macd > 0:
                buy_score += buy_weights['macd_bullish_strong']
                reasons.append("MACD金叉且为正值")
            elif macd > macd_signal:
                buy_score += buy_weights['macd_bullish']
                reasons.append("MACD金叉")

            if sma_10 > sma_30:
                buy_score += buy_weights['ma_bullish']
                reasons.append("均线多头排列")

            # 均线位置
            if current_price < sma_10:
                buy_score += buy_weights['price_below_ma']
                reasons.append("价格低于短期均线")

        # 🔥 成交量判断（最关键！提高权重）
        if volume_ratio < 0.5:
            buy_score -= 50  # 极度缩量,严重扣分
            warnings.append(f"⚠️  成交量严重不足(量比{volume_ratio:.1f}x)，买盘极弱")
        elif volume_ratio < 0.7:
            buy_score -= 35  # 缩量扣分
            warnings.append(f"⚠️  成交量不足(量比{volume_ratio:.1f}x)，买盘无力")
        elif volume_ratio > 2.0:
            buy_score += 35  # 极度放量,大幅加分
            reasons.append(f"极度放量突破(量比{volume_ratio:.1f}x)!")
        elif volume_ratio > 1.5:
            buy_score += 30  # 放量
            reasons.append(f"放量突破(量比{volume_ratio:.1f}x)")
        elif volume_ratio > 1.2:
            buy_score += 20  # 温和放量
            reasons.append(f"温和放量(量比{volume_ratio:.1f}x)")

        # 🚀 强势突破加成（RSI>70 但符合强势特征 = 追涨！）
        is_strong_breakout = (macd > macd_signal) and (sma_10 > sma_30) and (volume_ratio > 1.5)

        # 🛡️ 超跌反弹加成（低位 + RSI低 + 开始放量 = 反弹机会！）
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        is_oversold_bounce = (bb_position < 30) and (rsi < 40) and (volume_ratio > 1.2)

        # 🏢 获取公司基本面数据（用于判断超跌是机会还是陷阱）
        fundamentals_good = False
        fundamentals_bad = False
        is_large_cap = False  # 大型優質公司 (營收>35B)
        profit_margin = 0
        net_income = 0
        revenue_growth = 0
        total_revenue = 0

        try:
            ticker_obj = yf.Ticker('MU')
            info = ticker_obj.info

            # 获取财务数据
            net_income = info.get('netIncome', None)
            profit_margin = info.get('profitMargins', None)
            revenue_growth = info.get('revenueGrowth', None)
            total_revenue = info.get('totalRevenue', None)  # 年營收

            # 判断是否为大型优质公司（年营收>35B USD）
            if total_revenue is not None and total_revenue > 35_000_000_000:
                is_large_cap = True

            # 判断基本面好坏（优先使用利润率，因为更可靠）
            if profit_margin is not None and profit_margin > 0.10:  # 利潤率>10%
                fundamentals_good = True
            elif profit_margin is not None and profit_margin < 0:  # 利潤率為負
                fundamentals_bad = True
            elif net_income is not None and net_income > 0:  # 有淨利數據且>0
                fundamentals_good = True
            elif net_income is not None and net_income < 0:  # 淨利為負
                fundamentals_bad = True
        except:
            pass  # 如果無法獲取數據，跳過基本面判斷

        if is_oversold_bounce:
            # 超跌反弹 - 根据基本面判断是机会还是陷阱
            if fundamentals_good:
                # 优质公司超跌 = 黄金买点！
                base_score = 50

                # 🏢 大型优质公司额外加分 (营收>35B)
                if is_large_cap:
                    base_score += 20  # 大型公司额外+20分
                    buy_score += base_score
                    reasons.append(f"🏢 大型优质公司超跌! 布林带{bb_position:.1f}% + RSI{rsi:.1f}")
                    if total_revenue > 0:
                        reasons.append(f"年營收 ${total_revenue/1_000_000_000:.1f}B, 淨利率{profit_margin*100:.1f}%")
                    else:
                        reasons.append(f"淨利率{profit_margin*100:.1f}% (健康)")
                    reasons.append(f"放量({volume_ratio:.1f}x) - 機構級投資機會!")
                    reasons.append(f"💎 巴菲特式價值投資: 大型績優股超跌!")
                else:
                    buy_score += base_score
                    reasons.append(f"🎯 优质公司超跌! 布林带{bb_position:.1f}% + RSI{rsi:.1f}")
                    if profit_margin is not None and profit_margin > 0:
                        reasons.append(f"淨利率{profit_margin*100:.1f}% (健康), 放量({volume_ratio:.1f}x)")
                    else:
                        reasons.append(f"公司盈利良好, 放量({volume_ratio:.1f}x)")
                    reasons.append(f"💎 价值投资机会: 低价买入好公司!")
            elif fundamentals_bad:
                # 亏损公司超跌 = 继续下跌风险
                buy_score -= 30  # 扣分
                warnings.append(f"⚠️ 公司亏损,超跌可能持续")
                warnings.append(f"淨利为负,不是抄底机会")
            else:
                # 无法获取基本面,使用原逻辑
                buy_score += 35
                reasons.append(f"🛡️ 超跌反弹! 布林带{bb_position:.1f}% + RSI{rsi:.1f} + 放量({volume_ratio:.1f}x)")
                reasons.append(f"接近支撑位,反弹机会大")
        elif rsi > 70 and is_strong_breakout:
            # RSI虽高,但是强势突破,大幅加分!
            buy_score += 40
            reasons.append(f"🚀 强势突破! RSI{rsi:.1f} + MACD金叉 + 多头 + 放量({volume_ratio:.1f}x)")
            reasons.append(f"技术面多头排列,适合追涨")
        elif rsi > 70 and not is_strong_breakout:
            # RSI高但没有强势特征,扣分
            buy_score -= 20
            warnings.append(f"⚠️  RSI过高({rsi:.1f})但缺乏强势特征,风险较大")

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
            print(f"      • 关注支撑位: ${bb_lower:.2f}")
        elif buy_score >= 60:
            print(f"      • 多个买入信号确认,可以买入")
            print(f"      • 分批买入,建议买入 {suggested_buy_ratio}%")
            print(f"      • 设置止损: ${current_price * 0.95:.2f} (-5%)")
        else:
            print(f"      • 谨慎买入 {suggested_buy_ratio}%")
            print(f"      • 等待量能确认后再加仓")
            print(f"      • 设置止损: ${current_price * 0.95:.2f} (-5%)")

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

        # 1. RSI 超买判断 (使用动态权重)
        if rsi > 80:
            sell_score += sell_weights['rsi_severe']
            reasons.append(f"RSI 严重超买 ({rsi:.1f} > 80)")
        elif rsi > 70:
            sell_score += sell_weights['rsi_high']
            reasons.append(f"RSI 超买 ({rsi:.1f} > 70)")
        elif rsi > 65:
            sell_score += sell_weights['rsi_mild']
            reasons.append(f"RSI 偏高 ({rsi:.1f})")

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

        # 5. 价格远高于均线 (使用动态权重)
        price_vs_sma10 = ((current_price - sma_10) / sma_10) * 100
        if price_vs_sma10 > 10:
            sell_score += sell_weights['price_vs_ma_high']
            reasons.append(f"价格远高于10日均线 (+{price_vs_sma10:.1f}%)")
        elif price_vs_sma10 > 5:
            sell_score += sell_weights['price_vs_ma_mild']
            reasons.append(f"价格高于10日均线 (+{price_vs_sma10:.1f}%)")

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
            fundamentals_good_sell = False
            fundamentals_bad_sell = False
            profit_margin_sell = 0

            try:
                ticker_obj = yf.Ticker('MU')
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
        print(f"      • 关注支撑位: ${bb_lower:.2f}")
        print(f"      • 关注压力位: ${bb_upper:.2f}")

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
        'symbol': 'MU',
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
        print(f"\n📱 快速摘要:")
        print(f"   股票: {result['symbol']} (Micron)")
        print(f"   日期: {result['date']}")
        print(f"   价格: ${result['current_price']:.2f}")
        print(f"   信号: {result['signal']}")
        if result['strength'] > 0:
            print(f"   强度: {result['strength']:.2f}")
    else:
        print("\n❌ 信号生成失败")
