"""
增强版交易信号评分模块
========================
用于替换 get_trading_signal_*.py 中的评分逻辑

改进点:
1. 强势股识别 (RSI超买 + 多头确认 → 不扣分)
2. 组合信号识别 (MACD金叉 + 均线多头 + 放量 → 加分)
3. 智能区分"真超买"和"强势突破"
"""


def identify_strong_stock_pattern(rsi, macd, macd_signal, sma_10, sma_30, volume_ratio, bb_position):
    """
    识别强势股模式
    ================
    条件: RSI > 70 但同时满足:
    - MACD 金叉 且为正值
    - 均线多头排列
    - 放量 (> 1.2x)

    返回: (is_strong, strength_score, reasons)
    """
    # 检查强势股条件
    is_strong = (
        rsi > 70 and                    # RSI 超买
        macd > macd_signal and         # MACD 金叉
        macd > 0 and                   # MACD 为正
        sma_10 > sma_30 and            # 均线多头
        volume_ratio > 1.2             # 放量
    )

    if not is_strong:
        return False, 0, []

    # 计算强势程度 (0-100)
    strength = 0
    reasons = []

    # 1. RSI 强度评分
    if 70 < rsi <= 75:
        strength += 25
        reasons.append(f"⭐ RSI强势({rsi:.1f}) - 健康的超买")
    elif 75 < rsi <= 80:
        strength += 30
        reasons.append(f"⭐⭐ RSI超买({rsi:.1f}) - 强劲动能")
    elif 80 < rsi <= 85:
        strength += 25  # 开始降低
        reasons.append(f"⭐⭐ RSI严重超买({rsi:.1f}) - 短期风险增加")
    else:  # rsi > 85
        strength += 15
        reasons.append(f"⚠️  RSI极度超买({rsi:.1f}) - 注意回调风险")

    # 2. 成交量强度评分
    if volume_ratio > 3.0:
        strength += 35
        reasons.append(f"🔥 爆量突破({volume_ratio:.1f}x) - 资金大量涌入")
    elif volume_ratio > 2.0:
        strength += 30
        reasons.append(f"📈 大量放量({volume_ratio:.1f}x) - 买盘强劲")
    elif volume_ratio > 1.5:
        strength += 20
        reasons.append(f"📊 温和放量({volume_ratio:.1f}x) - 量能配合")
    else:  # 1.2 < volume_ratio <= 1.5
        strength += 10
        reasons.append(f"📊 略微放量({volume_ratio:.1f}x)")

    # 3. MACD 强度评分
    macd_strength = abs(macd - macd_signal)
    if macd_strength > 2.0:
        strength += 25
        reasons.append(f"💪 MACD金叉强劲(差距{macd_strength:.2f})")
    elif macd_strength > 1.0:
        strength += 20
        reasons.append(f"💪 MACD金叉明确(差距{macd_strength:.2f})")
    else:
        strength += 10
        reasons.append(f"💪 MACD金叉(差距{macd_strength:.2f})")

    # 4. 均线排列评分
    ma_gap_pct = ((sma_10 - sma_30) / sma_30) * 100
    if ma_gap_pct > 10:
        strength += 20
        reasons.append(f"📈 均线强势多头({ma_gap_pct:+.1f}%)")
    elif ma_gap_pct > 5:
        strength += 15
        reasons.append(f"📈 均线多头排列({ma_gap_pct:+.1f}%)")
    else:
        strength += 10
        reasons.append(f"📈 均线轻微多头({ma_gap_pct:+.1f}%)")

    return True, min(strength, 100), reasons


def calculate_bullish_combo_score(rsi, macd, macd_signal, sma_10, sma_30, volume_ratio, bb_position, current_price=None):
    """
    计算多头组合分数
    ==================
    经典多头组合:
    - MACD 金叉
    - 均线多头
    - 适当放量
    - 布林带健康位置
    - 价格流动性加成 (菜篮族经济学)

    返回: (combo_score, reasons)
    """
    combo_score = 0
    reasons = []

    # 1. MACD 金叉评分 (最高 30分)
    if macd > macd_signal and macd > 0:
        combo_score += 30
        reasons.append("MACD金叉且为正值")
    elif macd > macd_signal:
        combo_score += 20
        reasons.append("MACD金叉")
    elif macd < macd_signal and macd < 0:
        combo_score -= 30  # MACD死叉且为负，严重扣分
        reasons.append("⚠️  MACD死叉且为负值")
    elif macd < macd_signal:
        combo_score -= 20
        reasons.append("⚠️  MACD死叉")

    # 2. 均线多头评分 (最高 25分)
    if sma_10 > sma_30:
        combo_score += 25
        reasons.append("均线多头排列")
    else:
        combo_score -= 20
        reasons.append("⚠️  均线空头排列")

    # 3. 成交量评分 (最高 30分)
    if volume_ratio > 3.0:
        combo_score += 30
        reasons.append(f"爆量({volume_ratio:.1f}x)")
    elif volume_ratio > 2.0:
        combo_score += 28
        reasons.append(f"大幅放量({volume_ratio:.1f}x)")
    elif volume_ratio > 1.5:
        combo_score += 25
        reasons.append(f"放量突破({volume_ratio:.1f}x)")
    elif volume_ratio > 1.2:
        combo_score += 18
        reasons.append(f"温和放量({volume_ratio:.1f}x)")
    elif volume_ratio > 0.8:
        combo_score += 5
        reasons.append("成交量正常")
    elif volume_ratio > 0.5:
        combo_score -= 20
        reasons.append(f"⚠️  成交量不足({volume_ratio:.1f}x)")
    else:
        combo_score -= 35
        reasons.append(f"⚠️  成交量严重不足({volume_ratio:.1f}x)")

    # 4. 布林带位置评分 (最高 15分)
    # 关键改进: 50-90% 是健康强势位置
    if 50 <= bb_position <= 75:
        combo_score += 15
        reasons.append(f"布林带健康位置({bb_position:.1f}%)")
    elif 75 < bb_position <= 90:
        combo_score += 12
        reasons.append(f"布林带偏强位置({bb_position:.1f}%)")
    elif 90 < bb_position <= 100:
        combo_score += 8
        reasons.append(f"布林带接近上轨({bb_position:.1f}%)")
    elif bb_position > 100:
        combo_score += 5
        reasons.append(f"⚠️  突破布林带上轨({bb_position:.1f}%) - 高风险高收益")
    elif 30 <= bb_position < 50:
        combo_score += 5
        reasons.append(f"布林带中性位置({bb_position:.1f}%)")
    elif 10 <= bb_position < 30:
        combo_score += 10
        reasons.append(f"布林带偏低位置({bb_position:.1f}%) - 可能超跌")
    else:  # < 10
        combo_score += 12
        reasons.append(f"布林带极低位置({bb_position:.1f}%) - 超跌反弹机会")

    # 5. 价格流动性加成 (菜篮族经济学) - 最高 15分
    # 和成交量并列作为重要评分维度
    if current_price is not None:
        if current_price < 30:
            # 超低价股 (如6443 NT$28.70) -> +15分
            combo_score += 15
            reasons.append(f"💰 超低价股加成 (NT${current_price:.2f} < 30, +15分)")
        elif current_price < 60:
            # 低价股 (如8110 NT$56) -> +10分
            combo_score += 10
            reasons.append(f"💰 低价股加成 (NT${current_price:.2f} < 60, +10分)")
        elif current_price < 90:
            # 中低价股 (60-90) -> +5分
            combo_score += 5
            reasons.append(f"💰 中低价股加成 (NT${current_price:.2f} < 90, +5分)")
        # > NT$90 不加分

    return max(combo_score, -100), reasons


def calculate_enhanced_buy_score(rsi, macd, macd_signal, sma_10, sma_30, current_price,
                                  bb_upper, bb_lower, volume_ratio, ai_action, buy_weights,
                                  sentiment_score=0.0):
    """
    增强版买入评分系统
    ====================
    整合:
    1. 强势股识别
    2. 多头组合评分
    3. AI 模型建议
    4. 动态权重
    5. 题材股RSI钝化识别 (菜篮族经济学)

    返回: (final_score, signal_type, reasons, warnings, metadata)
    """
    # 计算布林带位置
    bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower) * 100) if (bb_upper - bb_lower) > 0 else 50

    # 0. 题材股RSI钝化识别 (菜篮族经济学核心逻辑)
    # 3个价格段，不同的sentiment阈值要求
    is_theme_stock = False
    theme_stock_tier = ""

    if rsi > 70:  # 基础条件：RSI超买
        if current_price < 30 and sentiment_score > 0.3:
            # 超低价股：要求强题材 (>0.3)
            is_theme_stock = True
            theme_stock_tier = "超低价题材股"
        elif current_price < 60 and sentiment_score > 0.2:
            # 低价股：要求中等题材 (>0.2)
            is_theme_stock = True
            theme_stock_tier = "低价题材股"
        elif current_price < 90 and sentiment_score > 0.15:
            # 中低价股：要求轻度题材 (>0.15)
            is_theme_stock = True
            theme_stock_tier = "中低价题材股"

    if is_theme_stock:
        print(f"\n🔥 识别到题材股RSI钝化 (菜篮族妖股模式)")
        print(f"   类型: {theme_stock_tier}")
        print(f"   价格: NT${current_price:.2f}")
        print(f"   情绪: {sentiment_score:.2f}")
        print(f"   RSI: {rsi:.1f} > 70 (钝化续涨)")

    # 1. 检查强势股模式
    is_strong, strong_strength, strong_reasons = identify_strong_stock_pattern(
        rsi, macd, macd_signal, sma_10, sma_30, volume_ratio, bb_position
    )

    # 2. 计算多头组合分数
    combo_score, combo_reasons = calculate_bullish_combo_score(
        rsi, macd, macd_signal, sma_10, sma_30, volume_ratio, bb_position, current_price
    )

    # 3. 基础技术指标评分
    base_score = 0
    base_reasons = []
    warnings = []

    # MACD 死叉直接拒绝
    if macd < macd_signal:
        base_score -= 100
        warnings.append("⚠️  MACD死叉,趋势转弱,不应买入!")
        signal_type = "WAIT"
    else:
        # RSI 评分 (改进: 强势股和题材股不扣分，反而加分！)
        if (is_strong or is_theme_stock) and rsi > 65:
            # 强势股/题材股 RSI 超买 = 正常现象，加分！
            if is_theme_stock:
                base_score += 15  # 题材股钝化加分
                base_reasons.append(f"🔥 题材股RSI钝化({rsi:.1f}) - 菜篮族续推 (+15分)")
            elif is_strong:
                base_score += 10  # 强势股钝化加分
                base_reasons.append(f"💪 强势股RSI钝化({rsi:.1f}) - 动能延续 (+10分)")
        elif rsi < 30:
            base_score += buy_weights.get('rsi_oversold', 30)
            base_reasons.append(f"✅ RSI超卖 ({rsi:.1f} < 30, +{buy_weights.get('rsi_oversold', 30)}分)")
        elif rsi < 50:
            base_score += buy_weights.get('rsi_low', 15)
            base_reasons.append(f"✅ RSI偏低 ({rsi:.1f}, +{buy_weights.get('rsi_low', 15)}分)")
        elif rsi > 80 and not is_strong and not is_theme_stock:
            # 弱势股 + 极高 RSI = 真的危险
            base_score -= 30
            warnings.append(f"⚠️  RSI极度超买({rsi:.1f})且无强势特征,回调风险极高")
        elif 70 < rsi <= 80 and not is_strong and not is_theme_stock:
            # 普通股票 RSI 70-80 = 适度警告
            base_score -= 15
            warnings.append(f"⚠️  RSI偏高({rsi:.1f})但无放量或趋势确认,建议谨慎")
        elif 65 < rsi <= 70 and not is_strong and not is_theme_stock:
            # 普通股票 RSI 65-70 = 轻微警告
            base_score -= 8
            warnings.append(f"⚠️  RSI略高({rsi:.1f}),注意是否有量能配合")

        # MACD 评分
        if macd > macd_signal and macd > 0:
            base_score += buy_weights.get('macd_bullish_strong', 35)
        elif macd > macd_signal:
            base_score += buy_weights.get('macd_bullish', 25)

        # 均线评分
        if sma_10 > sma_30:
            base_score += buy_weights.get('ma_bullish', 25)

        signal_type = "BUY"

    # 4. 综合评分 (根据不同模式调整权重)
    if is_strong:
        # 强势股模式: 优先强势特征
        final_score = (
            strong_strength * 0.40 +    # 强势模式 40%
            combo_score * 0.30 +        # 多头组合 30%
            (ai_action * 50) * 0.20 +   # AI 模型 20%
            base_score * 0.10           # 基础技术 10%
        )
        print(f"\n✨ 识别到强势股模式 (强度: {strong_strength}/100)")
    elif combo_score > 60:
        # 强力多头组合
        final_score = (
            combo_score * 0.40 +        # 多头组合 40%
            (ai_action * 50) * 0.30 +   # AI 模型 30%
            base_score * 0.30           # 基础技术 30%
        )
        print(f"\n📈 识别到强力多头组合 (分数: {combo_score}/100)")
    else:
        # 常规评分
        final_score = (
            base_score * 0.40 +         # 基础技术 40%
            (ai_action * 50) * 0.40 +   # AI 模型 40%
            combo_score * 0.20          # 多头组合 20%
        )

    # 合并理由
    all_reasons = []
    if is_strong:
        all_reasons.extend(strong_reasons)
    all_reasons.extend(combo_reasons)
    all_reasons.extend(base_reasons)

    # 返回元数据
    metadata = {
        'is_strong_stock': is_strong,
        'strong_strength': strong_strength,
        'combo_score': combo_score,
        'base_score': base_score,
        'ai_score': ai_action * 50
    }

    return final_score, signal_type, all_reasons, warnings, metadata


# 使用示例
if __name__ == "__main__":
    print("=" * 70)
    print("增强版买入评分系统 - 测试")
    print("=" * 70)

    # 默认权重
    default_weights = {
        'rsi_oversold': 30,
        'rsi_low': 15,
        'macd_bullish_strong': 35,
        'macd_bullish': 25,
        'ma_bullish': 25
    }

    # 测试案例 1: 2408.TW 强势股
    print("\n测试案例 1: 2408.TW (强势股)")
    print("-" * 70)
    score, signal, reasons, warnings, meta = calculate_enhanced_buy_score(
        rsi=77.54,
        macd=9.9546,
        macd_signal=8.0678,
        sma_10=169.30,
        sma_30=158.27,
        current_price=189.00,
        bb_upper=184.91,
        bb_lower=137.14,
        volume_ratio=1.49,
        ai_action=-1.0,  # AI 建议卖出
        buy_weights=default_weights
    )

    print(f"\n最终评分: {score:.2f}")
    print(f"信号类型: {signal}")
    print(f"强势股: {meta['is_strong_stock']} (强度: {meta['strong_strength']})")
    print(f"多头组合: {meta['combo_score']:.0f}")
    print(f"基础评分: {meta['base_score']:.0f}")
    print(f"AI评分: {meta['ai_score']:.0f}")

    if reasons:
        print(f"\n买入理由:")
        for reason in reasons[:5]:
            print(f"  • {reason}")

    if warnings:
        print(f"\n警告:")
        for warning in warnings:
            print(f"  • {warning}")

    # 测试案例 2: 普通超买
    print("\n" + "=" * 70)
    print("测试案例 2: 普通超买 (无强势确认)")
    print("-" * 70)
    score2, signal2, reasons2, warnings2, meta2 = calculate_enhanced_buy_score(
        rsi=75.0,
        macd=-0.5,
        macd_signal=0.2,
        sma_10=100.0,
        sma_30=102.0,
        current_price=105.0,
        bb_upper=110.0,
        bb_lower=95.0,
        volume_ratio=0.6,
        ai_action=0.3,
        buy_weights=default_weights
    )

    print(f"\n最终评分: {score2:.2f}")
    print(f"信号类型: {signal2}")
    print(f"强势股: {meta2['is_strong_stock']}")
    print(f"多头组合: {meta2['combo_score']:.0f}")

    if warnings2:
        print(f"\n警告:")
        for warning in warnings2:
            print(f"  • {warning}")

    print("\n" + "=" * 70)
