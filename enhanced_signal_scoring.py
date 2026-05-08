"""
增强版技术指标评分系统
====================================
改进点:
1. 强势股识别 (RSI超买 + 多头确认)
2. 组合信号识别 (MACD金叉 + 均线多头 + 放量)
3. 动态权重调整
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


class EnhancedSignalScoring:
    """增强版信号评分系统"""

    def __init__(self, ticker, use_feature_importance=True):
        self.ticker = ticker
        self.use_feature_importance = use_feature_importance

    def identify_strong_stock_pattern(self, indicators):
        """
        识别强势股模式
        ================
        条件: RSI > 70 但同时满足:
        - MACD 金叉
        - 均线多头
        - 放量 (> 1.2x)

        返回: True/False 和 强度分数
        """
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        sma10 = indicators.get('sma10', 0)
        sma30 = indicators.get('sma30', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        bb_position = indicators.get('bb_position', 50)

        # 检查强势股条件
        is_strong = (
            rsi > 70 and                      # RSI 超买
            macd > macd_signal and           # MACD 金叉
            macd > 0 and                     # MACD 为正
            sma10 > sma30 and                # 均线多头
            volume_ratio > 1.2               # 放量
        )

        if is_strong:
            # 计算强势程度 (0-100)
            strength = 0

            # RSI 越高，强势程度越高 (但有上限)
            if 70 < rsi < 80:
                strength += 30
            elif rsi >= 80:
                strength += 20  # 超过80反而降低，因为风险增加

            # 放量程度
            if volume_ratio > 2.0:
                strength += 30  # 大量放量
            elif volume_ratio > 1.5:
                strength += 20  # 温和放量
            else:
                strength += 10

            # MACD 强度
            macd_strength = abs(macd - macd_signal)
            if macd_strength > 1.0:
                strength += 20
            else:
                strength += 10

            # 均线差距
            ma_gap = ((sma10 - sma30) / sma30) * 100
            if ma_gap > 5:
                strength += 20
            elif ma_gap > 2:
                strength += 10

            return True, min(strength, 100)

        return False, 0

    def identify_bullish_combo(self, indicators):
        """
        识别多头组合信号
        ==================
        经典组合:
        1. MACD 金叉
        2. 均线多头排列
        3. 放量 (> 1.2x)
        4. 布林带中上位置 (50-100%)

        返回: 组合强度 (0-100)
        """
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        sma10 = indicators.get('sma10', 0)
        sma30 = indicators.get('sma30', 0)
        volume_ratio = indicators.get('volume_ratio', 1.0)
        bb_position = indicators.get('bb_position', 50)
        rsi = indicators.get('rsi', 50)

        combo_score = 0

        # 1. MACD 金叉 (+25分)
        if macd > macd_signal and macd > 0:
            combo_score += 25

        # 2. 均线多头 (+25分)
        if sma10 > sma30:
            combo_score += 25

        # 3. 放量 (+25分)
        if volume_ratio > 2.0:
            combo_score += 25
        elif volume_ratio > 1.5:
            combo_score += 20
        elif volume_ratio > 1.2:
            combo_score += 15

        # 4. 布林带位置 (+25分)
        # 关键改进: 50-90% 是健康的强势位置，不应扣分
        if 50 <= bb_position <= 90:
            combo_score += 25  # 健康强势
        elif 90 < bb_position <= 100:
            combo_score += 15  # 接近上轨，稍微谨慎
        elif bb_position > 100:
            combo_score += 5   # 突破上轨，高风险高收益
        elif 30 <= bb_position < 50:
            combo_score += 10  # 中性偏弱

        # 额外加分: RSI 配合
        if 50 < rsi < 70 and combo_score > 60:
            combo_score += 10  # RSI 健康 + 组合强劲

        return min(combo_score, 100)

    def calculate_enhanced_score(self, indicators, ai_action):
        """
        增强版综合评分
        ================
        整合:
        1. 传统技术指标评分
        2. 强势股识别
        3. 多头组合识别
        4. AI 模型输出

        返回: 综合分数 (-100 到 +100)
        """
        # 1. 检查强势股模式
        is_strong, strong_strength = self.identify_strong_stock_pattern(indicators)

        # 2. 检查多头组合
        bullish_combo_score = self.identify_bullish_combo(indicators)

        # 3. 基础技术指标评分 (简化版)
        base_score = self._calculate_base_score(indicators)

        # 4. AI 模型权重
        ai_score = ai_action * 50  # AI输出 -1到+1 映射到 -50到+50

        # 综合评分逻辑
        if is_strong:
            # 强势股: 优先考虑强势模式
            print(f"\n✨ 识别到强势股模式 (强度: {strong_strength}/100)")
            final_score = (
                strong_strength * 0.4 +      # 强势模式 40%
                bullish_combo_score * 0.3 +  # 多头组合 30%
                ai_score * 0.2 +             # AI 模型 20%
                base_score * 0.1             # 基础技术 10%
            )
        elif bullish_combo_score > 70:
            # 强力多头组合
            print(f"\n📈 识别到强力多头组合 (分数: {bullish_combo_score}/100)")
            final_score = (
                bullish_combo_score * 0.4 +  # 多头组合 40%
                ai_score * 0.3 +             # AI 模型 30%
                base_score * 0.3             # 基础技术 30%
            )
        else:
            # 常规评分
            final_score = (
                base_score * 0.4 +           # 基础技术 40%
                ai_score * 0.4 +             # AI 模型 40%
                bullish_combo_score * 0.2    # 多头组合 20%
            )

        return {
            'final_score': final_score,
            'is_strong_stock': is_strong,
            'strong_strength': strong_strength,
            'bullish_combo': bullish_combo_score,
            'base_score': base_score,
            'ai_score': ai_score
        }

    def _calculate_base_score(self, indicators):
        """基础技术指标评分"""
        score = 0

        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        sma10 = indicators.get('sma10', 0)
        sma30 = indicators.get('sma30', 0)

        # RSI 评分 (改进: 不再过度恐慌)
        if 40 < rsi < 60:
            score += 20  # 健康中性
        elif 30 < rsi <= 40:
            score += 10  # 偏弱但可接受
        elif rsi < 30:
            score -= 20  # 超卖
        elif 60 <= rsi < 70:
            score += 10  # 偏强
        # 注意: RSI > 70 不再直接扣分，交给强势股识别

        # MACD 评分
        if macd > macd_signal:
            score += 20
        else:
            score -= 20

        # 均线评分
        if sma10 > sma30:
            score += 20
        else:
            score -= 20

        return score


# 使用示例
if __name__ == "__main__":
    print("=" * 70)
    print("增强版技术指标评分系统 - 测试")
    print("=" * 70)

    # 测试案例 1: 强势股 (RSI超买 + 多头确认)
    print("\n测试案例 1: 强势股 (类似 2408.TW)")
    indicators_strong = {
        'rsi': 77.5,
        'macd': 9.95,
        'macd_signal': 8.07,
        'sma10': 169.30,
        'sma30': 158.27,
        'volume_ratio': 1.49,
        'bb_position': 108.6,
        'price': 189.0
    }

    scorer = EnhancedSignalScoring('2408.TW')
    result = scorer.calculate_enhanced_score(indicators_strong, ai_action=-1.0)

    print(f"\n评分结果:")
    print(f"  最终分数: {result['final_score']:.2f}")
    print(f"  强势股: {result['is_strong_stock']} (强度: {result['strong_strength']})")
    print(f"  多头组合: {result['bullish_combo']}/100")
    print(f"  基础分数: {result['base_score']}")
    print(f"  AI 分数: {result['ai_score']}")

    # 测试案例 2: 普通超买 (RSI高但无确认)
    print("\n" + "=" * 70)
    print("测试案例 2: 普通超买 (无强势确认)")
    indicators_normal_overbought = {
        'rsi': 75.0,
        'macd': -0.5,
        'macd_signal': 0.2,
        'sma10': 100.0,
        'sma30': 102.0,
        'volume_ratio': 0.8,
        'bb_position': 95.0,
        'price': 100.0
    }

    result2 = scorer.calculate_enhanced_score(indicators_normal_overbought, ai_action=-0.3)

    print(f"\n评分结果:")
    print(f"  最终分数: {result2['final_score']:.2f}")
    print(f"  强势股: {result2['is_strong_stock']} (强度: {result2['strong_strength']})")
    print(f"  多头组合: {result2['bullish_combo']}/100")
    print(f"  基础分数: {result2['base_score']}")
    print(f"  AI 分数: {result2['ai_score']}")

    print("\n" + "=" * 70)
    print("结论: 强势股识别系统可以区分「真超买」和「强势突破」")
    print("=" * 70)
