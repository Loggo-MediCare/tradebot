"""
动态交易信号权重计算器
基于特征重要性分析，为每个股票生成自适应的评分权重
"""
import sys
import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
from typing import Dict, Tuple


class DynamicWeightCalculator:
    """根据特征重要性动态计算交易信号权重"""

    # 默认权重 (如果没有特征重要性数据)
    # 注意: RSI 已替换为分析師目標價 (更有參考價值)
    # 注意: price_vs_ma 已移除 (特征重要性为0%)
    DEFAULT_WEIGHTS = {
        'analyst_target': 20,  # 分析師目標價 vs 當前價格
        'macd': 35,            # MACD 金叉/死叉
        'ma_trend': 23,        # 均线趋势
        'bb_position': 10,     # 布林带位置
        'volume': 15,          # 成交量（单独评分，不参与特征重要性）
        'fundamentals': 15,    # 基本面（净利润/利润率）
    }

    def __init__(self, ticker: str, project_dir: str = r"C:\Users\Silvi\Projects\trading-bot"):
        """
        初始化权重计算器

        Args:
            ticker: 股票代码，如 'NVDA', '2408.TW'
            project_dir: 项目目录路径
        """
        self.ticker = ticker
        self.project_dir = project_dir
        self.feature_importance = None
        self.weights = self.DEFAULT_WEIGHTS.copy()

        # 尝试加载特征重要性数据
        self._load_feature_importance()

        # 如果成功加载，计算动态权重
        if self.feature_importance:
            self._calculate_dynamic_weights()

    def _load_feature_importance(self):
        """从 JSON 文件加载特征重要性数据"""
        json_filename = f"{self.ticker}_feature_importance.json"
        json_path = os.path.join(self.project_dir, json_filename)

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feature_importance = data.get('feature_importance', {})
                    print(f"✅ 已加载 {self.ticker} 的特征重要性数据")
                    return True
            except Exception as e:
                # Only print error if loading failed, not if file doesn't exist
                print(f"⚠️  无法加载特征重要性数据: {e}")
                return False
        else:
            # File doesn't exist - silently use default weights
            # Uncomment the lines below if you want to see the warning:
            # print(f"⚠️  未找到特征重要性文件: {json_filename}")
            # print(f"   将使用默认权重")
            return False

    def _calculate_dynamic_weights(self):
        """根据特征重要性计算动态权重"""
        if not self.feature_importance:
            return

        # 映射：特征名 -> 信号类型
        # 注意：某些特征会合并到同一个信号类型
        # 注意：analyst_target 不在特征重要性中，使用固定權重
        feature_to_signal = {
            'MACD': 'macd',
            'macd': 'macd',
            'MACD_Signal': 'macd',
            'macd_signal': 'macd',
            'MACD_Histogram': 'macd',
            'macd_hist': 'macd',
            'MA_20': 'ma_trend',
            'MA_50': 'ma_trend',
            'MA_200': 'ma_trend',
            'MA50_slope': 'ma_trend',
            'BB_Position': 'bb_position',
            'bb_position': 'bb_position',
        }

        # 聚合特征重要性到信号类型
        # analyst_target 使用固定權重20 (不在ML特徵中)
        signal_importance = {
            'macd': 0,
            'ma_trend': 0,
            'bb_position': 0,
        }

        for feature, importance in self.feature_importance.items():
            signal_type = feature_to_signal.get(feature)
            if signal_type:
                signal_importance[signal_type] += importance

        # 归一化到总和=100的权重系统
        # 注意：成交量(volume)与基本面(fundamentals)不在特征重要性里，使用固定权重
        total_importance = sum(signal_importance.values())

        if total_importance > 0:
            # 将 38 分分配给技术指标 (保留20分给分析師目標價, 15分给成交量, 15分给基本面)
            allocation_budget = 38

            for signal_type, importance in signal_importance.items():
                # 按比例分配权重
                self.weights[signal_type] = int((importance / total_importance) * allocation_budget)

            # 分析師目標價 固定權重20
            self.weights['analyst_target'] = 20
            # 成交量与基本面保持固定权重
            self.weights['volume'] = 15
            self.weights['fundamentals'] = 15

            print(f"\n📊 {self.ticker} 动态权重计算结果:")
            print(f"   特征重要性总和: {total_importance:.4f}")
            print(f"   权重分配:")
            for signal_type, weight in self.weights.items():
                if signal_type == 'analyst_target':
                    print(f"     • {signal_type:14s}: {weight:3d} 分 (分析師共識)")
                elif signal_type == 'volume':
                    print(f"     • {signal_type:14s}: {weight:3d} 分 (固定)")
                elif signal_type == 'fundamentals':
                    print(f"     • {signal_type:14s}: {weight:3d} 分 (净利/利润率)")
                else:
                    importance_pct = signal_importance.get(signal_type, 0) * 100
                    print(f"     • {signal_type:14s}: {weight:3d} 分 (特征重要性: {importance_pct:5.2f}%)")

    def get_sell_weights(self) -> Dict[str, int]:
        """
        获取卖出信号的权重配置

        Returns:
            权重字典，包含:
            - target_below: 目標價低於現價 (賣出信號)
            - target_near: 目標價接近現價
            - macd_bearish: MACD 死叉权重
            - ma_bearish: 均线空头排列权重
            - bb_upper: 布林带上轨权重
            - bb_high: 布林带偏高权重
        """
        base_weights = self.weights

        return {
            # 分析師目標價相关 (目標價 < 現價 = 賣出信號)
            'target_below': int(base_weights['analyst_target'] * 0.6),    # 60% 目標價低於現價
            'target_near': int(base_weights['analyst_target'] * 0.4),     # 40% 目標價接近現價
            'fundamentals_negative': base_weights.get('fundamentals', 15), # 基本面转弱

            # RSI 相关
            'rsi_severe': 15,    # RSI > 80 严重超买
            'rsi_high': 10,      # RSI > 70 超买
            'rsi_mild': 5,       # RSI > 65 偏高

            # MACD 相关
            'macd_bearish': base_weights['macd'],              # 100% MACD权重

            # 均线相关
            'ma_bearish': base_weights['ma_trend'],            # 100% MA权重

            # 布林带相关
            'bb_upper': int(base_weights['bb_position'] * 0.7),  # 70% BB权重
            'bb_high': int(base_weights['bb_position'] * 0.3),   # 30% BB权重
        }

    def get_buy_weights(self) -> Dict[str, int]:
        """
        获取买入信号的权重配置

        Returns:
            权重字典，包含:
            - target_above_high: 目標價遠高於現價 (強力買入)
            - target_above: 目標價高於現價 (買入)
            - macd_bullish_strong: MACD 金叉且为正值权重
            - macd_bullish: MACD 金叉权重
            - ma_bullish: 均线多头排列权重
        """
        base_weights = self.weights

        return {
            # 分析師目標價相关 (目標價 > 現價 = 買入信號)
            'target_above_high': int(base_weights['analyst_target'] * 0.6),  # 60% 目標價遠高於現價
            'target_above': int(base_weights['analyst_target'] * 0.4),       # 40% 目標價高於現價
            'fundamentals_positive': base_weights.get('fundamentals', 15),   # 基本面健康

            # MACD 相关
            'macd_bullish_strong': int(base_weights['macd'] * 0.6),  # 60% MACD权重
            'macd_bullish': int(base_weights['macd'] * 0.4),         # 40% MACD权重

            # 均线相关
            'ma_bullish': base_weights['ma_trend'],            # 100% MA权重
        }

    def print_summary(self):
        """打印权重摘要"""
        print("\n" + "=" * 70)
        print(f"📊 {self.ticker} 交易信号权重配置")
        print("=" * 70)

        if self.feature_importance:
            print("\n✅ 使用动态权重 (基于特征重要性分析)")
        else:
            print("\n⚠️  使用默认权重 (未找到特征重要性数据)")

        print("\n卖出信号权重:")
        sell_weights = self.get_sell_weights()
        for key, weight in sell_weights.items():
            print(f"  • {key:25s}: {weight:3d} 分")

        print("\n买入信号权重:")
        buy_weights = self.get_buy_weights()
        for key, weight in buy_weights.items():
            print(f"  • {key:25s}: {weight:3d} 分")

        print("=" * 70)


def get_dynamic_weights(ticker: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    便捷函数：获取指定股票的动态权重

    Args:
        ticker: 股票代码

    Returns:
        (buy_weights, sell_weights) 元组
    """
    calculator = DynamicWeightCalculator(ticker)
    return calculator.get_buy_weights(), calculator.get_sell_weights()


# 测试代码
if __name__ == "__main__":
    print("=" * 70)
    print("动态交易信号权重计算器 - 测试")
    print("=" * 70)

    # 测试 NVDA (有特征重要性数据)
    print("\n测试 1: NVDA (应该有特征重要性数据)")
    calc_nvda = DynamicWeightCalculator('NVDA')
    calc_nvda.print_summary()

    # 测试 2408.TW (可能没有数据，使用默认权重)
    print("\n\n测试 2: 2408.TW")
    calc_2408 = DynamicWeightCalculator('2408.TW')
    calc_2408.print_summary()
