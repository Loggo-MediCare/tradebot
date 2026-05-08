"""
法人爆量策略交易系統
===================
基於法人爆量信號的量化交易策略

核心邏輯:
1. 爆量信號檢測 (量比 > 1.5x + 價格漲幅 > 2%)
2. 趨勢確認 (MA50 方向)
3. 風險管理 (停損停利點)
4. 專家系統整合評分
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# 導入爆量檢測模組
try:
    from volume_surge_detector import get_volume_signal
except ImportError:
    get_volume_signal = None


class InstitutionalVolumeStrategy:
    """
    法人爆量交易策略主類別
    """

    def __init__(self,
                 volume_multiplier: float = 1.5,
                 price_change_threshold: float = 0.02,
                 ma_short: int = 20,
                 ma_long: int = 50,
                 stop_loss: float = 0.05,
                 take_profit: float = 0.15,
                 position_size: float = 0.2):
        """
        初始化策略參數

        Args:
            volume_multiplier: 爆量倍數門檻
            price_change_threshold: 價格變化門檻
            ma_short: 短期均線週期
            ma_long: 長期均線週期
            stop_loss: 停損比例 (5%)
            take_profit: 停利比例 (15%)
            position_size: 倉位大小比例 (20%)
        """
        self.volume_multiplier = volume_multiplier
        self.price_change_threshold = price_change_threshold
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position_size = position_size

        # 策略狀態
        self.current_position = 0
        self.entry_price = 0
        self.stop_loss_price = 0
        self.take_profit_price = 0

        # 交易記錄
        self.trades = []
        self.signals = []

    def calculate_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """計算技術指標"""
        result = {
            'ma_short': 0,
            'ma_long': 0,
            'ma_trend': 'NEUTRAL',
            'rsi': 50,
            'macd': 0,
            'macd_signal': 0,
            'volume_sma': 0,
            'score': 0
        }

        if len(df) < max(self.ma_short, self.ma_long, 26):
            return result

        close = df['close']

        # 移動平均線
        ma_short = close.rolling(window=self.ma_short).mean()
        ma_long = close.rolling(window=self.ma_long).mean()

        result['ma_short'] = ma_short.iloc[-1]
        result['ma_long'] = ma_long.iloc[-1]

        # 趨勢判斷
        if result['ma_short'] > result['ma_long']:
            result['ma_trend'] = 'UP'
            result['score'] += 10
        elif result['ma_short'] < result['ma_long']:
            result['ma_trend'] = 'DOWN'
            result['score'] -= 5

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        result['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        if result['rsi'] > 70:
            result['score'] -= 5
        elif result['rsi'] < 30:
            result['score'] += 5

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()

        result['macd'] = macd.iloc[-1]
        result['macd_signal'] = signal.iloc[-1]

        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            result['score'] += 10
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            result['score'] -= 10

        result['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]

        return result

    def analyze_signal_strength(self,
                               volume_signal: Dict,
                               technical_indicators: Dict,
                               current_price: float) -> Dict:
        """分析信號強度"""
        result = {
            'overall_score': 0,
            'signal_strength': 'WEAK',
            'recommendation': 'HOLD',
            'confidence': 0,
            'reasons': [],
            'risk_level': 'MEDIUM'
        }

        total_score = 0
        reasons = []

        # 爆量信號評分
        if 'score_adjustment' in volume_signal:
            surge_score = volume_signal['score_adjustment']
            total_score += surge_score
            if surge_score > 0:
                reasons.append(f"爆量上漲信號 (+{surge_score}分)")
            elif surge_score < 0:
                reasons.append(f"爆量下跌信號 ({surge_score}分)")

        # 技術指標評分
        tech_score = technical_indicators.get('score', 0)
        total_score += tech_score
        reasons.append(f"技術指標: {tech_score}分")

        if technical_indicators['ma_trend'] == 'UP':
            reasons.append("多頭趨勢確認")
        elif technical_indicators['ma_trend'] == 'DOWN':
            reasons.append("空頭趨勢確認")

        # 價格位置評分
        if technical_indicators['ma_short'] > 0:
            price_to_ma = (current_price - technical_indicators['ma_short']) / technical_indicators['ma_short']
            if -0.02 < price_to_ma < 0.02:
                total_score += 5
                reasons.append("價格接近均線支撐")
            elif price_to_ma > 0.05:
                total_score -= 3
                reasons.append("價格偏高，注意回調")

        # 成交量確認
        volume_ratio = volume_signal.get('surge', {}).get('volume_ratio', 1)
        if volume_ratio > 2.0:
            total_score += 5
            reasons.append(f"極度爆量 ({volume_ratio:.1f}x)")

        result['overall_score'] = total_score

        # 信號強度
        if total_score >= 20:
            result['signal_strength'] = 'STRONG'
            result['confidence'] = min(90, 60 + total_score)
        elif total_score >= 10:
            result['signal_strength'] = 'MODERATE'
            result['confidence'] = min(75, 50 + total_score)
        else:
            result['signal_strength'] = 'WEAK'
            result['confidence'] = max(30, 40 + total_score)

        # 交易建議
        if total_score >= 15:
            result['recommendation'] = 'BUY'
            if volume_signal.get('surge', {}).get('type') == 'SURGE_DOWN':
                result['recommendation'] = 'SELL'
                result['risk_level'] = 'HIGH'
        elif total_score <= -10:
            result['recommendation'] = 'SELL'
            result['risk_level'] = 'HIGH'

        # 逆勢爆量觀望
        if (technical_indicators['ma_trend'] == 'DOWN' and
            volume_signal.get('surge', {}).get('type') == 'SURGE_UP'):
            result['recommendation'] = 'HOLD'
            result['confidence'] = max(30, result['confidence'] - 20)
            reasons.append("逆勢爆量，建議觀望")

        result['reasons'] = reasons
        return result

    def generate_trading_signal(self, df: pd.DataFrame) -> Dict:
        """生成交易信號"""
        signal = {
            'timestamp': datetime.now(),
            'current_price': df['close'].iloc[-1],
            'signal': 'HOLD',
            'strength': 'NEUTRAL',
            'confidence': 0,
            'recommended_action': '等待',
            'position_size': self.position_size,
            'stop_loss': 0,
            'take_profit': 0,
            'risk_reward_ratio': 0,
            'analysis': {},
            'reasons': []
        }

        try:
            # 獲取爆量信號
            if get_volume_signal:
                volume_signal = get_volume_signal(df)
            else:
                volume_signal = {'score_adjustment': 0, 'surge': {}}

            # 計算技術指標
            technical = self.calculate_technical_indicators(df)

            # 分析信號強度
            current_price = df['close'].iloc[-1]
            analysis = self.analyze_signal_strength(volume_signal, technical, current_price)

            signal['signal'] = analysis['recommendation']
            signal['strength'] = analysis['signal_strength']
            signal['confidence'] = analysis['confidence']
            signal['analysis'] = {
                'volume_signal': volume_signal,
                'technical_indicators': technical,
                'signal_analysis': analysis
            }
            signal['reasons'] = analysis['reasons']

            # 設置交易參數
            if analysis['recommendation'] == 'BUY':
                signal['recommended_action'] = '買入'
                signal['stop_loss'] = current_price * (1 - self.stop_loss)
                signal['take_profit'] = current_price * (1 + self.take_profit)
                signal['risk_reward_ratio'] = self.take_profit / self.stop_loss
            elif analysis['recommendation'] == 'SELL':
                signal['recommended_action'] = '賣出'
                signal['stop_loss'] = current_price * (1 + self.stop_loss)
                signal['take_profit'] = current_price * (1 - self.take_profit)
                signal['risk_reward_ratio'] = self.take_profit / self.stop_loss

            self.signals.append(signal.copy())

        except Exception as e:
            signal['recommended_action'] = f'錯誤: {str(e)}'
            signal['confidence'] = 0

        return signal

    def format_signal_output(self, signal: Dict) -> str:
        """格式化信號輸出"""
        output = []
        output.append("=" * 60)
        output.append("法人爆量策略交易信號")
        output.append("=" * 60)
        output.append(f"時間: {signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"當前價格: {signal['current_price']:.2f}")
        output.append(f"信號: {signal['signal']} ({signal['strength']})")
        output.append(f"信心度: {signal['confidence']:.1f}%")
        output.append(f"建議動作: {signal['recommended_action']}")

        if signal['signal'] != 'HOLD':
            output.append(f"建議倉位: {signal['position_size']*100:.0f}%")
            output.append(f"停損點: {signal['stop_loss']:.2f}")
            output.append(f"停利點: {signal['take_profit']:.2f}")
            output.append(f"風險報酬比: {signal['risk_reward_ratio']:.2f}")

        output.append("\n分析理由:")
        for reason in signal['reasons']:
            output.append(f"  - {reason}")

        output.append("=" * 60)
        return "\n".join(output)


def get_institutional_signal(df: pd.DataFrame) -> Dict:
    """
    快速獲取法人爆量信號 (用於整合到交易系統)

    Returns:
        dict: {
            'signal': str,
            'strength': str,
            'confidence': float,
            'score_adjustment': int,
            'signal_text': str,
            'stop_loss': float,
            'take_profit': float
        }
    """
    strategy = InstitutionalVolumeStrategy()
    full_signal = strategy.generate_trading_signal(df)

    result = {
        'signal': full_signal['signal'],
        'strength': full_signal['strength'],
        'confidence': full_signal['confidence'],
        'score_adjustment': 0,
        'signal_text': '',
        'stop_loss': full_signal['stop_loss'],
        'take_profit': full_signal['take_profit'],
        'reasons': full_signal['reasons']
    }

    # 計算評分調整
    if full_signal['signal'] == 'BUY' and full_signal['confidence'] > 60:
        result['score_adjustment'] = int(full_signal['confidence'] / 10)
        result['signal_text'] = f"法人爆量買入 (信心: {full_signal['confidence']:.0f}%)"
    elif full_signal['signal'] == 'SELL' and full_signal['confidence'] > 60:
        result['score_adjustment'] = -int(full_signal['confidence'] / 10)
        result['signal_text'] = f"法人爆量賣出 (信心: {full_signal['confidence']:.0f}%)"

    return result


# ======================================================
# 主程序測試
# ======================================================

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 60)
    print("Institutional Volume Strategy Test")
    print("=" * 60)

    # 模擬數據
    np.random.seed(42)
    n = 100

    prices = 100 + np.cumsum(np.random.randn(n) * 2)
    volumes = np.random.randint(100000, 500000, n)

    # 模擬爆量上漲
    prices[-1] = prices[-2] * 1.03
    volumes[-1] = volumes[-20:-1].mean() * 2.5

    df = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.98,
        'close': prices,
        'volume': volumes
    })

    # 測試
    strategy = InstitutionalVolumeStrategy()
    signal = strategy.generate_trading_signal(df)
    print(strategy.format_signal_output(signal))

    # 快速信號
    quick_signal = get_institutional_signal(df)
    print(f"\nQuick Signal: {quick_signal['signal_text']}")
    print(f"Score adjustment: {quick_signal['score_adjustment']:+d}")
