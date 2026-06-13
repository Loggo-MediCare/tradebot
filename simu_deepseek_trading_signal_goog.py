"""
爆量下跌再評估策略
================
針對昨日爆量下跌信號的後續分析與策略調整
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

class PostSurgeAnalysis:
    """
    爆量後續分析策略
    用於分析爆量下跌後的市場行為
    """
    
    def __init__(self):
        self.surge_signals = []
        self.follow_up_analysis = {}
        
    def analyze_post_surge_pattern(self, df: pd.DataFrame, surge_day_index: int = -1) -> Dict:
        """
        分析爆量後的價格模式
        
        Args:
            df: K線數據
            surge_day_index: 爆量日索引（負數表示從最後往前數）
            
        Returns:
            dict: 爆量後分析結果
        """
        result = {
            'surge_type': '',
            'days_since_surge': 0,
            'price_change_since_surge': 0,
            'volume_pattern': '',
            'support_test': False,
            'recovery_strength': 'WEAK',
            'recommendation': 'HOLD',
            'key_levels': [],
            'risk_assessment': {}
        }
        
        if len(df) < 5:
            return result
        
        # 獲取爆量日數據
        surge_date = df.index[surge_day_index]
        surge_close = df['close'].iloc[surge_day_index]
        surge_volume = df['volume'].iloc[surge_day_index]
        surge_low = df['low'].iloc[surge_day_index]
        
        # 計算爆量日前5日的平均成交量
        if len(df) > 5:
            avg_volume_before = df['volume'].iloc[surge_day_index-5:surge_day_index].mean()
        else:
            avg_volume_before = df['volume'].iloc[:surge_day_index].mean()
        
        volume_ratio = surge_volume / avg_volume_before if avg_volume_before > 0 else 1
        
        # 判斷爆量類型
        if surge_day_index < 0:
            days_since = -surge_day_index - 1
            
            # 分析爆量後的價格行為
            if days_since >= 1:
                # 爆量後第一日
                day1_close = df['close'].iloc[-1]
                day1_volume = df['volume'].iloc[-1]
                
                price_change = (day1_close - surge_close) / surge_close
                result['price_change_since_surge'] = price_change
                result['days_since_surge'] = days_since
                
                # 判斷爆量後行為
                if price_change > 0.01:  # 上漲超過1%
                    result['recovery_strength'] = 'STRONG'
                    result['recommendation'] = 'BUY_ON_DIP'
                    
                    # 檢查是否回補缺口
                    surge_open = df['open'].iloc[surge_day_index]
                    if day1_close > surge_open:
                        result['surge_type'] = 'BEAR_TRAP'  # 空頭陷阱
                        result['recommendation'] = 'BUY'
                        
                elif price_change < -0.01:  # 繼續下跌
                    result['recovery_strength'] = 'WEAK'
                    result['recommendation'] = 'SELL'
                    result['surge_type'] = 'DISTRIBUTION'  # 出貨
                    
                else:  # 盤整
                    result['recovery_strength'] = 'NEUTRAL'
                    result['recommendation'] = 'HOLD'
                    result['surge_type'] = 'CONSOLIDATION'
                    
                # 檢查量能變化
                volume_ratio_day1 = day1_volume / avg_volume_before
                if volume_ratio_day1 > 1.2:
                    result['volume_pattern'] = 'CONTINUED_HIGH_VOLUME'
                else:
                    result['volume_pattern'] = 'VOLUME_NORMALIZATION'
                    
        # 尋找關鍵支撐位
        support_levels = self.find_support_levels(df, surge_day_index)
        result['key_levels'] = support_levels
        
        # 檢查是否測試支撐
        current_price = df['close'].iloc[-1]
        for level in support_levels[:2]:  # 只檢查前兩個重要支撐
            if abs(current_price - level['price']) / level['price'] < 0.01:
                result['support_test'] = True
                result['recommendation'] = 'BUY_AT_SUPPORT'
                break
        
        # 風險評估 (改進版：獨立考慮價格跌幅)
        price_drop = result['price_change_since_surge']

        # 高風險條件 (任一成立即為高風險)
        is_high_risk = (
            price_drop < -0.05 or                          # 跌幅超過 5%
            (volume_ratio > 2.0 and price_drop < -0.02) or # 爆量 + 跌幅 > 2%
            (volume_ratio > 2.5)                           # 極度爆量
        )

        # 中風險條件
        is_moderate_risk = (
            (-0.05 <= price_drop < -0.02) or              # 跌幅 2%-5%
            (volume_ratio > 1.5 and price_drop < -0.01)   # 量增 + 下跌
        )

        # 低風險: 非高風險且非中風險
        is_low_risk = not is_high_risk and not is_moderate_risk

        result['risk_assessment'] = {
            'high_risk': is_high_risk,
            'moderate_risk': is_moderate_risk and not is_high_risk,
            'low_risk': is_low_risk
        }
        
        return result
    
    def find_support_levels(self, df: pd.DataFrame, surge_day_index: int) -> List[Dict]:
        """
        尋找關鍵支撐位
        
        Args:
            df: K線數據
            surge_day_index: 爆量日索引
            
        Returns:
            list: 支撐位列表
        """
        supports = []
        
        if len(df) < 20:
            return supports
        
        # 1. 爆量日低點
        surge_low = df['low'].iloc[surge_day_index]
        supports.append({
            'type': 'SURGE_DAY_LOW',
            'price': surge_low,
            'strength': 'STRONG',
            'description': f'爆量日低點 ${surge_low:.2f}'
        })
        
        # 2. 移動平均線支撐
        ma20 = df['close'].rolling(window=20).mean().iloc[surge_day_index]
        ma50 = df['close'].rolling(window=50).mean().iloc[surge_day_index]
        
        supports.append({
            'type': 'MA20',
            'price': ma20,
            'strength': 'MEDIUM',
            'description': f'20日均線 ${ma20:.2f}'
        })
        
        supports.append({
            'type': 'MA50',
            'price': ma50,
            'strength': 'STRONG',
            'description': f'50日均線 ${ma50:.2f}'
        })
        
        # 3. 前低支撐
        lookback = min(30, len(df))
        recent_lows = df['low'].iloc[surge_day_index-lookback:surge_day_index].nsmallest(3)
        
        for idx, low in enumerate(recent_lows):
            supports.append({
                'type': f'PRIOR_LOW_{idx+1}',
                'price': low,
                'strength': 'MEDIUM',
                'description': f'近期低點 ${low:.2f}'
            })
        
        # 4. 整數關卡 (動態計算基於當前價格)
        current_price = df['close'].iloc[surge_day_index]
        # 計算價格下方的整數關卡 (每5元為一個關卡)
        base_round = int(current_price / 5) * 5
        round_levels = [base_round - i * 5 for i in range(4) if base_round - i * 5 > 0]
        for level in round_levels:
            if level < current_price:
                supports.append({
                    'type': 'ROUND_NUMBER',
                    'price': level,
                    'strength': 'WEAK',
                    'description': f'整數關卡 ${level}'
                })
        
        # 按價格排序（從高到低）
        supports.sort(key=lambda x: x['price'], reverse=True)
        
        return supports
    
    def generate_trading_plan(self, df: pd.DataFrame, surge_analysis: Dict) -> Dict:
        """
        生成交易計劃
        
        Args:
            df: K線數據
            surge_analysis: 爆量分析結果
            
        Returns:
            dict: 交易計劃
        """
        plan = {
            'scenario': '',
            'action_plan': [],
            'entry_levels': [],
            'stop_loss': 0,
            'take_profit': [],
            'position_size': 0,
            'risk_reward_ratio': 0
        }
        
        current_price = df['close'].iloc[-1]
        surge_type = surge_analysis.get('surge_type', '')
        recommendation = surge_analysis.get('recommendation', 'HOLD')
        
        if surge_type == 'BEAR_TRAP':
            plan['scenario'] = '空頭陷阱，反轉在即'
            plan['action_plan'] = [
                '立即買入 30% 倉位',
                '若回測爆量日低點，加碼 20%',
                '停損設在爆量日低點下方 1%'
            ]
            
            surge_low = surge_analysis['key_levels'][0]['price']  # 爆量日低點
            plan['entry_levels'] = [
                {'price': current_price, 'size': '30%'},
                {'price': surge_low * 0.99, 'size': '20%'}
            ]
            
            plan['stop_loss'] = surge_low * 0.99
            plan['take_profit'] = [
                {'price': current_price * 1.05, 'size': '50%'},
                {'price': current_price * 1.10, 'size': '50%'}
            ]
            plan['position_size'] = 0.5  # 總倉位 50%
            plan['risk_reward_ratio'] = 2.0
            
        elif surge_type == 'DISTRIBUTION':
            plan['scenario'] = '機構出貨，趨勢轉弱'
            plan['action_plan'] = [
                '減持或清倉現有部位',
                '反彈至關鍵阻力位可建立空單',
                '嚴格停損'
            ]
            
            # 尋找阻力位
            resistance = self.find_resistance_levels(df)
            plan['entry_levels'] = [
                {'price': resistance[0]['price'], 'size': '20%', 'action': 'SELL'}
            ]
            
            plan['stop_loss'] = resistance[0]['price'] * 1.03
            plan['take_profit'] = [
                {'price': current_price * 0.95, 'size': '50%'},
                {'price': current_price * 0.90, 'size': '50%'}
            ]
            plan['position_size'] = 0.2
            plan['risk_reward_ratio'] = 1.5
            
        else:  # CONSOLIDATION 或其他
            plan['scenario'] = '盤整消化，觀望為主'
            plan['action_plan'] = [
                '暫時觀望，不建立新部位',
                '等待價格突破區間',
                '若回測支撐有效可輕倉試單'
            ]
            
            # 定義區間
            supports = [s['price'] for s in surge_analysis['key_levels'][:3]]
            resistances = self.find_resistance_levels(df)[:3]
            
            plan['entry_levels'] = [
                {'price': min(supports) * 0.99, 'size': '10%', 'condition': '支撐有效'},
                {'price': max([r['price'] for r in resistances]) * 1.01, 'size': '15%', 'condition': '突破阻力'}
            ]
            
            plan['position_size'] = 0.15
            plan['risk_reward_ratio'] = 2.0
            
        return plan
    
    def find_resistance_levels(self, df: pd.DataFrame) -> List[Dict]:
        """
        尋找關鍵阻力位
        """
        resistances = []
        
        if len(df) < 20:
            return resistances
        
        # 近期高點
        recent_highs = df['high'].iloc[-20:].nlargest(3)
        
        for idx, high in enumerate(recent_highs):
            resistances.append({
                'type': f'RECENT_HIGH_{idx+1}',
                'price': high,
                'strength': 'MEDIUM',
                'description': f'近期高點 ${high:.2f}'
            })
        
        # 移動平均線阻力
        ma20 = df['close'].rolling(window=20).mean().iloc[-1]
        resistances.append({
            'type': 'MA20',
            'price': ma20,
            'strength': 'WEAK',
            'description': f'20日均線 ${ma20:.2f}'
        })
        
        # 整數關卡 (動態計算基於當前價格)
        current_price = df['close'].iloc[-1]
        # 計算價格上方的整數關卡 (每5元為一個關卡)
        base_round = int(current_price / 5) * 5 + 5
        round_levels = [base_round + i * 5 for i in range(4)]
        for level in round_levels:
            if level > current_price:
                resistances.append({
                    'type': 'ROUND_NUMBER',
                    'price': level,
                    'strength': 'WEAK',
                    'description': f'整數關卡 ${level}'
                })
        
        # 按價格排序（從低到高）
        resistances.sort(key=lambda x: x['price'])
        
        return resistances


class VolumeSurgeFollowUpStrategy:
    """
    爆量後續跟蹤策略
    """
    
    def __init__(self):
        self.analysis_tool = PostSurgeAnalysis()
        self.watchlist = {}
        
    def reassess_google_signal(self, df: pd.DataFrame, yesterday_report: Dict) -> Dict:
        """
        重新評估 GOOG 信號，考慮爆量下跌
        """
        result = {
            'original_signal': yesterday_report.get('signal', 'BUY'),
            'reassessed_signal': 'HOLD',
            'confidence_change': 0,
            'key_findings': [],
            'trading_plan': {},
            'risk_adjustment': 0
        }
        
        # 分析爆量後續
        surge_analysis = self.analysis_tool.analyze_post_surge_pattern(df, -1)
        
        # 關鍵發現
        findings = []
        
        # 1. 爆量類型判斷
        surge_type = surge_analysis.get('surge_type', '')
        if surge_type == 'BEAR_TRAP':
            findings.append("⚠️ 爆量下跌可能是空頭陷阱，準備反轉")
            result['reassessed_signal'] = 'BUY'
            result['confidence_change'] = +10
        elif surge_type == 'DISTRIBUTION':
            findings.append("⚠️ 爆量下跌顯示機構出貨，趨勢轉弱")
            result['reassessed_signal'] = 'SELL'
            result['confidence_change'] = -20
            result['risk_adjustment'] = -15
        else:
            findings.append("爆量後盤整消化，需要更多時間確認方向")
            result['reassessed_signal'] = 'HOLD'
        
        # 2. 價格行為分析
        price_change = surge_analysis.get('price_change_since_surge', 0)
        if price_change > 0:
            findings.append(f"✅ 爆量後已上漲 {price_change:.1%}，顯示買盤承接")
        elif price_change < 0:
            findings.append(f"❌ 爆量後繼續下跌 {abs(price_change):.1%}，賣壓持續")
        
        # 3. 支撐位檢查
        if surge_analysis.get('support_test', False):
            findings.append("✅ 價格已測試關鍵支撐，反彈機率高")
            if result['reassessed_signal'] == 'HOLD':
                result['reassessed_signal'] = 'BUY_ON_DIP'
        else:
            findings.append("⚠️ 價格尚未測試關鍵支撐，下跌風險仍存")
        
        # 4. 量能分析
        volume_pattern = surge_analysis.get('volume_pattern', '')
        if volume_pattern == 'VOLUME_NORMALIZATION':
            findings.append("✅ 量能已恢復正常，恐慌情緒緩解")
        elif volume_pattern == 'CONTINUED_HIGH_VOLUME':
            findings.append("❌ 量能持續偏高，波動可能繼續")
        
        result['key_findings'] = findings
        
        # 生成交易計劃
        if surge_analysis.get('surge_type'):
            result['trading_plan'] = self.analysis_tool.generate_trading_plan(df, surge_analysis)
        
        return result
    
    def generate_recommendation_report(self, analysis_result: Dict) -> str:
        """
        生成推薦報告
        """
        report = []
        report.append("=" * 70)
        report.append("爆量下跌後續分析報告 - GOOG (Google)")
        report.append("=" * 70)
        
        report.append(f"\n📊 信號重新評估:")
        report.append(f"   原始信號: {analysis_result['original_signal']}")
        report.append(f"   重新評估: {analysis_result['reassessed_signal']}")
        report.append(f"   信心度變化: {analysis_result['confidence_change']:+d}")
        
        report.append(f"\n🔍 關鍵發現:")
        for finding in analysis_result['key_findings']:
            report.append(f"   • {finding}")
        
        # 交易計劃
        if analysis_result['trading_plan']:
            plan = analysis_result['trading_plan']
            report.append(f"\n🎯 交易計劃:")
            report.append(f"   情境: {plan['scenario']}")
            
            report.append(f"\n   📋 行動步驟:")
            for step in plan['action_plan']:
                report.append(f"      • {step}")
            
            report.append(f"\n   🎯 進場點:")
            for entry in plan['entry_levels']:
                price = entry.get('price', 0)
                size = entry.get('size', '')
                action = entry.get('action', 'BUY')
                condition = entry.get('condition', '')
                
                if condition:
                    report.append(f"      • ${price:.2f} ({size}) - {action} {condition}")
                else:
                    report.append(f"      • ${price:.2f} ({size}) - {action}")
            
            if plan['stop_loss']:
                report.append(f"\n   🛡️ 停損點: ${plan['stop_loss']:.2f}")
            
            if plan['take_profit']:
                report.append(f"   🎯 停利點:")
                for tp in plan['take_profit']:
                    report.append(f"      • ${tp['price']:.2f} ({tp['size']})")
        
        # 風險提示
        report.append(f"\n⚠️ 風險提示:")
        report.append("   1. 爆量下跌後波動通常較大")
        report.append("   2. 需要確認是否為假跌破")
        report.append("   3. 建議分批進場，嚴格停損")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


# ======================================================
# 主程序：爆量下跌後再評估
# ======================================================

def analyze_single_stock(ticker: str, strategy: VolumeSurgeFollowUpStrategy) -> Dict:
    """
    分析單一股票的爆量後續

    Args:
        ticker: 股票代碼
        strategy: 策略實例

    Returns:
        dict: 分析結果
    """
    import yfinance as yf

    result = {
        'ticker': ticker,
        'success': False,
        'signal': 'N/A',
        'confidence': 0,
        'current_price': 0,
        'entry_low': 0,
        'entry_high': 0,
        'stop_loss': 0,
        'target_1': 0,
        'target_2': 0,
        'risk_level': 'N/A',
        'message': ''
    }

    try:
        print(f"\n{'='*70}")
        print(f"分析 {ticker}")
        print(f"{'='*70}")

        # 獲取數據
        stock = yf.Ticker(ticker)
        df = stock.history(period="3mo")

        if df.empty or len(df) < 20:
            result['message'] = f"數據不足 ({len(df) if not df.empty else 0} 筆)"
            print(f"  [X] {result['message']}")
            return result

        # 轉換列名為小寫
        df.columns = [c.lower() for c in df.columns]

        # 計算基本數據
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        price_change = (current_price - prev_price) / prev_price

        # 計算量比
        avg_vol_5d = df['volume'].iloc[-6:-1].mean()
        volume_ratio = df['volume'].iloc[-1] / avg_vol_5d if avg_vol_5d > 0 else 1

        # 動態生成報告
        yesterday_report = {
            'signal': 'BUY' if price_change > 0 else 'SELL',
            'price': prev_price,
            'volume_ratio': volume_ratio,
            'price_change': price_change,
            'warning': f"量能變化 (量比: {volume_ratio:.1f}x, 漲跌幅: {price_change:.1%})"
        }

        # 重新評估
        reassessment = strategy.reassess_google_signal(df, yesterday_report)

        # 分析爆量後續模式
        analyzer = PostSurgeAnalysis()
        surge_analysis = analyzer.analyze_post_surge_pattern(df, -2)

        # 動態計算關鍵價位
        recent_low = df['low'].iloc[-20:].min()
        surge_low = df['low'].iloc[-2]
        recent_high = df['high'].iloc[-20:].max()

        # 支撐位
        support_1 = round(surge_low, 2)
        support_2 = round(recent_low * 0.98, 2)

        # 計算 ATR (平均真實波幅) 用於動態停損
        df_temp = df.copy()
        df_temp['tr'] = np.maximum(
            df_temp['high'] - df_temp['low'],
            np.maximum(
                abs(df_temp['high'] - df_temp['close'].shift(1)),
                abs(df_temp['low'] - df_temp['close'].shift(1))
            )
        )
        atr = df_temp['tr'].rolling(window=14).mean().iloc[-1]
        atr_pct = atr / current_price  # ATR 佔價格比例

        # 進出場價位
        entry_low = round(current_price * 0.97, 2)
        entry_high = round(current_price * 0.99, 2)

        # 停損計算: 取 ATR 2倍 或 8% 中較小者，但不超過 10%
        atr_stop = current_price * (1 - min(atr_pct * 2, 0.10))
        pct_stop = current_price * 0.92  # 固定 8% 停損
        stop_loss = round(max(atr_stop, pct_stop, recent_low * 0.98), 2)

        target_1 = round(current_price * 1.05, 2)
        target_2 = round(current_price * 1.10, 2)

        # 風險評估
        risk = surge_analysis.get('risk_assessment', {})
        if risk.get('high_risk'):
            risk_level = 'HIGH'
        elif risk.get('moderate_risk'):
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        # 填充結果
        result['success'] = True
        result['signal'] = reassessment['reassessed_signal']
        result['confidence'] = reassessment.get('confidence_change', 0)
        result['current_price'] = round(current_price, 2)
        result['entry_low'] = entry_low
        result['entry_high'] = entry_high
        result['stop_loss'] = stop_loss
        result['target_1'] = target_1
        result['target_2'] = target_2
        result['risk_level'] = risk_level
        result['surge_type'] = surge_analysis.get('surge_type', 'N/A')
        result['volume_ratio'] = round(volume_ratio, 2)
        result['price_change'] = round(price_change * 100, 2)

        # 輸出詳細分析
        print(f"  當前價格: ${current_price:.2f}")
        print(f"  漲跌幅: {price_change:.2%}")
        print(f"  量比: {volume_ratio:.2f}x")
        print(f"  爆量類型: {surge_analysis.get('surge_type', 'N/A')}")
        print(f"  信號: {reassessment['reassessed_signal']}")
        print(f"  風險等級: {risk_level}")
        print(f"  關鍵支撐: ${support_1:.2f} / ${support_2:.2f}")
        print(f"  進場區間: ${entry_low:.2f} - ${entry_high:.2f}")
        print(f"  停損: ${stop_loss:.2f}")
        print(f"  目標: ${target_1:.2f} - ${target_2:.2f}")

        result['message'] = 'OK'

    except Exception as e:
        result['message'] = str(e)
        print(f"  [X] 錯誤: {e}")

    return result


def run_batch_analysis(tickers: List[str]) -> List[Dict]:
    """
    批次分析多檔股票

    Args:
        tickers: 股票代碼列表

    Returns:
        list: 所有分析結果
    """
    print("\n" + "=" * 70)
    print("批次爆量後續分析")
    print("=" * 70)
    print(f"分析 {len(tickers)} 檔股票: {', '.join(tickers)}")

    strategy = VolumeSurgeFollowUpStrategy()
    results = []

    for ticker in tickers:
        result = analyze_single_stock(ticker, strategy)
        results.append(result)

    # 輸出總結報告
    print("\n" + "=" * 70)
    print("批次分析總結")
    print("=" * 70)
    print(f"{'股票':<12} {'價格':>10} {'漲跌%':>8} {'量比':>6} {'信號':<12} {'風險':<8} {'進場區間':<20}")
    print("-" * 90)

    buy_signals = []

    for r in results:
        if r['success']:
            signal_str = r['signal']
            entry_str = f"${r['entry_low']:.2f}-${r['entry_high']:.2f}"
            print(f"{r['ticker']:<12} ${r['current_price']:>8.2f} {r['price_change']:>7.2f}% {r['volume_ratio']:>5.2f}x {signal_str:<12} {r['risk_level']:<8} {entry_str:<20}")

            # 收集買入信號
            if r['signal'] in ['BUY', 'BUY_ON_DIP', 'BUY_AT_SUPPORT']:
                buy_signals.append(r)
        else:
            print(f"{r['ticker']:<12} {'N/A':>10} {'N/A':>8} {'N/A':>6} {'ERROR':<12} {'N/A':<8} {r['message']:<20}")

    # 買入建議
    if buy_signals:
        print("\n" + "=" * 70)
        print("買入建議股票")
        print("=" * 70)
        for r in buy_signals:
            print(f"\n[{r['ticker']}] 信號: {r['signal']}")
            print(f"   現價: ${r['current_price']:.2f}")
            print(f"   進場: ${r['entry_low']:.2f} - ${r['entry_high']:.2f}")
            print(f"   停損: ${r['stop_loss']:.2f}")
            print(f"   目標: ${r['target_1']:.2f} - ${r['target_2']:.2f}")
            print(f"   風險: {r['risk_level']}")
    else:
        print("\n目前無明確買入信號，建議觀望。")

    return results


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    # 批次分析股票清單
    WATCH_LIST = [
        "GOOG",      # Google
        "NVDA",      # NVIDIA
        "SNDK",       # Western Digital (SNDK 已被收購)
        "MU",        # Micron
        "6770.TW",   # 力積電 (台股)
    ]

    # 執行批次分析
    results = run_batch_analysis(WATCH_LIST)