"""美股 BKR (Vertex Pharma) AI 交易信号生成器"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, joblib, yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from dynamic_signal_weights import DynamicWeightCalculator
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, calculate_sentiment_score
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
from ma50_slope_analysis import calculate_ma50_slope, get_ma50_slope_score_adjustment
from model_accuracy_tracker import get_model_accuracy_display, should_mute_ai_signal
from breakout_detector import get_breakout_signal
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
from shared_market_checks import (evaluate_fundamentals_for_sell,
    calculate_obv, money_flow_strength, calculate_growth_score_adjustment)

SYMBOL = 'VRTX'
MODEL_FILE = 'xgb_vrtx_model.pkl'
FEATURE_COLS = ['rsi','macd','macd_signal','macd_hist','bb_position','K','D',
                'obv','obv_ma20','sma_10','sma_30','sma_50','sma_200',
                'volatility','atr','price_change_5d','price_change_10d',
                'price_change_20d','ma50_slope']

def add_indicators(df):
    df['sma_10']  = df['close'].rolling(10).mean()
    df['sma_30']  = df['close'].rolling(30).mean()
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12']  = df['close'].ewm(span=12).mean()
    df['ema_26']  = df['close'].ewm(span=26).mean()
    d = df['close'].diff()
    g = d.where(d > 0, 0).rolling(14).mean()
    l = (-d.where(d < 0, 0)).rolling(14).mean()
    df['rsi']         = 100 - (100 / (1 + g / (l + 1e-10)))
    df['macd']        = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist']   = df['macd'] - df['macd_signal']
    df['bb_m'] = df['close'].rolling(20).mean()
    df['bb_s'] = df['close'].rolling(20).std()
    df['bb_upper']    = df['bb_m'] + 2 * df['bb_s']
    df['bb_lower']    = df['bb_m'] - 2 * df['bb_s']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100).fillna(50)
    lo14 = df['low'].rolling(14).min()
    hi14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - lo14) / (hi14 - lo14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()
    df = calculate_obv(df)
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
    hl = df['high'] - df['low']
    hc = np.abs(df['high'] - df['close'].shift())
    lc = np.abs(df['low']  - df['close'].shift())
    df['atr']             = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['price_change_5d']  = df['close'].pct_change(5)  * 100
    df['price_change_10d'] = df['close'].pct_change(10) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100
    df['ma50_slope']       = df['sma_50'].diff(5) / df['sma_50'].shift(5) * 100
    return df.bfill().ffill()

def get_trading_signal():
    print("=" * 80)
    print(f"🤖 美股 {SYMBOL} (Vertex Pharma) AI 交易信号生成器")
    print("=" * 80)
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型準確度: {get_model_accuracy_display(SYMBOL)}")
    print("=" * 80)

    try:
        model = joblib.load(MODEL_FILE)
        print(f"✅ XGBoost 模型加载成功!")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}"); return None

    try:
        df = yf.download(SYMBOL, period='300d', progress=False, auto_adjust=True)
        if df.empty: print("❌ 无法获取数据"); return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close':'close','Volume':'volume','Open':'open','High':'high','Low':'low'}).reset_index()
        print(f"✅ 成功下载 {len(df)} 天数据")
        target_price = target_high = recommendation_mean = recommendation_key = None
        try:
            info = yf.Ticker(SYMBOL).info
            target_price        = info.get('targetMeanPrice')
            target_high         = info.get('targetHighPrice')
            recommendation_mean = info.get('recommendationMean')
            recommendation_key  = info.get('recommendationKey', '')
            num_analysts        = info.get('numberOfAnalystOpinions', 0)
            if target_price and num_analysts > 0:
                print(f"   📊 分析師目標價: ${target_price:.2f} / 評級: {recommendation_key}")
        except:
            pass
    except Exception as e:
        print(f"❌ 数据下载失败: {e}"); return None

    df = add_indicators(df)
    latest = df.iloc[-1]
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price   = float(latest['close'])
    rsi             = float(latest['rsi'])
    macd            = float(latest['macd'])
    macd_signal_val = float(latest['macd_signal'])
    sma_10          = float(latest['sma_10'])
    sma_30          = float(latest['sma_30'])
    bb_upper        = float(latest['bb_upper'])
    bb_lower        = float(latest['bb_lower'])
    current_volume  = float(latest['volume'])
    avg_volume_20   = float(df['volume'].tail(20).mean())
    volume_ratio    = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

    print(f"\n✅ 最新数据: {latest_date}  价格: ${current_price:.2f}  量比: {volume_ratio:.2f}x")
    print(f"   RSI: {rsi:.1f}  MACD: {macd:.4f}  SMA10: ${sma_10:.2f}  SMA30: ${sma_30:.2f}")

    feat       = df[FEATURE_COLS].iloc[[-1]]
    prediction = model.predict(feat)[0]
    proba      = model.predict_proba(feat)[0]
    if prediction == 1:
        action_value = max(0.1, float(proba[1] * 2 - 1))
    else:
        action_value = min(-0.1, -float(proba[0] * 2 - 1))
    print(f"\n🧠 XGBoost: {'买入' if prediction == 1 else '观望/卖出'}  概率: {proba[1]*100:.2f}%  动作值: {action_value:+.4f}")

    if should_mute_ai_signal(SYMBOL, threshold=52):
        print("⚠️  AI准确度低于52%，已静音AI信号")
        action_value = 0.0

    try:
        profit_positive = evaluate_fundamentals_for_sell(yf, SYMBOL)['good']
    except:
        profit_positive = None

    ma50_info = calculate_ma50_slope(df['close'], window=50, slope_period=5,
                                     current_rsi=rsi, profit_positive=profit_positive)
    print(f"\n📈 MA50斜率: {ma50_info['slope_pct']:+.4f}%  趋势: {ma50_info['trend']}")

    weight_calc = DynamicWeightCalculator(SYMBOL)
    buy_weights  = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

    sentiment_result = calculate_sentiment_score(SYMBOL, verbose=True)
    if not sentiment_result or sentiment_result.get('news_count', 0) == 0:
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}

    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        pattern_adjustment = get_pattern_score_adjustment(patterns)
    except:
        pattern_adjustment = 0

    strong_money, vol_ratio, flow_details = money_flow_strength(df)
    capital_inflow = flow_details.get('capital_inflow', False) if flow_details else False
    obv_bullish    = flow_details.get('obv_bullish', False)    if flow_details else False
    print(f"\n💰 即时资金流入: {'✅ 放量收阳' if capital_inflow else '❌ 未确认'}  OBV: {'bullish' if obv_bullish else 'bearish'}")

    print("\n" + "=" * 80)
    print("🎯 AI 交易信号")
    print("=" * 80)

    if action_value > 0.1:
        signal = "买入 (BUY)"; signal_emoji = "🟢"; strength = action_value
        suggested_price_low  = current_price * 0.995
        suggested_price_high = current_price * 1.000

        buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = \
            calculate_enhanced_buy_score_with_sentiment(
                rsi=rsi, macd=macd, macd_signal=macd_signal_val,
                sma_10=sma_10, sma_30=sma_30, current_price=current_price,
                bb_upper=bb_upper, bb_lower=bb_lower, volume_ratio=volume_ratio,
                ai_action=action_value, buy_weights=buy_weights, symbol=SYMBOL)

        buy_score += get_ma50_slope_score_adjustment(ma50_info) + pattern_adjustment

        bs = get_breakout_signal(df)
        if bs['detected']:
            if bs['type'] == 'TRUE_BREAKOUT':
                buy_score += 15; buy_reasons.append(bs['signal_text'])
            elif bs['type'] == 'FALSE_BREAKOUT':
                buy_score -= 10; buy_warnings.append(bs['signal_text'])

        ps = get_pattern_signal(df)
        if ps['patterns']:
            if ps['score_adjustment'] > 0:
                buy_score += ps['score_adjustment']; buy_reasons.append(f"型態: {ps['signal_text']}")
            elif ps['score_adjustment'] < 0:
                buy_warnings.append(f"型態警示: {ps['signal_text']}")

        vs = get_volume_signal(df)
        if vs['surge'] and vs['surge']['detected']:
            if vs['surge']['type'] == 'SURGE_UP':
                buy_score += 15; buy_reasons.append(vs['surge']['signal_text'])
            elif vs['surge']['type'] == 'SURGE_DOWN':
                buy_warnings.append(vs['surge']['signal_text'])

        buy_score = max(0, min(100, buy_score))
        adjusted_buy_strength = max(min((buy_score / 100) * strength, 1.0), 0)
        suggested_buy_ratio   = int(adjusted_buy_strength * 100)
        if buy_score < 20: signal = "观望 (WAIT)"; signal_emoji = "🟡"

        print(f"\n{signal_emoji} 信号: {signal}")
        print(f"   AI强度: {strength:.2f}  评分: {buy_score}/100")
        if buy_score >= 20:
            print(f"   建议买入比例: {suggested_buy_ratio}%")
            print(f"   建议价格区间: ${suggested_price_low:.2f} - ${suggested_price_high:.2f}")
        if buy_warnings:
            print("\n   ⚠️  警告:")
            for w in buy_warnings: print(f"      • {w}")
        if buy_reasons:
            print("\n   📌 买入理由:")
            for i, r in enumerate(buy_reasons, 1): print(f"      {i}. {r}")

    elif action_value < -0.1:
        signal = "卖出 (SELL)"; signal_emoji = "🔴"; strength = abs(action_value)
        suggested_price_low  = current_price * 1.000
        suggested_price_high = current_price * 1.005
        sell_score = 0; reasons = []

        ma50_adj = get_ma50_slope_score_adjustment(ma50_info)
        if ma50_adj < 0:
            sell_score += abs(ma50_adj); reasons.append("MA50趨勢向下")

        if target_price and current_price > 0:
            upside = (target_price - current_price) / current_price * 100
            if upside < -10:
                sell_score += sell_weights.get('target_below', 20)
                reasons.append(f"目標價低於現價 {upside:.1f}%")
            elif upside < 5:
                sell_score += sell_weights.get('target_near', 10)
                reasons.append(f"上漲空間有限 ({upside:.1f}%)")

        if macd < macd_signal_val:
            sell_score += sell_weights.get('macd_bearish', 15); reasons.append("MACD 死叉")
        if sma_10 < sma_30:
            sell_score += sell_weights.get('ma_bearish', 10); reasons.append("均線空頭排列")

        bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        if bb_pos > 90:
            sell_score += sell_weights.get('bb_upper', 20); reasons.append(f"接近布林帶上軌 ({bb_pos:.1f}%)")
        elif bb_pos > 80:
            sell_score += sell_weights.get('bb_high', 10); reasons.append("偏高接近上軌")

        is_strong = (macd >= macd_signal_val) and (sma_10 >= sma_30) and (volume_ratio > 1.2)
        if volume_ratio > 2.5 and rsi < 80 and is_strong:
            sell_score = 0; reasons = ["🚀 超強勢突破，不賣出"]
        elif volume_ratio < 0.5 and current_price > sma_10:
            sell_score += 15; reasons.append(f"價漲量縮 ({volume_ratio:.1f}x)")

        _growth = calculate_growth_score_adjustment(yf, SYMBOL)
        if _growth['adjustment'] > 0 and sell_score > 0:
            sell_score = max(0, sell_score - _growth['adjustment'])
            for g in _growth['reasons']: reasons.append(f'🌱 {g}')

        adjusted_strength    = min(sell_score / 100, 1.0)
        suggested_sell_ratio = int(adjusted_strength * 100)
        if sell_score == 0: signal = "持有 (HOLD - 强势突破)"; signal_emoji = "🟢"

        print(f"\n{signal_emoji} 信号: {signal}")
        print(f"   AI强度: {strength:.2f}  评分: {sell_score}/100")
        if sell_score > 0:
            print(f"   建议卖出比例: {suggested_sell_ratio}%")
            print(f"   建议价格区间: ${suggested_price_low:.2f} - ${suggested_price_high:.2f}")
        if reasons:
            label = "持有理由" if sell_score == 0 else "卖出理由"
            print(f"\n   📌 {label}:")
            for i, r in enumerate(reasons, 1): print(f"      {i}. {r}")
    else:
        signal = "持有 (HOLD)"; signal_emoji = "🟡"
        print(f"\n{signal_emoji} 信号: {signal}")
        print("   市场观望，暂不操作")

    print("\n" + "=" * 80)
    print(f"   {get_model_accuracy_display(SYMBOL)}")
    return {'symbol': SYMBOL, 'signal': signal, 'current_price': current_price,
            'rsi': rsi, 'date': latest_date}

if __name__ == '__main__':
    result = get_trading_signal()
    if result:
        print(f"\n📱 快速摘要:")
        print(f"   股票: {result['symbol']}")
        print(f"   日期: {result['date']}")
        print(f"   价格: ${result['current_price']:.2f}")
        print(f"   信号: {result['signal']}")
    else:
        print("\n❌ 信号生成失败")
