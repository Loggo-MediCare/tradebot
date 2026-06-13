"""
台股 7805 (OTC) AI 交易信号生成器
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from dynamic_signal_weights import DynamicWeightCalculator
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output, calculate_sentiment_score
from candlestick_patterns import analyze_candlestick_patterns, format_pattern_output, get_pattern_score_adjustment
from ma50_slope_analysis import calculate_ma50_slope, format_ma50_slope_output, get_ma50_slope_score_adjustment
from model_accuracy_tracker import ModelAccuracyTracker, get_model_accuracy_display, should_mute_ai_signal
from structure_pattern_analysis import detect_structure_patterns, format_structure_pattern_output, get_structure_score_adjustment
from triangle_pattern import detect_triangle, triangle_breakout
from breakout_detector import get_breakout_signal
from pattern_engine import get_pattern_signal
from volume_surge_detector import get_volume_signal
from shared_market_checks import evaluate_fundamentals_for_sell, calculate_obv as shared_calculate_obv, money_flow_strength as shared_money_flow_strength
from breakout_long_red import get_breakout_long_red_signal
from chart_visualizer import plot_candlestick

TICKER = '7805.TWO'
MODEL_PATH = r"C:\Users\Silvi\Projects\trading-bot\ppo_7805_two_improved"

class ImprovedTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        price = float(row['close'])
        total = self.balance + self.shares_held * price
        return np.array([float(self.shares_held), float(self.balance), price,
            float(row.get('sma_10', 0)), float(row.get('sma_30', 0)), float(row.get('sma_50', 0)),
            float(row.get('rsi', 50)), float(row.get('macd', 0)), float(row.get('macd_signal', 0)),
            float(row.get('bb_upper', 0)), float(row.get('bb_lower', 0)), float(row.get('volume', 0)),
            float(self.total_profit), (self.shares_held * price) / total if total > 0 else 0,
            self.balance / total if total > 0 else 1], dtype=np.float32)

    def step(self, action):
        action = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        action = np.clip(action, -1.0, 1.0)
        price = float(self.df.iloc[self.current_step]['close'])
        if action < -0.1 and self.shares_held > 0:
            sell = int(self.shares_held * abs(action))
            if sell > 0: self.balance += sell * price; self.shares_held -= sell
        elif action > 0.1:
            buy = int((self.balance // price) * action)
            if buy > 0: self.balance -= buy * price; self.shares_held += buy
        self.total_profit = (self.balance + self.shares_held * price) - self.initial_balance
        self.current_step += 1
        return self._get_observation(), float(self.total_profit / self.initial_balance), self.current_step >= len(self.df) - 1, False, {}

def add_technical_indicators(df):
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    return df.bfill().ffill()

# 使用共享版資金流邏輯，確保所有股票一致
calculate_obv = shared_calculate_obv
money_flow_strength = shared_money_flow_strength

def get_trading_signal():
    print("=" * 80)
    print(f"AI Trading Signal - {TICKER}")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Accuracy: {get_model_accuracy_display(TICKER)}")
    print("=" * 80)

    try:
        model = PPO.load(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model load failed: {e}")
        return None

    print("\nDownloading market data...")
    try:
        import yfinance as yf
        df = yf.download(TICKER, period='90d', progress=False, auto_adjust=True)
        if df.empty:
            print("No data available")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Close': 'close', 'Volume': 'volume', 'Open': 'open', 'High': 'high', 'Low': 'low'}).reset_index()
        print(f"Downloaded {len(df)} days of data")

        target_price, target_high, recommendation_mean = None, None, None
        try:
            info = yf.Ticker(TICKER).info
            target_price = info.get('targetMeanPrice')
            target_high = info.get('targetHighPrice')
            recommendation_mean = info.get('recommendationMean')
            if target_price:
                print(f"   Analyst Target: NT${target_price:.2f}")
        except:
            pass
    except Exception as e:
        print(f"Data download failed: {e}")
        return None

    df = add_technical_indicators(df)
    latest = df.iloc[-1]
    try:
        latest_date = str(pd.to_datetime(df.iloc[-1]['Date']).date())
    except:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price = float(latest['close'])
    rsi = float(latest['rsi'])
    macd = float(latest['macd'])
    macd_signal = float(latest['macd_signal'])
    sma_10 = float(latest['sma_10'])
    sma_30 = float(latest['sma_30'])
    sma_50 = float(latest['sma_50'])
    bb_upper = float(latest['bb_upper'])
    bb_lower = float(latest['bb_lower'])
    volume = float(latest['volume'])
    avg_vol = float(df['volume'].tail(20).mean())
    volume_ratio = volume / avg_vol if avg_vol > 0 else 1.0

    print(f"\nLatest Date: {latest_date}")
    print(f"Price: NT${current_price:.2f}")
    print(f"Volume: {int(volume):,} ({volume_ratio:.2f}x avg)")

    print("\n" + "=" * 80)
    print("Technical Indicators")
    print("=" * 80)
    print(f"RSI (10):     {rsi:.2f}  {'[Overbought]' if rsi > 70 else '[Oversold]' if rsi < 30 else '[Neutral]'}")
    print(f"MACD:         {macd:.4f}  {'[Bullish]' if macd > macd_signal else '[Bearish]'}")
    print(f"SMA 10/30:    NT${sma_10:.2f} / NT${sma_30:.2f}  {'[Uptrend]' if sma_10 > sma_30 else '[Downtrend]'}")
    print(f"BB Range:     NT${bb_lower:.2f} - NT${bb_upper:.2f}")

    # MA50 Slope
    print("\n" + "=" * 80)
    print("MA50 Trend Analysis")
    print("=" * 80)
    try:
        profit_positive = evaluate_fundamentals_for_sell(yf, TICKER)['good']
    except Exception:
        profit_positive = None
    ma50_info = calculate_ma50_slope(
        df['close'],
        window=50,
        slope_period=5,
        current_rsi=rsi,
        profit_positive=profit_positive
    )
    print(f"MA50 Slope: {ma50_info['slope_pct']:+.4f}% - {ma50_info['trend']}")

    # Sentiment
    print("\n" + "=" * 80)
    print("Market Sentiment (FinBERT)")
    print("=" * 80)
    sentiment_result = calculate_sentiment_score(TICKER, verbose=False)
    if sentiment_result and sentiment_result['news_count'] > 0:
        pass  # 情緒分析結果已計算，輸出已移至後續統一顯示
    else:
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}
    print("\n" + "=" * 80)
    print("Candlestick Patterns")
    print("=" * 80)
    try:
        patterns = analyze_candlestick_patterns(df, days=5)
        print(format_pattern_output(patterns))
        pattern_adj = get_pattern_score_adjustment(patterns)
    except:
        pattern_adj = 0

    # Structure
    print("\n" + "=" * 80)
    print("Structure Patterns")
    print("=" * 80)
    try:
        structure = detect_structure_patterns(df, window=60)
        print(format_structure_pattern_output(structure))
        structure_adj = get_structure_score_adjustment(structure)
    except:
        structure_adj = 0

    # AI Prediction
    env = ImprovedTradingEnv(df)
    env.current_step = len(df) - 1
    obs = env._get_observation()
    action, _ = model.predict(obs, deterministic=True)
    action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    ai_muted = should_mute_ai_signal(TICKER, threshold=52)
    if ai_muted:
        print("⚠️  AI模型準確度低於52%，已靜音AI交易動作（action=0）")
        action_value = 0.0

    print("\n" + "=" * 80)
    print("AI Trading Signal")
    print("=" * 80)
    print(f"Model Action: {action_value:+.4f}")

    weight_calc = DynamicWeightCalculator(TICKER)
    buy_weights = weight_calc.get_buy_weights()
    sell_weights = weight_calc.get_sell_weights()

    if action_value > 0.1:
        signal = "BUY"
        buy_score, _, reasons, warnings, _, _ = calculate_enhanced_buy_score_with_sentiment(
            rsi=rsi, macd=macd, macd_signal=macd_signal, sma_10=sma_10, sma_30=sma_30,
            current_price=current_price, bb_upper=bb_upper, bb_lower=bb_lower,
            volume_ratio=volume_ratio, ai_action=action_value, buy_weights=buy_weights, symbol=TICKER)
        buy_score += get_ma50_slope_score_adjustment(ma50_info) + structure_adj
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


        buy_score = max(0, min(100, buy_score))
        print(f"\n{'BUY' if buy_score >= 20 else 'WAIT'} Signal")
        print(f"   Strength: {action_value:.2f}")
        print(f"   Score: {buy_score}/100")
        if reasons:
            print(f"   Reasons: {', '.join(reasons[:3])}")
    elif action_value < -0.1:
        signal = "SELL"
        sell_score = 50
        bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
        if macd < macd_signal: sell_score += 15
        if sma_10 < sma_30: sell_score += 15
        if bb_pos > 80: sell_score += 10
        if rsi > 70: sell_score += 10
        sell_score = max(0, min(100, sell_score))
        print(f"\nSELL Signal")
        print(f"   Strength: {abs(action_value):.2f}")
        print(f"   Score: {sell_score}/100")
    else:
        signal = "HOLD"
        print(f"\nHOLD Signal - No action recommended")

    print("\n" + "=" * 80)
    print("Risk Warning: AI-generated signals are for reference only")
    print("=" * 80)

    return {'date': latest_date, 'symbol': TICKER, 'price': current_price, 'signal': signal, 'action': action_value}

if __name__ == "__main__":
    result = get_trading_signal()
    if result:
        print(f"\nSummary: {result['symbol']} @ NT${result['price']:.2f} -> {result['signal']}")

