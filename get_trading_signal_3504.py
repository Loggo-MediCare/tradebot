"""
台股 3504 (YOUNG OPTICS INC) AI 交易信号生成器 (PPO / DQN 自動選最佳)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MPLBACKEND'] = 'Agg'

import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from chart_visualizer import plot_candlestick
from tavily_news import print_tavily_news
from model_accuracy_tracker import get_model_accuracy_display, get_best_model_type, get_best_model_display
from backtest_utils import calculate_ppo_backtest_roi, print_ppo_action_line

TICKER      = '3504.TW'
TICKER_NAME = 'YOUNG OPTICS INC'
DQN_PATH    = r'C:\Users\Silvi\Projects\trading-bot\3504.TW_improved_anti_overfit_model.keras'
PPO_PATH    = r'C:\Users\Silvi\Projects\trading-bot\ppo_3504_tw_improved'
WINDOW_SIZE  = 5
FEATURE_COLS = ['SMA_5','SMA_10','SMA_20','SMA_50','ROC_5','ROC_10',
                'RSI','Volatility_10','Volatility_20','MACD','Signal_Line',
                'BB_position','Volume_ratio']


def create_features(df):
    f = df.copy()
    f['SMA_5']  = f['Close'].rolling(5).mean()
    f['SMA_10'] = f['Close'].rolling(10).mean()
    f['SMA_20'] = f['Close'].rolling(20).mean()
    f['SMA_50'] = f['Close'].rolling(50).mean()
    f['ROC_5']  = f['Close'].pct_change(5) * 100
    f['ROC_10'] = f['Close'].pct_change(10) * 100
    delta = f['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    f['RSI'] = 100 - (100 / (1 + gain / loss))
    f['Volatility_10'] = f['Close'].pct_change().rolling(10).std()
    f['Volatility_20'] = f['Close'].pct_change().rolling(20).std()
    ema12 = f['Close'].ewm(span=12, adjust=False).mean()
    ema26 = f['Close'].ewm(span=26, adjust=False).mean()
    f['MACD']        = ema12 - ema26
    f['Signal_Line'] = f['MACD'].ewm(span=9, adjust=False).mean()
    sma20 = f['Close'].rolling(20).mean()
    std20 = f['Close'].rolling(20).std()
    bb_up = sma20 + std20 * 2
    bb_lo = sma20 - std20 * 2
    f['BB_position']  = (f['Close'] - bb_lo) / (bb_up - bb_lo)
    f['Volume_SMA']   = f['Volume'].rolling(20).mean()
    f['Volume_ratio'] = f['Volume'] / f['Volume_SMA']
    f = f.dropna()
    scaler = MinMaxScaler()
    f[FEATURE_COLS] = scaler.fit_transform(f[FEATURE_COLS])
    return f


def get_state(df_feat, step, balance=10000, shares=0, initial_balance=10000):
    market_state = df_feat[FEATURE_COLS].iloc[step-WINDOW_SIZE:step].values.flatten()
    current_price = float(df_feat['Close'].iloc[step])
    portfolio_value = balance + shares * current_price
    account_state = np.array([
        balance / initial_balance,
        shares / 100,
        portfolio_value / initial_balance,
        (portfolio_value - initial_balance) / initial_balance,
        0.0,
    ])
    return np.concatenate([market_state, account_state])


def load_best_model():
    best_type, score_ppo, score_dqn, score_xgb, score_hyb = get_best_model_type(TICKER)
    order = [best_type, 'DQN' if best_type == 'PPO' else 'PPO']
    for mtype in order:
        try:
            if mtype == 'DQN' and os.path.exists(DQN_PATH):
                m = tf.keras.models.load_model(DQN_PATH)
                print(f"DQN 模型加载成功 (score: {score_dqn})")
                return m, 'DQN'
            elif mtype == 'PPO' and os.path.exists(PPO_PATH + '.zip'):
                from stable_baselines3 import PPO as PPO_cls
                m = PPO_cls.load(PPO_PATH)
                print(f"PPO 模型加载成功 (score: {score_ppo})")
                return m, 'PPO'
        except Exception as e:
            print(f"  {mtype} 加载失败: {e}")
    return None, None


def get_trading_signal():
    print("=" * 80)
    print(f"AI 3504.TW (YOUNG OPTICS INC) 交易信号生成器")
    print(f"模型準確度: {get_model_accuracy_display(TICKER)}")
    print(get_best_model_display(TICKER))
    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    model, model_type = load_best_model()
    if model is None:
        print("所有模型加载失败"); return None
    print(f"使用模型: {model_type}")

    print("\n下载市场数据...")
    try:
        import yfinance as yf
        df = yf.download(TICKER, period='200d', progress=False, auto_adjust=True)
        if df.empty:
            print("无法获取数据"); return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.reset_index()
        print(f"成功下载 {len(df)} 天数据")
    except Exception as e:
        print(f"数据下载失败: {e}"); return None

    df_feat = create_features(df)
    if len(df_feat) < WINDOW_SIZE + 1:
        print("数据不足"); return None

    try:
        latest_date = str(pd.to_datetime(df_feat.iloc[-1]['Date']).date())
    except Exception:
        latest_date = datetime.now().strftime('%Y-%m-%d')

    current_price = float(df_feat['Close'].iloc[-1])
    print(f"最新日期: {latest_date}  价格: NT${current_price:.2f}")

    step     = len(df_feat) - 1
    state    = get_state(df_feat, step)
    _mtype = locals().get('model_type', 'DQN')
    _ppo_roi, _bh_roi, _av = None, None, None
    if _mtype == 'PPO':
        _row = df_feat.iloc[-1] if 'df_feat' in dir() else df.iloc[-1]
        _p   = current_price
        _ppo_obs = np.array([
            0.0, 10000.0, _p,
            float(_row.get('sma_10', _row.get('SMA_10', _p))),
            float(_row.get('sma_30', _row.get('SMA_20', _p))),
            float(_row.get('sma_50', _row.get('SMA_50', _p))),
            float(_row.get('rsi', _row.get('RSI', 50))),
            float(_row.get('macd', _row.get('MACD', 0))),
            float(_row.get('macd_signal', _row.get('Signal_Line', 0))),
            float(_row.get('bb_upper', _p * 1.05)),
            float(_row.get('bb_lower', _p * 0.95)),
            float(df['Volume'].iloc[-1] if 'df' in dir() else 0),
            0.0, 0.0, 1.0,
        ], dtype=np.float32)
        action_raw, _ = model.predict(_ppo_obs, deterministic=True)
        av = float(action_raw[0]) if hasattr(action_raw, '__len__') else float(action_raw)
        _ppo_roi, _bh_roi = calculate_ppo_backtest_roi(model, df)
        _av = av
        action = 1 if av > 0.1 else (2 if av < -0.1 else 0)
        q_values = np.array([0.0, 0.0, 0.0]); q_values[action] = 1.0
    else:
        q_values = model.predict(state.reshape(1, -1))[0]
        action   = int(np.argmax(q_values))
    signals  = {0: ('持有 (HOLD)', 'Y'), 1: ('买入 (BUY)', 'G'), 2: ('卖出 (SELL)', 'R')}
    signal, _ = signals[action]
    confidence = float(q_values[action] - np.mean(q_values))

    sma20     = df['Close'].rolling(20).mean().iloc[-1]
    std20     = df['Close'].rolling(20).std().iloc[-1]
    bb_up     = sma20 + std20 * 2
    bb_lo     = sma20 - std20 * 2
    delta     = df['Close'].diff()
    rsi       = float((100 - (100 / (1 + delta.where(delta>0,0).rolling(14).mean() /
                (-delta.where(delta<0,0)).rolling(14).mean()))).iloc[-1])
    vol_ratio = float(df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1])

    print("\n" + "=" * 80)
    print(f"Q值: Hold={q_values[0]:.3f}  Buy={q_values[1]:.3f}  Sell={q_values[2]:.3f}")
    if _av is not None:
        print_ppo_action_line(_av, _ppo_roi, _bh_roi)
    print(f"信号: {signal}  (信心差值: {confidence:+.3f})")
    print(f"RSI: {rsi:.1f}  量比: {vol_ratio:.2f}x  BB上轨: NT${bb_up:.2f}  BB下轨: NT${bb_lo:.2f}")

    if action == 1:
        print(f"建议买入价格区间: NT${current_price*0.995:.2f} - NT${current_price:.2f}")
        print(f"设置止损: NT${current_price*0.95:.2f} (-5%)")
    elif action == 2:
        print(f"建议卖出价格区间: NT${current_price:.2f} - NT${current_price*1.005:.2f}")
    else:
        print(f"关注支撑: NT${bb_lo:.2f}  压力: NT${bb_up:.2f}")


    # ── FinBERT 情緒分析 ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("📰 市场情绪分析 (FinBERT NLP Engine)")
    print("=" * 80)
    from finbert_enhanced_scoring import calculate_sentiment_score, format_sentiment_output
    sentiment_result = calculate_sentiment_score('3504.TW', verbose=False)
    if sentiment_result and sentiment_result['news_count'] > 0:
        print(format_sentiment_output(sentiment_result))
    else:
        print("⚠️  未找到相关新闻，情绪分析不可用")
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}

    # ── Tavily 即時新聞 ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🌐 YOUNG OPTICS INC (3504.TW) 即時新聞  (Tavily REST API)")
    print("=" * 80)
    print_tavily_news('3504.TW', 'YOUNG OPTICS INC', max_results=5)
    print("\n" + "=" * 80)
    print("本信号由 AI 模型生成，仅供参考，不构成投资建议")
    print("=" * 80)
    return {'date': latest_date, 'symbol': TICKER, 'current_price': current_price,
            'signal': signal, 'action': action, 'rsi': rsi}


if __name__ == "__main__":
    result = get_trading_signal()
    if result:
        print(f"\n信号生成成功!")
        try:
            import yfinance as yf
            chart_df = yf.Ticker(TICKER).history(period="6mo")
            chart_df.columns = [c.lower() for c in chart_df.columns]
            plot_candlestick(chart_df, TICKER, save_path='3504_chart.png')
        except Exception as e:
            print(f"图表生成失败: {e}")
        print(f"\n股票: {result['symbol']} (YOUNG OPTICS INC)")
        print(f"日期: {result['date']}  价格: NT${result['current_price']:.2f}")
        print(f"信号: {result['signal']}")
        print(get_model_accuracy_display(TICKER))
    else:
        print("\n信号生成失败")
