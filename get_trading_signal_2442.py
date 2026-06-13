"""


台股 2442 (新美齊) AI 交易信号生成器


使用 DQN Keras 模型


"""


import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


os.environ['MPLBACKEND'] = 'Agg'


import sys


import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import numpy as np


import pandas as pd


import tensorflow as tf


from sklearn.preprocessing import MinMaxScaler


from datetime import datetime


import warnings


warnings.filterwarnings('ignore')


from model_accuracy_tracker import get_model_accuracy_display
from tw_news_tracker import print_tavily_news_tw


TICKER = '2442.TW'


TICKER_NAME = '新美齊'


MODEL_PATH = r'C:\Users\Silvi\Projects\trading-bot\2442.TW_improved_anti_overfit_model.keras'


WINDOW_SIZE = 5


FEATURE_COLS = ['SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'ROC_5', 'ROC_10',


                'RSI', 'Volatility_10', 'Volatility_20', 'MACD', 'Signal_Line',


                'BB_position', 'Volume_ratio']


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


    sma20  = f['Close'].rolling(20).mean()


    std20  = f['Close'].rolling(20).std()


    bb_up  = sma20 + std20 * 2


    bb_lo  = sma20 - std20 * 2


    f['BB_position'] = (f['Close'] - bb_lo) / (bb_up - bb_lo)


    f['Volume_SMA']  = f['Volume'].rolling(20).mean()


    f['Volume_ratio'] = f['Volume'] / f['Volume_SMA']


    f = f.dropna()


    scaler = MinMaxScaler()


    f[FEATURE_COLS] = scaler.fit_transform(f[FEATURE_COLS])


    return f


def get_state(df_feat, step, balance=10000, shares=0, initial_balance=10000):


    start = step - WINDOW_SIZE


    market_state = df_feat[FEATURE_COLS].iloc[start:step].values.flatten()


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


def get_trading_signal():


    print("=" * 80)


    print(f"🤖 台股 {TICKER} ({TICKER_NAME}) AI 交易信号生成器")


    accuracy_display = get_model_accuracy_display(TICKER)


    print(f"模型準確度: {accuracy_display}")


    print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


    print("=" * 80)


    # 1. 載入模型


    print(f"\n📦 加载 AI 模型: {MODEL_PATH}")


    try:


        model = tf.keras.models.load_model(MODEL_PATH)


        print("✅ 模型加载成功!")


    except Exception as e:


        print(f"❌ 模型加载失败: {e}")


        return None


    # 2. 下载数据


    print("\n📊 下载最新市场数据...")


    try:


        import yfinance as yf


        df = yf.download(TICKER, period='200d', progress=False, auto_adjust=True)


        if df.empty:


            print("❌ 无法获取数据")


            return None


        if isinstance(df.columns, pd.MultiIndex):


            df.columns = df.columns.droplevel(1)


        df = df.reset_index()


        print(f"✅ 成功下载 {len(df)} 天数据")


    except Exception as e:


        print(f"❌ 数据下载失败: {e}")


        return None


    # 3. 建立特征


    print("\n🔧 计算技术指标...")


    df_feat = create_features(df)


    if len(df_feat) < WINDOW_SIZE + 1:


        print("❌ 数据不足，无法生成信号")


        return None


    latest = df_feat.iloc[-1]


    try:


        latest_date = str(pd.to_datetime(df_feat.iloc[-1]['Date']).date())


    except Exception:


        latest_date = datetime.now().strftime('%Y-%m-%d')


    current_price = float(latest['Close'])


    rsi_raw       = float(df_feat['RSI'].iloc[-1])


    macd_raw      = float(df_feat['MACD'].iloc[-1])


    volume_ratio  = float(df_feat['Volume_ratio'].iloc[-1])


    print(f"✅ 最新数据日期: {latest_date}")


    print(f"   当前价格: NT${current_price:.2f}")


    # 4. 取得觀測值並預測


    step  = len(df_feat) - 1


    state = get_state(df_feat, step)


    q_values = model.predict(state.reshape(1, -1), verbose=0)[0]


    action   = int(np.argmax(q_values))


    action_names = {0: '持有 (HOLD)', 1: '买入 (BUY)', 2: '卖出 (SELL)'}


    action_emojis = {0: '🟡', 1: '🟢', 2: '🔴'}


    signal      = action_names[action]


    signal_emoji = action_emojis[action]


    confidence  = float(q_values[action] - np.mean(q_values))


    # 5. 技术指標输出


    print("\n" + "=" * 80)


    print("📊 技术指标分析")


    print("=" * 80)


    # Use unscaled values from original df for display


    orig = df.iloc[-1]


    sma5  = df['Close'].rolling(5).mean().iloc[-1]


    sma20 = df['Close'].rolling(20).mean().iloc[-1]


    std20 = df['Close'].rolling(20).std().iloc[-1]


    bb_up = sma20 + std20 * 2


    bb_lo = sma20 - std20 * 2


    bb_pos = (current_price - bb_lo) / (bb_up - bb_lo) * 100 if (bb_up - bb_lo) > 0 else 50


    delta = df['Close'].diff()


    gain  = delta.where(delta > 0, 0).rolling(14).mean()


    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()


    rsi   = float((100 - (100 / (1 + gain / loss))).iloc[-1])


    ema12 = df['Close'].ewm(span=12, adjust=False).mean()


    ema26 = df['Close'].ewm(span=26, adjust=False).mean()


    macd  = float((ema12 - ema26).iloc[-1])


    sig   = float((ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1])


    vol_sma = df['Volume'].rolling(20).mean().iloc[-1]


    vol_ratio = float(df['Volume'].iloc[-1] / vol_sma) if vol_sma > 0 else 1.0


    print(f"RSI (14):        {rsi:.2f}  {'[超买]' if rsi > 70 else '[超卖]' if rsi < 30 else '[中性]'}")


    print(f"MACD:            {macd:.4f}  {'[金叉]' if macd > sig else '[死叉]'}")


    print(f"SMA 5:           NT${float(sma5):.2f}")


    print(f"SMA 20:          NT${float(sma20):.2f}  {'[多头]' if sma5 > sma20 else '[空头]'}")


    print(f"布林带上轨:      NT${bb_up:.2f}")


    print(f"布林带下轨:      NT${bb_lo:.2f}")


    print(f"当前价格位置:    {bb_pos:.1f}% {'⚡ 上軌外延' if current_price > bb_up else ('⚠️ 下軌外延' if current_price < bb_lo else '布林带内')}")


    print(f"量比:            {vol_ratio:.2f}x  {'[放量]' if vol_ratio > 1.5 else '[缩量]' if vol_ratio < 0.7 else '[正常]'}")


    # 6. AI 信号输出


    print("\n" + "=" * 80)


    print("🎯 AI 交易信号")


    print("=" * 80)


    print(f"Q 值: Hold={q_values[0]:.4f}  Buy={q_values[1]:.4f}  Sell={q_values[2]:.4f}")


    print(f"\n{signal_emoji} 信号: {signal}")


    print(f"   信心度差值: {confidence:+.4f}")


    if action == 1:


        print(f"   建议买入价格区间: NT${current_price * 0.995:.2f} - NT${current_price:.2f}")


        print(f"\n   💡 操作建议:")


        if rsi < 30:


            print(f"      • RSI超卖({rsi:.1f}), 买入信号强")


        elif rsi > 70:


            print(f"      • RSI偏高({rsi:.1f}), 谨慎追高，建议分批买入")


        else:


            print(f"      • 分批买入，设置止损 NT${current_price * 0.95:.2f} (-5%)")


    elif action == 2:


        print(f"   建议卖出价格区间: NT${current_price:.2f} - NT${current_price * 1.005:.2f}")


        print(f"\n   💡 操作建议:")


        if rsi > 70:


            print(f"      • RSI超买({rsi:.1f}), 卖出信号强")


        else:


            print(f"      • 适度减仓，保留部分仓位观察后续")


    else:


        print(f"\n   💡 操作建议:")


        print(f"      • 继续观察市场走势")


        print(f"      • 关注支撑位: NT${bb_lo:.2f}")


        print(f"      • 关注压力位: NT${bb_up:.2f}")


    print("\n" + "=" * 80)


    print("⚠️  风险提示")


    print("=" * 80)


    print("   • 本信号由 AI 模型生成,仅供参考,不构成投资建议")


    print("   • 股市有风险,投资需谨慎")


    print("=" * 80)


    return {


        'date': latest_date,


        'symbol': TICKER,


        'current_price': current_price,


        'signal': signal,


        'action': action,


        'rsi': rsi,


        'macd': macd,


    }


if __name__ == "__main__":


    result = get_trading_signal()


    if result:


        print(f"\n✅ 信号生成成功!")


        try:


            import yfinance as yf


            chart_df = yf.Ticker(TICKER).history(period="6mo")


            chart_df.columns = [c.lower() for c in chart_df.columns]


            plot_candlestick(chart_df, TICKER, save_path=f"2442_chart.png")


        except Exception as e:


            print(f"   圖表生成失敗: {e}")


        print(f"\n📱 快速摘要:")


        print(f"   股票: {result['symbol']} ({TICKER_NAME})")


        print(f"   日期: {result['date']}")


        print(f"   价格: NT${result['current_price']:.2f}")


        print(f"   信号: {result['signal']}")


        print(f"   {get_model_accuracy_display(TICKER)}")


    else:


        print("\n❌ 信号生成失败")



# ── Tavily 即時新聞 ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '=' * 80)
    print('🌐 2442 新美齊 即時新聞  (Tavily REST API)')
    print('=' * 80)
    print_tavily_news_tw('2442', '新美齊', max_results=5)