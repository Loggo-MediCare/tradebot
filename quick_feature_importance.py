"""
快速生成特征重要性 JSON (不需要完整训练)
=============================================
这个脚本跳过完整的 PPO 训练，只进行:
1. 下载历史数据
2. 计算技术指标
3. 训练随机森林分类器
4. 生成特征重要性 JSON

优点: 每个股票只需 10-30 秒 (vs 完整训练的 30-120 分钟)
缺点: 不生成 PPO 模型文件 (只生成 JSON)
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 股票列表
STOCKS = {
    # 台股
    '1519.TW': '华城',
    '2317.TW': '鸿海',
    '2330.TW': '台积电',
    '2337.TW': '旺宏',
    '2344.TW': '华邦电',
    '2408.TW': '南亚科',
    '3017.TW': '奇鋐',
    '3711.TW': '日月光投控',
    '3715.TW': '定颖投控',
    '4938.TW': '和碩',
    '4991.TW': '环宇-KY',
    '6175.TW': '立敦',
    '6209.TW': '今國光',
    '6269.TW': '台郡',
    '6443.TW': '元晶',
    '6515.TW': '穎崴',
    '6770.TW': '力积电',
    '6805.TW': '富世達',
    '8131.TW': '福懋科',
    '8210.TW': '勤誠',
    # 美股
    'AAPL': 'Apple',
    'AVGO': 'Broadcom',
    'GOOG': 'Google',
    'MU': 'Micron',
    'NVDA': 'NVIDIA',
}

def calculate_technical_indicators(df):
    """计算所有技术指标"""
    # 如果是多级列索引，展平为单级
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['MA_20'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['MA_20'] - (df['BB_std'] * 2)
    df['bb_position'] = ((df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])) * 100

    # KD 指标
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    rsv = ((df['Close'] - low_9) / (high_9 - low_9)) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()

    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(window=20).mean()

    # 移动平均线
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # 波动率
    df['volatility'] = df['Close'].pct_change().rolling(window=20).std()

    # ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    # 价格变化
    df['price_change_5d'] = df['Close'].pct_change(periods=5)
    df['price_change_20d'] = df['Close'].pct_change(periods=20)

    # MA50 斜率
    df['MA50_slope'] = df['MA_50'].diff(periods=5)

    # 未来价格方向 (目标变量)
    df['future_close'] = df['Close'].shift(-5)
    df['future_direction'] = (df['future_close'] > df['Close']).astype(int)

    return df

def analyze_feature_importance(ticker, name):
    """分析单个股票的特征重要性"""
    print(f"\n{'='*70}")
    print(f"📊 {ticker} ({name})")
    print(f"{'='*70}")

    try:
        # 下载数据
        print(f"▶️  下载历史数据...")
        df = yf.download(ticker, start='2015-01-01', end='2024-12-31', progress=False)

        if df.empty or len(df) < 300:
            print(f"❌ 数据不足: {len(df)} 条记录")
            return False

        print(f"✅ 数据下载完成: {len(df)} 条记录")

        # 计算技术指标
        print(f"▶️  计算技术指标...")
        df = calculate_technical_indicators(df)

        # 特征列表
        features = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
            'MA_20', 'MA_50', 'MA_200',
            'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
        ]

        # 准备数据
        ml_data = df.dropna(subset=features + ['future_direction'])

        if len(ml_data) < 100:
            print(f"❌ 有效数据不足: {len(ml_data)} 条记录")
            return False

        X = ml_data[features]
        y = ml_data['future_direction']

        print(f"✅ 有效数据: {len(X)} 条记录")

        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )

        # 训练随机森林分类器
        print(f"▶️  训练随机森林模型...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1  # 使用所有CPU核心
        )
        rf_model.fit(X_train, y_train)

        # 获取特征重要性
        importances = rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # 预测准确度
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"✅ 模型准确率: {accuracy:.4f}")

        # 显示特征重要性
        print(f"\n🏅 特征重要性排名:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"   {row['Feature']:20s}: {row['Importance']:.4f}")

        # 保存 JSON
        json_data = {
            'ticker': ticker,
            'name': name,
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'model_accuracy': float(accuracy),
            'data_points': int(len(X)),
            'feature_importance': {
                row['Feature']: float(row['Importance'])
                for _, row in feature_importance_df.iterrows()
            }
        }

        json_filename = f'{ticker}_feature_importance.json'
        json_path = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"✅ JSON 已保存: {json_filename}")
        return True

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_existing_json():
    """检查已存在的 JSON 文件"""
    project_dir = r'C:\Users\Silvi\Projects\trading-bot'
    existing = []
    missing = []

    for ticker in STOCKS.keys():
        json_file = f'{ticker}_feature_importance.json'
        json_path = os.path.join(project_dir, json_file)

        if os.path.exists(json_path):
            existing.append(ticker)
        else:
            missing.append(ticker)

    return existing, missing

if __name__ == "__main__":
    print("=" * 70)
    print("快速生成特征重要性 JSON 文件")
    print("=" * 70)
    print(f"总共 {len(STOCKS)} 个股票")
    print(f"预计时间: {len(STOCKS) * 15} 秒 (约 {len(STOCKS) * 15 / 60:.1f} 分钟)\n")

    # 检查现有 JSON
    existing, missing = check_existing_json()

    print(f"📊 当前状态:")
    print(f"  ✅ 已存在: {len(existing)} 个")
    if existing:
        for ticker in existing[:5]:  # 只显示前5个
            print(f"     • {ticker}")
        if len(existing) > 5:
            print(f"     ... 还有 {len(existing) - 5} 个")

    print(f"\n  ⏳ 需要生成: {len(missing)} 个")
    if missing:
        for ticker in missing[:5]:
            print(f"     • {ticker} ({STOCKS[ticker]})")
        if len(missing) > 5:
            print(f"     ... 还有 {len(missing) - 5} 个")

    if not missing:
        print("\n🎉 所有 JSON 文件已存在!")
        print("=" * 70)
        sys.exit(0)

    # 开始生成
    print("\n" + "=" * 70)
    print("🏁 开始生成")
    print("=" * 70)

    success_count = 0
    failed_tickers = []

    for ticker in missing:
        name = STOCKS[ticker]
        try:
            if analyze_feature_importance(ticker, name):
                success_count += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断")
            print(f"已完成: {success_count}/{len(missing)} 个")
            break
        except Exception as e:
            print(f"\n❌ 未预期的错误: {e}")
            failed_tickers.append(ticker)

    # 最终报告
    print("\n" + "=" * 70)
    print("📊 生成完成报告")
    print("=" * 70)
    print(f"成功: {success_count}/{len(missing)} 个")

    if failed_tickers:
        print(f"\n失败的股票:")
        for ticker in failed_tickers:
            print(f"  ❌ {ticker} ({STOCKS[ticker]})")

    # 最终状态
    existing_final, missing_final = check_existing_json()
    print(f"\n当前 JSON 文件状态:")
    print(f"  ✅ 已存在: {len(existing_final)}/{len(STOCKS)} 个")
    if missing_final:
        print(f"  ⏳ 仍需生成: {len(missing_final)} 个")

    print("=" * 70)
