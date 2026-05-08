"""
批量更新所有训练文件，添加特征重要性分析
"""
import os
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 训练文件列表
train_files = [
    'train_1519_taiwan_improved.py',
    'train_2317_taiwan_improved.py',
    'train_2330_taiwan_improved.py',
    'train_2337_taiwan_improved.py',
    'train_2344_taiwan_improved.py',
    'train_2408_taiwan_improved.py',
    'train_3711_taiwan_improved.py',
    'train_3715_taiwan_improved.py',
    'train_4991_taiwan_improved.py',
    'train_6175_taiwan_improved.py',
    'train_6269_taiwan_improved.py',
    'train_6515_taiwan_improved.py',
    'train_6770_taiwan_improved.py',
    'train_8131_taiwan_improved.py',
    'train_aapl_improved.py',
    'train_avgo_improved.py',
    'train_goog_improved.py',
    'train_mu_improved.py',
    # 'train_nvda_improved.py',  # Already updated
]

# 需要添加的导入
sklearn_imports = """from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score"""

# 需要添加的技术指标代码
additional_indicators = """
    # KD指标 (Stochastic Oscillator)
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['K'] = ((df['close'] - low_14) / (high_14 - low_14) * 100).fillna(50)
    df['D'] = df['K'].rolling(3).mean()

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_MA'] = df['OBV'].rolling(20).mean()

    # 移动平均线
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    df['MA_200'] = df['close'].rolling(200).mean()

    # 波动性指标
    df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # 价格变化率
    df['price_change_5d'] = df['close'].pct_change(5) * 100
    df['price_change_20d'] = df['close'].pct_change(20) * 100

    # MA50斜率
    df['MA50_slope'] = df['MA_50'].diff(5) / df['MA_50'].shift(5) * 100

    # 未来涨跌方向 (用于特征重要性分析)
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)"""

# 布林带位置修复
bb_position_fix = """    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range * 100).fillna(50)"""

# 特征重要性分析函数
feature_importance_function = """
# ==========================================
# 3. 特征重要性分析
# ==========================================
def analyze_feature_importance(df, ticker):
    \"\"\"分析技术指标的重要性\"\"\"
    print("\\n🧠 第 5 步：特徵重要性分析 (ML Model)")
    print("=" * 70 + "\\n")

    # 選擇特徵 (所有技術指標)
    features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_position', 'K', 'D', 'OBV', 'OBV_MA',
        'MA_20', 'MA_50', 'MA_200',
        'volatility', 'ATR', 'price_change_5d', 'price_change_20d', 'MA50_slope'
    ]

    # 排除 NaN 值
    ml_data = df.dropna(subset=features + ['future_direction'])

    if len(ml_data) == 0:
        print("❌ 數據不足，無法執行特徵重要性分析。請檢查NaN值是否過多。")
        return

    X = ml_data[features]
    y = ml_data['future_direction']

    print(f"用於分析的數據點總數: {len(X)}")

    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 劃分訓練集和測試集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    # 訓練隨機森林分類器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    # 獲取特徵重要性分數
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 預測準確度
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型在測試集上的預測準確率: {accuracy:.4f}\\n")
    print("✅ 特徵重要性計算完成")

    print("\\n" + "=" * 70)
    print("🏅 技術指標重要性排名")
    print("=" * 70)
    print(feature_importance_df.to_string(index=False))

    # 視覺化特徵重要性
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='#3498DB')
    plt.xlabel("特徵重要性分數 (Feature Importance Score)", fontweight='bold')
    plt.ylabel("技術指標 (Technical Indicator)", fontweight='bold')
    plt.title(f"{ticker} 基於隨機森林模型的技術指標重要性", fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    filename = f'{ticker}_feature_importance.png'
    plt.savefig(filename, dpi=300)
    print(f"✅ 特徵重要性圖表已儲存: {filename}")
    plt.close()

    return feature_importance_df
"""


def update_train_file(filename):
    """更新单个训练文件"""
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = False

        # 1. 添加sklearn导入（如果还没有）
        if 'from sklearn.ensemble import RandomForestClassifier' not in content:
            # 在现有导入后添加
            import_pattern = r'(from stable_baselines3\.common\.vec_env import DummyVecEnv\n)'
            if re.search(import_pattern, content):
                content = re.sub(
                    import_pattern,
                    r'\1' + sklearn_imports + '\n',
                    content
                )
                modified = True

        # 2. 添加MultiIndex处理（如果还没有）
        if 'isinstance(df.columns, pd.MultiIndex)' not in content:
            download_pattern = r'(df = yf\.download\([^)]+\))\s*\n\s*(if df\.empty:)'
            if re.search(download_pattern, content):
                multiindex_code = """

        # 处理MultiIndex列
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        """
                content = re.sub(
                    download_pattern,
                    r'\1' + multiindex_code + r'\2',
                    content
                )
                modified = True

        # 3. 修复布林带位置计算（如果存在旧版本）
        if "df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])" in content:
            old_bb_pattern = r"    # Bollinger Bands\s+df\['bb_middle'\][^\n]*\n[^\n]*\n[^\n]*\n[^\n]*\n[^\n]*"
            if re.search(old_bb_pattern, content, re.MULTILINE):
                content = re.sub(
                    old_bb_pattern,
                    bb_position_fix,
                    content,
                    flags=re.MULTILINE
                )
                modified = True

        # 4. 添加额外的技术指标（如果还没有）
        if "'future_direction'" not in content:
            # 在fillna之前添加
            fillna_pattern = r"(    # 填充缺失值\s+df = df\.fillna)"
            if re.search(fillna_pattern, content):
                content = re.sub(
                    fillna_pattern,
                    additional_indicators + '\n\n\\1',
                    content
                )
                modified = True

        # 5. 添加特征重要性分析函数（如果还没有）
        if 'def analyze_feature_importance' not in content:
            # 在训练函数之前添加
            train_pattern = r'(# ={40,}\n# \d+\. 训练模型\n# ={40,})'
            if re.search(train_pattern, content):
                content = re.sub(
                    train_pattern,
                    feature_importance_function + '\n\\1',
                    content
                )
                # 更新后续注释编号
                content = content.replace('# 3. 训练模型', '# 4. 训练模型')
                modified = True

        # 6. 添加特征重要性分析调用（如果还没有）
        if 'analyze_feature_importance(df' not in content:
            # 在数据分割后、训练前添加
            split_pattern = r'(print\(f"  测试集: \{len\(test_df\)\} 天"\)\s*\n)'
            if re.search(split_pattern, content):
                analysis_call = """
    # 3. 特征重要性分析
    feature_importance = analyze_feature_importance(df, TICKER)

"""
                content = re.sub(
                    split_pattern,
                    r'\1' + analysis_call,
                    content
                )
                modified = True

        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 更新成功: {filename}")
            return True
        else:
            print(f"⏭️  跳过 (已是最新): {filename}")
            return True

    except Exception as e:
        print(f"❌ 更新失败 {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("批量更新训练文件 - 添加特征重要性分析")
    print("=" * 70)
    print(f"\n将更新 {len(train_files)} 个文件...")
    print("\n新增功能:")
    print("  1. ✅ sklearn机器学习库导入")
    print("  2. ✅ 修复MultiIndex列处理")
    print("  3. ✅ 修复布林带位置计算")
    print("  4. ✅ 新增17个技术指标 (KD, OBV, ATR, 等)")
    print("  5. ✅ 随机森林特征重要性分析")
    print("  6. ✅ 生成可视化图表")
    print("\n" + "=" * 70)
    print("\n开始更新...\n")

    success_count = 0
    for filename in train_files:
        if update_train_file(filename):
            success_count += 1

    print("\n" + "=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(train_files)} 个文件")
    print("=" * 70)
