"""
Script to add analyze_feature_importance function to training files that are missing it
"""
import os
import re

# The function code to insert
ANALYZE_FUNCTION = '''def analyze_feature_importance(df, ticker):
    """分析技术指标的重要性"""
    print("\\n特徵重要性分析 (ML Model)")
    print("=" * 70)
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
        print("數據不足，無法執行特徵重要性分析。請檢查NaN值是否過多。")
        return None
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
    print(f"模型在測試集上的預測準確率: {accuracy:.4f}")
    print("特徵重要性計算完成")
    print("=" * 70)
    print("技術指標重要性排名")
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
    filename = f'{ticker.replace(".", "_")}_feature_importance.png'
    plt.savefig(filename, dpi=300)
    print(f"特徵重要性圖表已儲存: {filename}")
    plt.close()

    # 保存为 JSON 文件供交易信号使用
    import json
    from datetime import datetime
    json_data = {
        'ticker': ticker,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'model_accuracy': float(accuracy),
        'feature_importance': {
            row['Feature']: float(row['Importance'])
            for _, row in feature_importance_df.iterrows()
        }
    }
    json_filename = f'{ticker.replace(".", "_")}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"特徵重要性數據已保存: {json_filename}")

    return feature_importance_df
'''

# Additional technical indicators to add
ADDITIONAL_INDICATORS = '''    # KD指标 (Stochastic Oscillator)
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
    df['future_direction'] = (df['close'].shift(-5) > df['close']).astype(int)
'''

FILES_TO_FIX = [
    'train_3711_taiwan_improved.py',
    'train_3715_taiwan_improved.py',
    'train_6175_taiwan_improved.py',
]

def fix_file(filepath):
    """Fix a single file by adding missing function and indicators"""
    print(f"\n处理文件: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if function already exists
    if 'def analyze_feature_importance' in content:
        print(f"  ✓ 文件已有 analyze_feature_importance 函数")
        return False

    # Check if it needs additional indicators
    needs_indicators = 'OBV' not in content or 'MA_200' not in content

    # Add matplotlib backend if missing
    if "matplotlib.use('Agg')" not in content:
        content = content.replace(
            "import matplotlib\nmatplotlib.rcParams",
            "import matplotlib\nmatplotlib.use('Agg')  # 使用非交互式后端，避免 Tcl/Tk 错误\nmatplotlib.rcParams"
        )
        print(f"  ✓ 添加了 matplotlib backend 配置")

    # Add additional indicators if needed
    if needs_indicators:
        # Find where to insert (before the fillna line in add_technical_indicators)
        pattern = r"(    df\['bb_position'\] = .*\n)(    df = df\.fillna)"
        if re.search(pattern, content):
            content = re.sub(pattern, r'\1' + ADDITIONAL_INDICATORS + r'\2', content)
            print(f"  ✓ 添加了额外的技术指标")

    # Add analyze_feature_importance function before train_improved_model
    pattern = r'(def train_improved_model\(df, ticker)'
    if re.search(pattern, content):
        content = re.sub(pattern, ANALYZE_FUNCTION + '\n' + r'\1', content)
        print(f"  ✓ 添加了 analyze_feature_importance 函数")

    # Update main section to use the function properly
    # Replace the call to use train_df instead of df
    content = re.sub(
        r'feature_importance = analyze_feature_importance\(df, TICKER\)',
        'print("\\n执行特征重要性分析...")\\n    feature_importance = analyze_feature_importance(train_df, TICKER)',
        content
    )

    # Add feature importance to summary if missing
    if '包含特征重要性分析' not in content:
        content = content.replace(
            '    print("  ✅ 训练 100,000 步 (更充分)")',
            '    print("  ✅ 训练 100,000 步 (更充分)")\\n    print("  ✅ 包含特征重要性分析")'
        )

    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✅ 文件修复完成!")
    return True

if __name__ == '__main__':
    print("开始修复缺失 analyze_feature_importance 函数的文件...")
    print("=" * 70)

    fixed_count = 0
    for filepath in FILES_TO_FIX:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed_count += 1
        else:
            print(f"\n⚠️  文件不存在: {filepath}")

    print("\n" + "=" * 70)
    print(f"完成! 共修复了 {fixed_count} 个文件")
