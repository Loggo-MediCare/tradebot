"""
批量添加特征重要性分析函数到缺失的训练文件
针对: 3711, 3715, 6175, 6515
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import re

# 需要添加功能的文件
missing_files = [
    'train_3711_taiwan_improved.py',
    'train_3715_taiwan_improved.py',
    'train_6175_taiwan_improved.py',
    'train_6515_taiwan_improved.py',
]

# 特征重要性分析函数（完整版本）
FEATURE_IMPORTANCE_FUNCTION = '''
# ==========================================
# 5. 特征重要性分析
# ==========================================
def analyze_feature_importance(df, ticker):
    """分析技术指标的重要性"""
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
    json_filename = f'{ticker}_feature_importance.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 特徵重要性數據已保存: {json_filename}")

    return feature_importance_df
'''

def add_feature_importance_function(filename):
    """为训练文件添加特征重要性分析函数"""
    filepath = os.path.join(r'C:\Users\Silvi\Projects\trading-bot', filename)

    if not os.path.exists(filepath):
        print(f"❌ 文件不存在: {filename}")
        return False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已经有这个函数
        if 'def analyze_feature_importance' in content:
            print(f"⏭️  已存在特征重要性函数: {filename}")
            return True

        # 提取股票代码（从文件名）
        ticker_match = re.search(r'train_(\d+)_taiwan', filename)
        if ticker_match:
            ticker_num = ticker_match.group(1)
            ticker = f"{ticker_num}.TW"
        else:
            print(f"⚠️  无法从文件名提取股票代码: {filename}")
            return False

        # 查找插入位置：在 "4. 训练模型" 注释之前
        pattern = r'(# ={40,}\n# 4\. 训练模型\n# ={40,})'

        if not re.search(pattern, content):
            print(f"⚠️  未找到插入点: {filename}")
            return False

        # 插入特征重要性函数
        content = re.sub(pattern, FEATURE_IMPORTANCE_FUNCTION + '\n\\1', content)

        # 在主程序中添加调用（在回测之后）
        # 查找 "开始回测" 部分后的位置
        backtest_pattern = r'(print\("=" \* 70\)\n    print\("✅ 回测完成!"\)\n    print\("=" \* 70\))'

        if re.search(backtest_pattern, content):
            # 在回测完成后添加特征重要性分析调用
            call_code = f'''

    # ==========================================
    # 特征重要性分析
    # ==========================================
    analyze_feature_importance(df, '{ticker}')
'''
            content = re.sub(backtest_pattern, '\\1' + call_code, content)
        else:
            # 如果找不到回测完成标记，尝试在文件末尾添加
            print(f"⚠️  未找到回测完成标记，尝试添加到末尾: {filename}")
            content += f'\n\n    # 特征重要性分析\n    analyze_feature_importance(df, "{ticker}")\n'

        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ 成功添加特征重要性函数: {filename}")
        return True

    except Exception as e:
        print(f"❌ 处理失败 {filename}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("批量添加特征重要性分析函数")
    print("=" * 70)
    print(f"\n目标文件: {len(missing_files)} 个\n")

    success_count = 0
    for filename in missing_files:
        if add_feature_importance_function(filename):
            success_count += 1
        print()

    print("=" * 70)
    print(f"完成! 成功更新 {success_count}/{len(missing_files)} 个文件")
    print("=" * 70)
