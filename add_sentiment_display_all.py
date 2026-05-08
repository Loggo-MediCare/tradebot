"""
为所有交易信号生成器添加显著的情绪分析显示
Add prominent sentiment display to all trading signal generators
"""
import os
import sys
import io
import re

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def add_sentiment_display(filepath):
    """在文件中添加显著的情绪分析显示"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 检查是否已经有情绪分析显示
    if '📰 市场情绪分析 (FinBERT NLP Engine)' in content:
        return False, "Already has prominent sentiment display"

    # 查找插入点（在 "# 7.5 占位变量" 或 "# 8. 生成交易建议" 之前）
    # 需要在动态权重计算器之后，交易信号生成之前

    pattern = r'(    # 7\.5 占位变量（将在买入信号部分填充）\n    sentiment_result = None\n)'

    if not re.search(pattern, content):
        # 尝试另一个模式
        pattern = r'(    # 8\. 生成交易建议\n)'

    if not re.search(pattern, content):
        return False, "Could not find insertion point"

    # 替换内容
    sentiment_code = r'''    # 7.5 获取市场情绪分析（FinBERT + VADER）
    print("\n" + "=" * 80)
    print("📰 市场情绪分析 (FinBERT NLP Engine)")
    print("=" * 80)

    from finbert_enhanced_scoring import calculate_sentiment_score, format_sentiment_output
    sentiment_result = calculate_sentiment_score('SYMBOL_PLACEHOLDER', verbose=True)

    if sentiment_result and sentiment_result['news_count'] > 0:
        print(format_sentiment_output(sentiment_result))
    else:
        print("⚠️  未找到相关新闻，情绪分析不可用")
        sentiment_result = {'sentiment_score': 0.0, 'news_count': 0, 'sentiment_label': '中性'}

'''

    # 从文件名提取股票代码
    basename = os.path.basename(filepath)
    if 'get_trading_signal_' in basename:
        symbol = basename.replace('get_trading_signal_', '').replace('.py', '')
        # 转换为大写并添加后缀
        if symbol.isdigit():
            symbol = f"{symbol}.TW"
        else:
            symbol = symbol.upper()

        sentiment_code = sentiment_code.replace('SYMBOL_PLACEHOLDER', symbol)

    # 执行替换
    if re.search(r'# 7\.5 占位变量', content):
        new_content = re.sub(
            r'    # 7\.5 占位变量（将在买入信号部分填充）\n    sentiment_result = None\n',
            sentiment_code,
            content
        )
    else:
        new_content = re.sub(
            r'(    # 8\. 生成交易建议\n)',
            sentiment_code + r'\1',
            content
        )

    if new_content == content:
        return False, "Replacement failed"

    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return True, "Success"

def main():
    # 获取所有交易信号文件
    signal_files = [f for f in os.listdir('.')
                   if f.startswith('get_trading_signal_') and f.endswith('.py')]

    print(f"找到 {len(signal_files)} 个交易信号文件")
    print("="*70)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for filename in sorted(signal_files):
        filepath = os.path.join('.', filename)
        success, msg = add_sentiment_display(filepath)

        if success:
            print(f"✅ {filename}: {msg}")
            success_count += 1
        elif "Already has" in msg:
            print(f"⏭️  {filename}: {msg}")
            skip_count += 1
        else:
            print(f"❌ {filename}: {msg}")
            fail_count += 1

    print("="*70)
    print(f"完成统计:")
    print(f"  成功添加: {success_count}")
    print(f"  已存在: {skip_count}")
    print(f"  失败: {fail_count}")
    print(f"  总计: {len(signal_files)}")

if __name__ == "__main__":
    main()
