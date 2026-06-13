# FinBERT 情绪分析整合指南

## 📚 概述

FinBERT 是一个专门针对金融文本进行情绪分析的 BERT 模型，能够分析新闻标题和摘要，判断市场情绪（正面/中性/负面）。

本系统已将 FinBERT 整合到交易信号生成系统中，可以为买入/卖出决策提供额外的市场情绪支持。

---

## 🗂️ 相关文件

### 核心模块

1. **`finbert_sentiment.py`** - FinBERT 情绪分析核心模块
   - `FinBERTAnalyzer` 类：初始化和使用 FinBERT 模型
   - `fetch_news_via_rss()`: 从 Yahoo Finance RSS 抓取新闻
   - `get_sentiment_score()`: 计算单条新闻的情绪分数
   - `analyze_stock_sentiment()`: 分析股票的整体市场情绪

2. **`finbert_enhanced_scoring.py`** - 增强评分模块
   - `calculate_sentiment_score()`: 计算情绪分数并转换为评分调整值
   - `calculate_enhanced_buy_score_with_sentiment()`: 整合 FinBERT 的买入评分系统
   - `format_sentiment_output()`: 格式化情绪分析结果输出

### 示例文件

3. **`get_trading_signal_nvda_finbert.py`** - 整合 FinBERT 的 NVDA 交易信号示例

---

## 🚀 安装依赖

首次使用需要安装以下依赖包：

```bash
pip install transformers torch nltk
```

**注意：** 首次运行时，FinBERT 模型（约 400MB）会自动下载，请耐心等待。

---

## 💡 使用方法

### 方法 1: 独立使用 FinBERT 分析

```python
from finbert_sentiment import FinBERTAnalyzer

# 初始化分析器
analyzer = FinBERTAnalyzer()

# 分析单只股票
result = analyzer.analyze_stock_sentiment('NVDA', verbose=True)

# 输出结果
print(f"情绪分数: {result['sentiment_score']}")
print(f"情绪判断: {result['sentiment_label']}")
print(f"新闻数量: {result['news_count']}")
```

### 方法 2: 在交易信号中使用

修改你的 `get_trading_signal_xxx.py` 文件：

#### Step 1: 导入模块

```python
# 在文件开头添加
from finbert_enhanced_scoring import calculate_enhanced_buy_score_with_sentiment, format_sentiment_output
```

#### Step 2: 修改评分调用

将原有的：
```python
buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata = calculate_enhanced_buy_score(
    rsi=rsi,
    macd=macd,
    # ... 其他参数
)
```

改为：
```python
buy_score, signal_override, buy_reasons, buy_warnings, buy_metadata, sentiment_result = calculate_enhanced_buy_score_with_sentiment(
    rsi=rsi,
    macd=macd,
    # ... 其他参数
    symbol='NVDA'  # 添加股票代码参数
)
```

#### Step 3: 显示情绪分析结果

在买入建议输出后添加：
```python
# 显示FinBERT情绪分析结果
if sentiment_result and sentiment_result['news_count'] > 0:
    print("\n" + format_sentiment_output(sentiment_result))
```

---

## 📊 情绪分数映射

FinBERT 返回的情绪分数会自动转换为评分调整值：

| 情绪分数 | 情绪标签 | 评分调整 |
|---------|---------|---------|
| > +0.30  | 强烈正面 | **+20 分** |
| +0.15 ~ +0.30 | 正面 | **+10 分** |
| +0.05 ~ +0.15 | 轻微正面 | **+5 分** |
| -0.05 ~ +0.05 | 中性 | **0 分** |
| -0.15 ~ -0.05 | 轻微负面 | **-5 分** |
| -0.30 ~ -0.15 | 负面 | **-10 分** |
| < -0.30  | 强烈负面 | **-20 分** |

**示例：**
- 原始技术评分：45/100
- FinBERT 情绪分数：+0.25（正面）
- 评分调整：+10 分
- **最终评分：55/100** ✅

---

## 🔍 新闻来源映射

系统会根据股票代码自动选择合适的搜索关键字：

### 美股
- `NVDA` → 搜索 "NVDA"
- `AAPL` → 搜索 "AAPL"
- `GOOGL` → 搜索 "GOOGL"

### 欧股
- `RHM.DE` → 搜索 "RHM.DE" + 过滤 "Rheinmetall"

### 台股 (使用 ADR 或英文名)
- `2330.TW` (台积电) → 搜索 "TSM" (ADR)
- `3711.TW` (日月光) → 搜索 "ASX" (ADR)
- `2308.TW` (台达电) → 搜索 "2308.TW" + 过滤 "Delta"
- `6442.TW` (兆丰金) → 搜索 "6442.TW" + 过滤 "Mega Financial"

**修改映射表：**

编辑 `finbert_sentiment.py` 中的 `SEARCH_MAPPING` 字典：

```python
SEARCH_MAPPING = {
    '你的股票代码': ('搜索关键字', '可选过滤关键字'),
    # 示例
    '2451.TW': ('2451.TW', 'Transcend'),
}
```

---

## 🎯 输出示例

```
================================================================================
🗞️  市场情绪分析 (FinBERT)
================================================================================
新闻数量:     8 则
情绪分数:     +0.245
情绪判断:     📈 正面
评分调整:     +10 分

📰 热点新闻:
   1. Nvidia Stock Surges on AI Chip Demand Forecast
   2. NVDA Beats Q4 Earnings Expectations, Raises Guidance
   3. Analysts Upgrade Nvidia Price Target to $180
```

---

## ⚙️ 配置选项

### 调整抓取的新闻数量

```python
# 在 finbert_sentiment.py 中
news_items = self.fetch_news_via_rss(
    search_term,
    filter_kw,
    max_news=10  # 改为 5 或 15
)
```

### 禁用 FinBERT（回退到技术指标）

如果 FinBERT 模型加载失败或不想使用，系统会自动回退：

```python
# 自动返回
{
    'sentiment_score': 0.0,
    'sentiment_label': '未启用',
    'news_count': 0,
    'score_adjustment': 0
}
```

---

## 📈 实战案例

### 案例 1: OMER (Omeros Corporation)

**技术面：**
- RSI: 71.1 (超买)
- MACD: 金叉
- 量比: 1.7x (放量)
- 技术评分: **52/100**

**FinBERT 情绪分析：**
- 新闻数量: 6 则
- 情绪分数: +0.32 (强烈正面)
- 评分调整: **+20 分**

**最终评分: 72/100** → **强力买入** 🟢

---

### 案例 2: RHM.DE (莱茵金属)

**技术面：**
- RSI: 53.89 (中性)
- MACD: 金叉（负值区）
- 量比: 0.54x (缩量)
- 技术评分: **9/100**

**FinBERT 情绪分析：**
- 新闻数量: 10 则
- 情绪分数: +0.28 (正面，地缘政治利好)
- 评分调整: **+10 分**

**最终评分: 19/100** → **仍为观望** 🟡

**分析：** 虽然情绪正面，但技术面过弱，评分调整后仍低于买入阈值（20分），系统正确拒绝买入。

---

## ⚠️ 注意事项

1. **网络要求：** 需要访问 Yahoo Finance RSS (可能需要代理)
2. **首次运行：** 模型下载需要 5-10 分钟
3. **台股限制：** 台股新闻较少，建议使用 ADR 代码搜索
4. **非交易时段：** 可能抓不到最新新闻
5. **不可过度依赖：** 情绪分析作为辅助参考，不应单独决策

---

## 🔧 故障排除

### 问题 1: ModuleNotFoundError: transformers

**解决：**
```bash
pip install transformers torch
```

### 问题 2: 新闻抓取失败

**原因：** 网络连接问题或 Yahoo Finance 限制

**解决：**
- 检查网络连接
- 使用代理
- 修改搜索关键字映射

### 问题 3: 模型加载失败

**解决：**
```bash
# 手动下载模型
python -c "from transformers import BertTokenizer, BertForSequenceClassification; BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')"
```

---

## 📚 参考资料

- **FinBERT 模型：** https://huggingface.co/yiyanghkust/finbert-tone
- **Yahoo Finance RSS：** https://feeds.finance.yahoo.com/
- **Transformers 文档：** https://huggingface.co/docs/transformers/

---

## 📝 更新日志

**v1.0 (2025-12-31)**
- ✅ 初始版本发布
- ✅ 支持美股/欧股/台股新闻抓取
- ✅ 整合到交易信号评分系统
- ✅ 自动评分调整 (-20 ~ +20 分)

---

**有问题？** 查看 `finbert_sentiment.py` 和 `finbert_enhanced_scoring.py` 的代码注释 📖
