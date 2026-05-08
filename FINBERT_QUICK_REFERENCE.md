# FinBERT 快速参考卡 Quick Reference

## 🚀 5分钟快速开始

### 1. 查看股票新闻和情绪

```bash
# 单个股票
python check_stock_news.py OMER

# 批量对比
python check_stock_news.py OMER NVDA 6443.TW 8110.TW
```

### 2. 运行交易信号（含FinBERT）

```bash
# 单个信号
python get_trading_signal_omer.py
python get_trading_signal_nvda.py

# 批量运行
python run_all_western.py     # 10只西方股票
python run_all_local_tw.py    # 26只台股
```

### 3. 添加妖股题材标注

编辑 `manual_sentiment_override.json`:

```json
{
  "6443.TW": {
    "theme": "SpaceX IPO 2026",
    "sentiment_score": 0.45,
    "score_adjustment": 20,
    "notes": "元晶为SpaceX唯一台厂供应商",
    "valid_until": "2026-07-01",
    "risk_warning": "妖股操作，沿5日线止损",
    "technical_override": {
      "ignore_rsi_overbought": true
    }
  }
}
```

---

## 📊 评分调整速查表

| 情绪分数 | 标签 | 调整 | 示例 |
|---------|------|------|------|
| > +0.30 | 🚀 强烈正面 | **+20** | FDA批准、大订单 |
| +0.15~+0.30 | 📈 正面 | **+10** | 业绩增长 |
| +0.05~+0.15 | 轻微正面 | **+5** | 常规利好 |
| -0.05~+0.05 | ➖ 中性 | **0** | 无明显情绪 |
| -0.15~-0.05 | 轻微负面 | **-5** | 小幅利空 |
| -0.30~-0.15 | 📉 负面 | **-10** | 业绩下滑 |
| < -0.30 | 📉 强烈负面 | **-20** | 重大丑闻 |

---

## 🎯 适用性速查

| 股票类型 | FinBERT | 人工标注 | 推荐 |
|---------|---------|---------|------|
| 美股 | ✅✅✅✅✅ | ⭐⭐⭐ | **FinBERT** |
| 欧股 | ✅✅✅✅ | ⭐⭐⭐ | **FinBERT** |
| 台股大型股 | ✅✅✅✅ | ⭐⭐⭐ | **FinBERT** |
| 台股低价股 | ❌ | ✅✅✅✅✅ | **人工标注** |
| 题材妖股 | ❌ | ✅✅✅✅✅ | **人工标注** |

---

## 🔧 故障排除

### FinBERT 加载失败
```bash
pip install transformers torch --upgrade
```

### 台股无新闻
→ 添加到 `manual_sentiment_override.json`

### 恢复备份
```bash
copy signal_backups\get_trading_signal_xxx.py.20251231_055225.bak get_trading_signal_xxx.py
```

---

## 📁 核心文件速查

| 文件 | 功能 |
|------|------|
| `finbert_sentiment.py` | FinBERT 核心引擎 |
| `finbert_enhanced_scoring.py` | 情绪评分整合 |
| `manual_sentiment_loader.py` | 人工标注加载器 |
| `manual_sentiment_override.json` | 题材标注数据库 ⭐ |
| `check_stock_news.py` | 新闻查看工具 |
| `taiwan_news_sentiment.py` | 台湾新闻分析（实验） |

---

## 💡 最佳实践

### ✅ DO
- 美股优先使用 FinBERT
- 妖股必须人工标注
- 定期检查标注有效期
- 验证新闻是否相关

### ❌ DON'T
- 不要盲目相信AI评分
- 不要忽略人工判断
- 不要忘记备份
- 不要标注过期题材

---

## 🎯 实战案例

### OMER (成功案例)
- FinBERT: +0.141 (轻微正面)
- 新闻: FDA批准新药
- 调整: +5分
- 结果: 买入信号 ✅

### 6443 元晶 (需人工)
- FinBERT: 无新闻 ❌
- 人工标注: SpaceX题材 +20分
- 结果: 手动覆盖 ✅

---

**完整文档:** [FinBERT_README.md](FinBERT_README.md)
