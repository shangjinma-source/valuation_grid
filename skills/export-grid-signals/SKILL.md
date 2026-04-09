# Skill: export-grid-signals

## 功能

调用 valuation_grid 后端接口，获取低频网格策略信号的文字摘要。

每份信号包含：
- 信号名称（持有等待/补仓/止盈/大跌抄底等）
- 操作建议（买入金额/卖出份额）
- 操作原因

## 接口

```
GET http://localhost:8000/v1/signals/grid
```

## 定时任务

- **交易日 14:50** 执行一次（盘中信号，最接近收盘决策的时段）
- 非交易日不执行

## 调用方式

```python
from scripts.run import export_grid_signals
result = export_grid_signals()
```

返回值：
```json
{
  "success": true,
  "count": 3,
  "signals": [...],
  "message": "📋 老婆 · 网格信号 14:50\n...\n📋 老板 · 网格信号 14:50\n..."
}
```

## 消息模板

成功时，发送文字消息到飞书群：

```
📋 {owner} · 网格信号 {HH:MM}
========================================
  🟢 {fund_code} {fund_name}
     建议：买入 ¥{amount:.2f}
     原因：{reason}

  🔴 {fund_code} {fund_name}
     建议：卖出 {sell_shares}份 (该批次{sell_pct}%)
     批次：{target_batch}

  ⚪ {fund_code} {fund_name}
     建议：持有等待
```

无数据时：

```
今日暂无网格信号
```

## 注意事项

- 后端 generate_all_signals() 并发请求外部数据源，超时设 180 秒
- 纯标准库，零依赖
- 只输出文字建议，不生成图片
