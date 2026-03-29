---
name: export-valuation-images
description: 定时导出基金实时估值图片。自动启动/检测本地 FastAPI 后端，获取板块估值图片（base64），解码保存为 PNG 文件。仅在交易日 15:00/22:00 执行。返回结构化结果供 agent 发送到飞书群。
---

# 基金估值图片导出技能

## 功能概述

本技能用于定时导出基金实时估值图片，从本地运行的估值系统获取各板块的估值表格截图，解码并保存为 PNG 文件，供 agent 后续发送到飞书群。

## 何时使用

**触发条件：**
- 用户请求导出/发送基金估值图片
- 定时任务触发（交易日 15:00 或 22:00）
- 需要获取实时估值数据并分享

**不触发：**
- 非交易日（周末、法定节假日）
- 非执行时间窗口
- 用户仅查询估值数据（不需要图片）

## 核心功能

### 1. 自动启动后端服务

- 检测端口 8000 是否有服务运行
- 如未运行，自动启动 `uvicorn app:app`
- 等待服务就绪（最多 30 秒）

### 2. 调用后端接口

```
GET http://localhost:8000/v1/export/images
```

返回格式：
```json
{
  "count": 3,
  "images": [
    {
      "sector": "科技半导体",
      "filename": "科技半导体_0330_1430.png",
      "image_base64": "iVBORw0KGgoAAAANSUhEUg..."
    }
  ]
}
```

### 2. 图片处理

- 将 base64 解码为 PNG 文件
- 保存到临时目录：`%TEMP%/valuation_export_YYYYMMDD_HHMMSS/`
- 使用接口返回的 filename 作为文件名

### 3. 返回结构化结果

成功：
```json
{
  "success": true,
  "count": 3,
  "images": [
    {"sector": "科技半导体", "filepath": "C:\\Users\\...\\科技半导体_0330_1430.png"},
    {"sector": "消费医药", "filepath": "C:\\Users\\...\\消费医药_0330_1430.png"}
  ]
}
```

失败：
```json
{
  "success": false,
  "error": "错误信息"
}
```

## 定时任务规则

| 时段 | 时间窗口 | 说明 |
|------|----------|------|
| 盘中估值 | 14:45 - 15:15 | 最接近收盘值的时段 |
| 收盘净值 | 21:45 - 22:15 | 净值开始公布后 |

**非交易日不执行**（周六、周日、法定节假日）

## 使用方法

### 直接调用

```python
from export_valuation_images import export_valuation_images

result = export_valuation_images()

if result["success"]:
    if result["count"] > 0:
        # 发送每张图片到飞书
        for img in result["images"]:
            send_to_feishu(
                caption=f"📊 {img['sector']} · 实时估值 {HH:MM}",
                image_path=img["filepath"]
            )
    else:
        send_to_feishu(text="今日无可导出的估值数据")
else:
    print(f"导出失败：{result['error']}")
```

### Agent 消息模板

**成功导出（每张图片独立消息）：**
```
📊 {sector} · 实时估值 {HH:MM}
[图片附件]
```

**无数据（count=0）：**
```
今日无可导出的估值数据
```

## 错误处理

| 错误类型 | 返回 |
|----------|------|
| 网络错误 | `{"success": false, "error": "网络错误：..."}` |
| HTTP 错误 | `{"success": false, "error": "HTTP 错误 404: ..."}` |
| JSON 解析错误 | `{"success": false, "error": "JSON 解析错误：..."}` |
| 非执行时间 | `{"success": false, "error": "非执行时间：仅在交易日 15:00 或 22:00 执行"}` |

## 技术细节

- **接口超时：** 120 秒（批量估值需要时间）
- **仅使用标准库：** urllib、json、base64、tempfile、os、subprocess、socket
- **图片格式：** PNG（2x 高清）
- **临时文件：** 自动保存到系统临时目录 `%TEMP%/valuation_export_YYYYMMDD_HHMMSS/`
- **自动启动：** 检测并启动后端服务（uvicorn），无需手动操作

## 注意事项

1. 后端接口首次调用可能需要 30-60 秒（并发请求外部数据源）
2. base64 解码后即为标准 PNG 文件，无需额外处理
3. 如果接口返回 `count: 0`，不发送图片消息
4. 每张图片作为独立消息发送到飞书群

## 脚本位置

```
E:\Git\valuation_grid\skills\export-valuation-images\scripts\export_valuation_images.py
```

## 测试运行

```bash
cd E:\Git\valuation_grid\skills\export-valuation-images\scripts
python export_valuation_images.py
```
