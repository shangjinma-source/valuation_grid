# Intraday Fund Valuation (盘中基金估值)

基于公开持仓数据和实时行情，对中国公募基金进行盘中净值估算的轻量级后端服务。

## 核心功能

- **盘中估值** — 根据最新季报持仓权重 × 个股实时涨跌幅，加权计算基金日内估值变动
- **置信度评估** — 综合持仓覆盖率（70%权重）和持仓数据时效性（30%权重）输出置信度评分
- **ETF 联接穿透** — 自动识别 ETF 联接基金，穿透至底层 ETF 持仓进行估值；支持手动映射
- **自选板块管理** — 用户可自定义板块分组，批量管理基金观察列表
- **近 5 日涨幅** — 批量估值时并发获取基金近 5 个交易日实际净值涨幅

## 技术架构

```
┌──────────────┐      ┌──────────────┐      ┌──────────────────────┐
│   Client     │─────▶│   app.py     │─────▶│   core.py            │
│  (REST API)  │      │  FastAPI 路由 │      │  估值计算 / State 管理│
└──────────────┘      └──────────────┘      └───────┬──────────────┘
                                                    │
                                              ┌─────▼──────────────┐
                                              │   providers.py     │
                                              │  数据源抓取 / 缓存  │
                                              └─────┬──────────────┘
                                                    │
                                    ┌───────────────┼───────────────┐
                                    ▼               ▼               ▼
                             天天基金 API      新浪行情 API     东方财富 API
                            (持仓 / 基金名)    (实时股价)      (近5日涨幅)
```

### 文件结构

```
intraday_valuation/
├── app.py              # FastAPI 应用入口，定义全部 REST 接口
├── core.py             # 估值引擎：加权计算、置信度评分、State 读写
├── providers.py        # 数据提供层：持仓抓取、行情拉取、缓存管理
├── data/
│   └── state.json      # 用户自选板块与基金列表（运行时自动生成）
├── cache/
│   └── holdings_*.json # 基金持仓缓存文件（30天有效期，自动生成/定期刷新）
├── demo.html           # 前端交互界面（独立原型 UI）
└── README.md
```

## 快速启动

```bash
# 安装依赖
pip install fastapi uvicorn pydantic

# 启动服务
python app.py

# 或使用热重载模式
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

服务默认运行在 `http://localhost:8000`。

## API 接口

### State 管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/state` | 读取全部自选板块和基金列表 |
| `POST` | `/v1/state` | 覆盖保存板块和基金列表 |

### 估值

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/valuation/{fund_code}` | 单基金盘中估值 |
| `POST` | `/v1/valuation/batch` | 批量估值（上限 500 只） |
| `GET` | `/v1/valuation/state` | 按自选板块分组返回全部估值 |

### 基金信息

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/v1/fund/{fund_code}/name` | 查询基金名称 |

### ETF 联接映射

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/etf-link` | 设置联接基金 → 目标 ETF 映射 |
| `DELETE` | `/v1/etf-link/{link_code}` | 删除联接基金映射 |

### 持仓缓存管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/v1/holdings/refresh` | 手动触发持仓缓存刷新 |

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 服务健康检查 |

## 接口示例

### 单基金估值

```bash
curl http://localhost:8000/v1/valuation/017193
```

```json
{
  "fund_code": "017193",
  "fund_name": "天弘中证有色金属指数C",
  "asof_time": "2026-01-31 14:30:00",
  "holdings_asof_date": "2024-12-31",
  "estimation_change": -1.2345,
  "week_change": null,
  "confidence": 0.85,
  "coverage": {
    "stock_total_weight": 94.52,
    "parsed_weight": 90.12,
    "covered_weight": 90.12,
    "residual_weight": 4.40,
    "missing_tickers": ["688xxx"]
  },
  "notes": ["残差4.4%按平均涨幅-1.37%估算", "持仓日期: 2024-12-31"]
}
```

### 保存自选板块

```bash
curl -X POST http://localhost:8000/v1/state \
  -H "Content-Type: application/json" \
  -d '{
    "sectors": [
      {
        "name": "有色金属",
        "funds": [
          {"code": "017193", "alias": "天弘有色"},
          {"code": "000216", "alias": ""}
        ]
      }
    ]
  }'
```

### 全板块估值

```bash
curl http://localhost:8000/v1/valuation/state
```

## 估值算法说明

### 估值涨跌幅 (`estimation_change`)

$$\text{estimation\_change} = \sum_{i} w_i \times \Delta p_i$$

- $w_i$：第 $i$ 只重仓股占基金净值比例（来自最新季报）
- $\Delta p_i$：该股票当日实时涨跌幅

对于未覆盖的残差权重，使用已覆盖持仓的加权平均涨跌幅进行补偿估算。

### 置信度 (`confidence`)

$$\text{confidence} = 0.7 \times C_{\text{coverage}} + 0.3 \times C_{\text{freshness}}$$

- **覆盖率** $C_{\text{coverage}}$：已获取行情的股票权重 / 股票总仓位
- **时效性** $C_{\text{freshness}}$：持仓数据距今天数的线性衰减
  - ≤ 30 天 → 1.0
  - ≥ 180 天 → 0.0
  - 中间线性插值

### 覆盖率字段 (`coverage`)

| 字段 | 说明 |
|------|------|
| `stock_total_weight` | 季报中股票总仓位（%） |
| `parsed_weight` | 成功解析的持仓权重合计（%） |
| `covered_weight` | 已获取实时行情的持仓权重合计（%） |
| `residual_weight` | 残差权重 = `stock_total_weight` - `covered_weight` |
| `missing_tickers` | 未获取到行情的股票代码列表 |

## 缓存策略

| 数据类型 | 存储方式 | 有效期 | 说明 |
|----------|---------|--------|------|
| 基金持仓 | 文件缓存 (`cache/`) | 30 天 | 后台每 6 小时自动刷新即将过期的缓存 |
| 股票行情 | 内存缓存 | 8 秒 | 盘中行情高频更新 |
| 近5日涨幅 | 内存缓存 | 1 小时 | 基金净值每日更新一次 |

## 数据来源

| 数据 | 来源 | 接口 |
|------|------|------|
| 基金持仓 / 名称 / 仓位比例 | 天天基金 | `fund.eastmoney.com` |
| 持仓明细 | 东方财富 | `fundf10.eastmoney.com` |
| 股票实时行情 | 新浪财经 | `hq.sinajs.cn` |
| 基金净值 / 近5日涨幅 | 东方财富 | `api.fund.eastmoney.com` |

## 运行依赖

- Python 3.8+
- [FastAPI](https://fastapi.tiangolo.com/)
- [Uvicorn](https://www.uvicorn.org/)
- [Pydantic](https://docs.pydantic.dev/)

无需其他第三方数据库或消息队列，所有数据通过文件和内存缓存管理。
