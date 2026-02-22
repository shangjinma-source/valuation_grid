"""
app.py - API入口：/state + /valuation + /fund/name + /position + /strategy
"""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from core import (
    load_state, save_state, validate_state,
    calculate_valuation, calculate_valuation_batch, calculate_valuation_by_state
)
from providers import (
    get_fund_name, set_etf_link_target, get_etf_link_target, clear_etf_link_target,
    refresh_stale_holdings
)
from positions import (
    add_batch, sell_batch, delete_batch, update_fund_config,
    get_fund_position, get_all_positions, remove_fund, update_sell_nav,
    delete_sell_record, update_fee_schedule, sell_fifo,
    get_groups, add_group, update_group, delete_group,
    make_fund_key, parse_fund_key, rename_fund_key
)
from strategy import (
    generate_signal, generate_all_signals, get_signal_history,
    get_vol_sensitivity_info, update_vol_sensitivity, clear_vol_sensitivity,
    auto_calibrate_vol_sensitivity,
    backfill_signal_outcomes, calc_signal_win_rate
)

# ============================================================
# 启动时刷新持仓缓存（后台线程，不阻塞服务就绪）
# ============================================================

@asynccontextmanager
async def lifespan(app):
    t = threading.Thread(target=refresh_stale_holdings, daemon=True)
    t.start()
    yield

app = FastAPI(
    title="Ease Grid",
    description="基金盘中估值 + 本地自选板块管理 + 低频网格交易策略",
    version="2.2.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 数据模型
# ============================================================

class FundItem(BaseModel):
    code: str
    alias: Optional[str] = ""

class SectorItem(BaseModel):
    name: str
    funds: List[FundItem]

class StateModel(BaseModel):
    version: Optional[int] = 1
    updated_at: Optional[str] = None
    sectors: List[SectorItem]

class BatchRequest(BaseModel):
    fund_codes: List[str]

# ============================================================
# State 管理 API
# ============================================================

@app.get("/v1/state")
def get_state():
    """读取全部状态（板块+基金列表）"""
    return load_state()

@app.post("/v1/state")
def post_state(state: StateModel):
    """覆盖保存状态"""
    state_dict = state.model_dump()
    valid, msg = validate_state(state_dict)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    if save_state(state_dict):
        return {"success": True, "message": "保存成功"}
    else:
        raise HTTPException(status_code=500, detail="保存失败")

# ============================================================
# 基金信息 API
# ============================================================

@app.get("/v1/fund/{fund_code}/name")
def get_fund_name_api(fund_code: str):
    """获取基金名称"""
    name = get_fund_name(fund_code)
    if name:
        return {"fund_code": fund_code, "name": name}
    else:
        return {"fund_code": fund_code, "name": None, "error": "无法获取基金名称"}

@app.get("/v1/fund/{fund_code}/nav-history")
def get_nav_history(fund_code: str, days: int = 15):
    """获取基金最近N日真实净值涨跌"""
    from providers import get_fund_nav_history
    data = get_fund_nav_history(fund_code, days)
    return {"fund_code": fund_code, "days": len(data), "history": data}

# ============================================================
# ETF联接基金映射 API
# ============================================================

class ETFLinkRequest(BaseModel):
    link_code: str
    etf_code: str

@app.post("/v1/etf-link")
def set_etf_link(req: ETFLinkRequest):
    """设置ETF联接基金的目标ETF映射"""
    if not req.link_code or not req.etf_code:
        raise HTTPException(status_code=400, detail="link_code和etf_code不能为空")
    if len(req.link_code) != 6 or len(req.etf_code) != 6:
        raise HTTPException(status_code=400, detail="基金代码必须是6位")

    set_etf_link_target(req.link_code, req.etf_code)

    # 清除该基金的持仓缓存，以便重新获取
    from pathlib import Path
    cache_path = Path(__file__).parent / "cache" / f"holdings_{req.link_code}.json"
    if cache_path.exists():
        cache_path.unlink()

    return {"success": True, "message": f"已设置 {req.link_code} -> {req.etf_code}"}

@app.delete("/v1/etf-link/{link_code}")
def delete_etf_link(link_code: str):
    """删除ETF联接基金映射"""
    clear_etf_link_target(link_code)
    return {"success": True, "message": f"已清除 {link_code} 的映射"}


# ============================================================
# 估值 API
# ============================================================

@app.get("/v1/valuation/{fund_code}")
def get_valuation(fund_code: str):
    """单基金估值"""
    return calculate_valuation(fund_code)

@app.post("/v1/valuation/batch")
def post_valuation_batch(req: BatchRequest):
    """批量估值"""
    if not req.fund_codes:
        return {"items": []}
    if len(req.fund_codes) > 500:
        raise HTTPException(status_code=400, detail="单次最多500只基金")
    return {"items": calculate_valuation_batch(req.fund_codes)}

@app.get("/v1/valuation/state")
def get_valuation_state():
    """按当前state返回所有基金估值（板块分组）"""
    return calculate_valuation_by_state()

# ============================================================
# 持仓缓存刷新 API
# ============================================================

@app.post("/v1/holdings/refresh")
def post_holdings_refresh():
    """手动触发持仓缓存刷新（刷新所有即将过期的基金）"""
    summary = refresh_stale_holdings()
    return summary

# ============================================================
# 健康检查
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok"}


# ============================================================
# 持仓管理 API（新增）
# ============================================================

class BuyRequest(BaseModel):
    amount: float
    nav: float
    note: Optional[str] = ""
    buy_date: Optional[str] = None  # 格式 YYYY-MM-DD，默认今天
    is_supplement: Optional[bool] = False  # 是否为补仓买入
    owner: Optional[str] = ""  # 持仓所有者标签（多人模式）

class SellRequest(BaseModel):
    batch_id: str
    sell_shares: float              # 卖出份额（必填）
    sell_nav: Optional[float] = None  # 确认净值（选填，待确认时不填）
    sell_date: Optional[str] = None   # 卖出日期 YYYY-MM-DD

class FundConfigRequest(BaseModel):
    max_position: Optional[int] = None
    fund_name: Optional[str] = None

class FeeScheduleItem(BaseModel):
    days: Optional[int] = None  # None = 兜底档
    rate: float                 # 费率%

class FeeScheduleRequest(BaseModel):
    schedule: List[FeeScheduleItem]


@app.get("/v1/positions")
def get_positions():
    """获取全部持仓"""
    return get_all_positions()


@app.get("/v1/position/{fund_code}")
def get_position(fund_code: str):
    """获取单基金持仓汇总"""
    return get_fund_position(fund_code)


@app.post("/v1/position/{fund_code}/buy")
def buy_fund(fund_code: str, req: BuyRequest):
    """新增买入批次（支持 owner 参数实现同基金多人持仓）"""
    if req.amount <= 0 or req.nav <= 0:
        raise HTTPException(status_code=400, detail="amount和nav必须大于0")
    fund_key = make_fund_key(fund_code, req.owner or "")
    batch = add_batch(fund_key, req.amount, req.nav, req.note or "",
                      req.buy_date, req.is_supplement)
    return {"success": True, "batch": batch, "fund_key": fund_key}


@app.post("/v1/position/{fund_code}/sell")
def sell_fund(fund_code: str, req: SellRequest):
    """卖出批次（按份额）"""
    if req.sell_shares <= 0:
        raise HTTPException(status_code=400, detail="卖出份额必须大于0")
    try:
        result = sell_batch(fund_code, req.batch_id, req.sell_shares,
                            req.sell_nav, req.sell_date)
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/v1/position/{fund_code}/batch/{batch_id}")
def delete_fund_batch(fund_code: str, batch_id: str):
    """删除错误录入的批次"""
    if delete_batch(fund_code, batch_id):
        return {"success": True, "message": f"已删除 {batch_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"批次 {batch_id} 不存在")


@app.delete("/v1/position/{fund_code}")
def remove_fund_position(fund_code: str):
    """从策略面板完全移除一只基金"""
    if remove_fund(fund_code):
        return {"success": True, "message": f"已移除 {fund_code}"}
    else:
        raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")


class RenameFundKeyRequest(BaseModel):
    new_owner: str  # 新标签（空字符串表示去除标签）

@app.post("/v1/position/{fund_code}/rename")
def rename_fund_key_api(fund_code: str, req: RenameFundKeyRequest):
    """给已有持仓设置/修改标签（重命名存储键）"""
    real_code, _ = parse_fund_key(fund_code)
    new_key = make_fund_key(real_code, req.new_owner.strip())
    if fund_code == new_key:
        return {"success": True, "message": "标签未变化", "new_key": new_key}
    if rename_fund_key(fund_code, new_key):
        return {"success": True, "message": f"已重命名 {fund_code} → {new_key}", "new_key": new_key}
    else:
        raise HTTPException(status_code=400, detail="重命名失败（目标键可能已存在）")


@app.put("/v1/position/{fund_code}/config")
def update_config(fund_code: str, req: FundConfigRequest):
    """更新基金配置"""
    if update_fund_config(fund_code, req.max_position, req.fund_name):
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="更新失败")


@app.put("/v1/position/{fund_code}/fee-schedule")
def update_fee_schedule_api(fund_code: str, req: FeeScheduleRequest):
    """更新基金卖出费率表"""
    schedule = [{"days": s.days, "rate": s.rate} for s in req.schedule]
    if update_fee_schedule(fund_code, schedule):
        return {"success": True, "schedule": schedule}
    else:
        raise HTTPException(status_code=500, detail="更新失败")


class UpdateSellNavRequest(BaseModel):
    sell_record_id: str
    sell_nav: float


class SellFifoRequest(BaseModel):
    total_sell_shares: float
    sell_nav: Optional[float] = None
    sell_date: Optional[str] = None


@app.post("/v1/position/{fund_code}/sell-fifo")
def sell_fund_fifo(fund_code: str, req: SellFifoRequest):
    """按FIFO顺序卖出指定总份额（模拟支付宝先进先出行为）"""
    if req.total_sell_shares <= 0:
        raise HTTPException(status_code=400, detail="卖出份额必须大于0")
    try:
        result = sell_fifo(fund_code, req.total_sell_shares,
                           req.sell_nav, req.sell_date)
        return {"success": True, **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/v1/position/{fund_code}/sell-nav")
def update_sell_nav_api(fund_code: str, req: UpdateSellNavRequest):
    """补录卖出确认净值"""
    if req.sell_nav <= 0:
        raise HTTPException(status_code=400, detail="净值必须大于0")
    try:
        result = update_sell_nav(fund_code, req.sell_record_id, req.sell_nav)
        return {"success": True, "record": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/v1/position/{fund_code}/sell-record/{sell_record_id}")
def delete_sell_record_api(fund_code: str, sell_record_id: str):
    """删除卖出记录"""
    if delete_sell_record(fund_code, sell_record_id):
        return {"success": True, "message": f"已删除卖出记录 {sell_record_id}"}
    else:
        raise HTTPException(status_code=404, detail=f"卖出记录 {sell_record_id} 不存在")


# ============================================================
# 策略信号 API（新增）
# ============================================================

@app.get("/v1/strategy/signals")
def get_all_strategy_signals():
    """获取全部基金策略信号"""
    return generate_all_signals()


@app.get("/v1/strategy/signal/{fund_code}")
def get_strategy_signal(fund_code: str):
    """获取单基金策略信号"""
    return generate_signal(fund_code)


@app.get("/v1/strategy/history")
def get_all_signal_history(limit: int = 30):
    """获取全部基金信号历史"""
    return get_signal_history(limit=limit)


@app.get("/v1/strategy/history/{fund_code}")
def get_fund_signal_history(fund_code: str, limit: int = 30):
    """获取单基金信号历史"""
    return get_signal_history(fund_code=fund_code, limit=limit)


@app.post("/v1/strategy/backfill")
def backfill_outcomes():
    """回填历史信号的 outcome 字段（建议收盘后调用）"""
    updated = backfill_signal_outcomes()
    return {"success": True, "updated": updated}


@app.get("/v1/strategy/win-rate")
def get_all_win_rate(lookback: int = 30):
    """获取全部基金信号胜率统计"""
    return calc_signal_win_rate(lookback=lookback)


@app.get("/v1/strategy/win-rate/{fund_code}")
def get_fund_win_rate(fund_code: str, lookback: int = 30):
    """获取单基金信号胜率统计"""
    return calc_signal_win_rate(fund_code=fund_code, lookback=lookback)


# ============================================================
# 波动率灵敏度 API
# ============================================================

@app.get("/v1/position/{fund_code}/vol-sensitivity")
def get_vol_sensitivity_api(fund_code: str):
    """获取基金波动率灵敏度信息"""
    return get_vol_sensitivity_info(fund_code)


class VolSensitivityRequest(BaseModel):
    sensitivity: float


@app.put("/v1/position/{fund_code}/vol-sensitivity")
def set_vol_sensitivity_api(fund_code: str, req: VolSensitivityRequest):
    """手动设置波动率灵敏度（覆盖自动校准）"""
    if req.sensitivity < 0.5 or req.sensitivity > 1.5:
        raise HTTPException(status_code=400, detail="灵敏度范围 0.5~1.5")
    if update_vol_sensitivity(fund_code, req.sensitivity):
        return {"success": True, "sensitivity": round(req.sensitivity, 2)}
    raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")


@app.delete("/v1/position/{fund_code}/vol-sensitivity")
def clear_vol_sensitivity_api(fund_code: str):
    """清除手动设置，恢复自动校准"""
    if clear_vol_sensitivity(fund_code):
        return {"success": True, "message": "已恢复自动校准"}
    raise HTTPException(status_code=404, detail=f"基金 {fund_code} 不存在")


@app.post("/v1/position/{fund_code}/vol-sensitivity/calibrate")
def recalibrate_vol_sensitivity_api(fund_code: str):
    """强制重新校准（清除缓存并立即计算）"""
    # 先清除缓存
    clear_vol_sensitivity(fund_code)
    # 触发重新计算
    info = get_vol_sensitivity_info(fund_code)
    return {"success": True, **info}


# ============================================================
# 分组管理 API
# ============================================================

class GroupCreateRequest(BaseModel):
    name: str

class GroupUpdateRequest(BaseModel):
    name: Optional[str] = None
    fund_codes: Optional[List[str]] = None

@app.get("/v1/groups")
def get_groups_api():
    """获取全部分组"""
    return {"groups": get_groups()}

@app.post("/v1/groups")
def create_group(req: GroupCreateRequest):
    """新建分组"""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="分组名称不能为空")
    group = add_group(req.name.strip())
    return {"success": True, "group": group}

@app.put("/v1/groups/{group_id}")
def update_group_api(group_id: str, req: GroupUpdateRequest):
    """更新分组"""
    if update_group(group_id, req.name, req.fund_codes):
        return {"success": True}
    raise HTTPException(status_code=404, detail="分组不存在")

@app.delete("/v1/groups/{group_id}")
def delete_group_api(group_id: str):
    """删除分组"""
    if delete_group(group_id):
        return {"success": True}
    raise HTTPException(status_code=404, detail="分组不存在")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)