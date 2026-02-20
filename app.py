"""
app.py - API入口：/state + /valuation + /fund/name
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

# ============================================================
# 启动时刷新持仓缓存（后台线程，不阻塞服务就绪）
# ============================================================

@asynccontextmanager
async def lifespan(app):
    t = threading.Thread(target=refresh_stale_holdings, daemon=True)
    t.start()
    yield

app = FastAPI(
    title="盘中估值工具",
    description="基金盘中估值 + 本地自选板块管理",
    version="2.0.0",
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)