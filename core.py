"""
core.py - 估值计算 + state管理 + coverage/confidence
"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from providers import get_holdings, get_quotes, get_fund_5day_change

# === 配置 ===
DATA_DIR = Path(__file__).parent / "data"
STATE_FILE = DATA_DIR / "state.json"

# === 文件锁 ===
_state_lock = threading.Lock()

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# State 管理（板块+基金持久化）
# ============================================================

def _empty_state() -> dict:
    return {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": []
    }

def load_state() -> dict:
    _ensure_data_dir()
    if not STATE_FILE.exists():
        return _empty_state()
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return _empty_state()

def save_state(state: dict) -> bool:
    _ensure_data_dir()
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state.setdefault("version", 1)

    with _state_lock:
        try:
            tmp_file = STATE_FILE.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            tmp_file.replace(STATE_FILE)
            return True
        except:
            return False

def validate_state(state: dict) -> tuple:
    if not isinstance(state, dict):
        return False, "state必须是对象"
    if "sectors" not in state:
        return False, "缺少sectors字段"
    if not isinstance(state["sectors"], list):
        return False, "sectors必须是数组"
    for i, sector in enumerate(state["sectors"]):
        if not isinstance(sector, dict):
            return False, f"sectors[{i}]必须是对象"
        if "name" not in sector:
            return False, f"sectors[{i}]缺少name字段"
        if "funds" not in sector or not isinstance(sector["funds"], list):
            return False, f"sectors[{i}]缺少funds数组"
    return True, "ok"

# ============================================================
# 估值计算
# ============================================================

def _calc_staleness_score(holdings_date_str: Optional[str]) -> float:
    """计算持仓时效性得分（0~1）"""
    if not holdings_date_str:
        return 0.0
    try:
        holdings_date = datetime.strptime(holdings_date_str, "%Y-%m-%d")
        days_old = (datetime.now() - holdings_date).days
        if days_old <= 30:
            return 1.0
        elif days_old >= 180:
            return 0.0
        else:
            return 1.0 - (days_old - 30) / 150.0
    except:
        return 0.0

def calculate_valuation(fund_code: str) -> dict:
    """计算单基金盘中估值涨跌幅"""
    result = {
        "fund_code": fund_code,
        "fund_name": None,
        "asof_time": None,
        "holdings_asof_date": None,
        "estimation_change": None,
        "week_change": None,
        "confidence": 0.0,
        "coverage": {
            "stock_total_weight": 0.0,
            "parsed_weight": 0.0,
            "covered_weight": 0.0,
            "residual_weight": 0.0,
            "missing_tickers": []
        },
        "notes": []
    }

    # 获取近5日涨幅（由批量路由统一处理，此处不调用）

    # 1. 获取持仓
    holdings = get_holdings(fund_code)

    if holdings.get("error"):
        result["notes"].append(f"持仓获取失败: {holdings['error']}")
        return result

    if not holdings.get("positions"):
        result["notes"].append("无持仓数据")
        return result

    result["fund_name"] = holdings.get("fund_name")
    result["holdings_asof_date"] = holdings.get("holdings_asof_date")
    result["coverage"]["stock_total_weight"] = holdings.get("stock_total_weight", 0)
    result["coverage"]["parsed_weight"] = holdings.get("parsed_weight", 0)

    if holdings.get("_stale"):
        result["notes"].append("使用过期缓存持仓")

    if holdings.get("is_etf_link"):
        result["notes"].append(f"ETF联接穿透: {holdings.get('etf_target')}")

    # 2. 获取行情
    tickers = [p["stock_code"] for p in holdings["positions"]]
    quotes_result = get_quotes(tickers)
    quotes = quotes_result["quotes"]
    missing = quotes_result["missing"]

    if not quotes:
        result["notes"].append("无法获取任何行情数据")
        result["coverage"]["missing_tickers"] = missing
        return result

    # 3. 计算估值涨跌幅
    estimation_change = 0.0
    covered_weight = 0.0
    asof_time = None

    for pos in holdings["positions"]:
        code = pos["stock_code"]
        weight = pos["weight"]

        if code in quotes:
            pct_change = quotes[code]["pct_change"]
            estimation_change += weight * pct_change / 100.0
            covered_weight += weight
            if asof_time is None:
                asof_time = quotes[code]["asof_time"]

    # 残差权重 = 股票总仓位 - 已覆盖权重
    stock_total = holdings.get("stock_total_weight", 0)
    parsed_weight = holdings.get("parsed_weight", 0)

    # 数据保护：如果持仓权重合计 > 股票总仓位（可能来自不同报告期），
    # 以实际持仓权重为准，避免残差为负或覆盖率溢出
    if parsed_weight > stock_total:
        stock_total = parsed_weight
        result["coverage"]["stock_total_weight"] = stock_total

    residual_weight = stock_total - covered_weight

    # 对于残差部分，用已覆盖持仓的平均涨跌幅来估算
    if covered_weight > 0 and residual_weight > 0:
        avg_change = estimation_change / covered_weight * 100
        residual_contribution = residual_weight * avg_change / 100
        estimation_change += residual_contribution
        result["notes"].append(f"残差{residual_weight:.1f}%按平均涨幅{avg_change:.2f}%估算")

    result["estimation_change"] = round(estimation_change, 4)
    result["asof_time"] = asof_time
    result["coverage"]["covered_weight"] = round(covered_weight, 2)
    result["coverage"]["residual_weight"] = round(max(0, residual_weight), 2)
    result["coverage"]["missing_tickers"] = missing

    # 4. 计算置信度
    if stock_total > 0:
        coverage_score = covered_weight / stock_total
    else:
        coverage_score = 0.0

    staleness_score = _calc_staleness_score(result["holdings_asof_date"])
    confidence = 0.7 * coverage_score + 0.3 * staleness_score
    result["confidence"] = round(confidence, 3)

    # 5. 添加说明
    if missing:
        result["notes"].append(f"缺失{len(missing)}只股票行情")
    if result["holdings_asof_date"]:
        result["notes"].append(f"持仓日期: {result['holdings_asof_date']}")

    # 6. 附带近3个交易日真实涨跌幅（已结算数据，不含今天）
    from providers import get_fund_nav_history
    history = get_fund_nav_history(fund_code, 5)
    today_str = datetime.now().strftime("%Y-%m-%d")
    result["recent_changes"] = [
        {"date": h["date"], "change": h["change"]}
        for h in history if h["date"] != today_str
    ][:3]  # 过滤今天后取前3条

    # 7. 收盘后用真实净值涨跌替代估值
    #    盘中估值只在交易时段可靠，收盘后新浪行情数据不再反映当日真实涨跌
    #    （尤其含港股持仓时，A股/港股收盘时间不同导致偏差更大）
    #    替换条件：当天已收盘（15:05后、或非交易日）且有真实净值数据
    if _is_market_closed() and result["recent_changes"]:
        latest = result["recent_changes"][0]
        if latest["change"] is not None:
            result["estimation_change"] = round(latest["change"], 4)
            result["_source"] = "nav"  # 标记数据来源：净值
            result["notes"].append(f"使用真实净值涨跌 {latest['date']}")
        else:
            result["_source"] = "estimation"
    else:
        result["_source"] = "estimation"

    return result


def _is_market_closed() -> bool:
    """判断当天是否已收盘（或非交易日）
    True = 可以安全用真实净值替代估值
    盘前/盘中/午休 都返回 False，只有15:05后和非交易日返回 True
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=周一 ... 6=周日
    if weekday >= 5:
        return True  # 周末
    hhmm = now.hour * 100 + now.minute
    if hhmm >= 1505:
        return True  # 收盘后
    if hhmm < 915:
        return True  # 盘前
    return False  # 盘中或午休，继续用估值

def calculate_valuation_batch(fund_codes: List[str]) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor
    from providers import get_fund_5day_change

    # 1. 批量计算估值
    results = [calculate_valuation(code) for code in fund_codes]

    # 2. 并发获取近5日涨幅
    week_data = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(get_fund_5day_change, code): code for code in fund_codes}
        for future in futures:
            code = futures[future]
            try:
                week_data[code] = future.result(timeout=20)
            except:
                week_data[code] = None

    # 3. 合并
    for r in results:
        r["week_change"] = week_data.get(r["fund_code"])

    return results

def calculate_valuation_by_state() -> dict:
    state = load_state()
    result = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": []
    }

    for sector in state.get("sectors", []):
        sector_result = {
            "name": sector["name"],
            "funds": []
        }
        for fund in sector.get("funds", []):
            code = fund.get("code", "")
            if code:
                val = calculate_valuation(code)
                val["alias"] = fund.get("alias", "")
                sector_result["funds"].append(val)
        result["sectors"].append(sector_result)

    return result


if __name__ == "__main__":
    print("=== 测试单基金估值 ===")
    v = calculate_valuation("017193")
    print(json.dumps(v, ensure_ascii=False, indent=2))