"""
core.py - 估值计算 + state管理 + coverage/confidence
"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from .providers import get_holdings, get_quotes, get_fund_5day_change, get_etf_realtime_change, get_fundgz_estimation

# === 配置 ===
DATA_DIR = Path(__file__).parent.parent / "data"
STATE_FILE = DATA_DIR / "state.json"

# === 置信度校准：估值偏差历史 ===
DEVIATION_FILE = DATA_DIR / "confidence_deviations.json"
_MAX_DEVIATION_RECORDS = 200  # 每基金最多保留条数

# === 盘中估值缓存（用于收盘后偏差记录） ===
# {fund_code: {"date": "2026-03-16", "est": -1.91}}
_intraday_estimation_cache: Dict[str, dict] = {}
_intraday_cache_loaded = False  # v5.22: 防止并发重复加载
_INTRADAY_CACHE_FILE = DATA_DIR / "intraday_cache.json"

# v5.22: deviation 文件读写锁，防止并发 load/save 导致历史数据丢失
_deviation_lock = threading.Lock()

# v5.22: 盘中缓存文件锁（保护 intraday_cache.json 的读写）
_intraday_cache_lock = threading.Lock()

def _ensure_intraday_cache_loaded():
    """进程重启后从文件恢复盘中估值缓存（线程安全，只加载一次）。
    v5.22: 从 calculate_valuation 内部提取出来，由 batch 入口统一调用，
    确保在任何并发估值计算之前完成加载。"""
    global _intraday_cache_loaded
    if _intraday_cache_loaded:
        return
    with _intraday_cache_lock:
        if _intraday_cache_loaded:  # double-check
            return
        if _INTRADAY_CACHE_FILE.exists():
            try:
                with open(_INTRADAY_CACHE_FILE, "r", encoding="utf-8") as f:
                    _intraday_estimation_cache.update(json.load(f))
            except Exception:
                pass
        _intraday_cache_loaded = True

# v5.22: 批量偏差缓冲区，收盘后先收集所有偏差，最后一次性写入
# {fund_code: {"date": ..., "est": ..., "nav": ..., "deviation": ...}}
_deviation_buffer: Dict[str, dict] = {}

def _load_deviations() -> dict:
    """加载偏差历史 {fund_code: [{date, est, nav, deviation}, ...]}
    调用方需自行持有 _deviation_lock"""
    _ensure_data_dir()
    if not DEVIATION_FILE.exists():
        return {}
    try:
        with open(DEVIATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_deviations(data: dict):
    """全量写入偏差文件。调用方需自行持有 _deviation_lock"""
    _ensure_data_dir()
    try:
        tmp = DEVIATION_FILE.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(DEVIATION_FILE)
    except Exception:
        pass

def record_deviation(fund_code: str, date: str, est_change: float, nav_change: float):
    """收集一条估值 vs 净值偏差到缓冲区（线程安全，不立即写文件）。
    v5.22: 改为先缓冲，由 flush_deviations() 统一写入，
    避免并发逐只写文件导致 load 到不完整数据覆盖历史。"""
    record = {
        "date": date,
        "est": round(est_change, 4),
        "nav": round(nav_change, 4),
        "deviation": round(abs(est_change - nav_change), 4)
    }
    with _deviation_lock:
        # 用 fund_code+date 作为 key 去重
        buf_key = f"{fund_code}__{date}"
        _deviation_buffer[buf_key] = (fund_code, record)

def flush_deviations():
    """将缓冲区中的偏差记录一次性合并写入文件（线程安全）。
    v5.22: 收盘后由 calculate_valuation_batch 在所有估值计算完成后统一调用。"""
    with _deviation_lock:
        if not _deviation_buffer:
            return
        devs = _load_deviations()
        for buf_key, (fund_code, record) in _deviation_buffer.items():
            records = devs.setdefault(fund_code, [])
            # 去重：同一天不重复记录
            if any(r["date"] == record["date"] for r in records):
                continue
            records.insert(0, record)
            # 截断
            devs[fund_code] = records[:_MAX_DEVIATION_RECORDS]
        _save_deviations(devs)
        _deviation_buffer.clear()

def calibrate_confidence(fund_code: str, raw_confidence: float) -> float:
    """基于历史偏差校准置信度。
    思路：如果历史偏差中位数很小（估值很准），则向上修正 confidence。
    数据不足（<5条）时不校准，返回原值。
    """
    with _deviation_lock:
        devs = _load_deviations()
    records = devs.get(fund_code, [])
    if len(records) < 1:
        return raw_confidence

    # 取最近30条偏差
    recent = [r["deviation"] for r in records[:30]]
    recent.sort()
    median_dev = recent[len(recent) // 2]

    # 偏差越小 → 校准后 confidence 越高
    # median_dev < 0.3pp → 估值非常准，confidence 至少 0.85
    # median_dev < 0.5pp → 估值较准，confidence 至少 0.70
    # median_dev < 1.0pp → 一般准，confidence 至少 0.60
    # median_dev >= 1.0pp → 不太准，不上调
    if median_dev < 0.3:
        floor = 0.85
    elif median_dev < 0.5:
        floor = 0.70
    elif median_dev < 1.0:
        floor = 0.60
    else:
        floor = raw_confidence  # 不校准

    calibrated = max(raw_confidence, floor)
    return round(calibrated, 3)

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

def _try_nav_fallback(result: dict, fund_code: str) -> dict:
    """当盘中估值无法计算时（QDII/LOF/无持仓），尝试用最新真实净值填充。
    返回填充后的 result（原地修改）。"""
    from .providers import get_fund_nav_history, get_fund_name
    history = get_fund_nav_history(fund_code, 5)
    if not history:
        return result

    today_str = datetime.now().strftime("%Y-%m-%d")

    # 优先用今天的净值，其次用最新一条
    entry = next(
        (h for h in history if h["date"] == today_str and h.get("change") is not None),
        None
    )
    if entry is None:
        entry = next((h for h in history if h.get("change") is not None), None)
    if entry is None:
        return result

    result["estimation_change"] = round(entry["change"], 4)
    result["_source"] = "nav"
    # 标记净值日期，前端据此显示"X月X日净值"
    result["_nav_date"] = entry["date"]
    if entry["date"] == today_str:
        result["notes"].append(f"使用当日真实净值涨跌")
    else:
        result["notes"].append(f"使用 {entry['date']} 真实净值涨跌")

    # 补充基金名称（如果还没有的话）
    if not result.get("fund_name"):
        result["fund_name"] = get_fund_name(fund_code)

    # recent_changes 供前端柱状图
    result["recent_changes"] = [
        {"date": h["date"], "change": h["change"]}
        for h in history if h.get("change") is not None
    ][:5]

    return result


def _try_etf_realtime_fallback(result: dict, fund_code: str) -> dict:
    """当基金是商品类ETF联接（黄金/原油等）时，
    直接用目标ETF的实时交易涨跌幅作为盘中估值。
    仅限白名单中的商品类ETF，避免影响正常股票类ETF联接基金。"""
    from .providers import get_etf_link_target, get_etf_realtime_change, get_fund_nav_history, get_fund_name

    etf_target = get_etf_link_target(fund_code)
    if not etf_target:
        return result

    # 仅限商品类ETF（黄金/原油等），底层持仓是合约无法解析股票持仓
    # 如需扩展，在此添加对应的目标ETF代码
    COMMODITY_ETF_CODES = {
        # 黄金ETF（跟踪Au99.99）
        "518800",   # 国泰黄金ETF
        "518880",   # 华安黄金ETF
        "159934",   # 易方达黄金ETF
        "159937",   # 博时黄金ETF
        "518660",   # 工银黄金ETF
        "159812",   # 前海开源黄金ETF
        # 上海金ETF（跟踪SHAU）
        "159830",   # 天弘上海金ETF
        "518600",   # 广发上海金ETF
        "518680",   # 富国上海金ETF
        "518860",   # 建信上海金ETF
        "159831",   # 嘉实上海金ETF
        "518890",   # 中银上海金ETF
    }
    if etf_target not in COMMODITY_ETF_CODES:
        return result

    quote = get_etf_realtime_change(etf_target)
    if not quote or quote.get("pct_change") is None:
        return result

    pct = quote["pct_change"]
    result["fund_name"] = get_fund_name(fund_code)
    result["estimation_change"] = round(pct, 4)  # pct已是百分比形式（如3.55），与nav fallback一致
    result["asof_time"] = quote.get("asof_time")
    result["confidence"] = 0.9  # ETF实时价格可靠度高
    result["_source"] = "etf_realtime"
    result["notes"].append(f"ETF实时行情: {etf_target} ({quote.get('name', '')})")

    # 附带近几日真实涨跌幅
    history = get_fund_nav_history(fund_code, 5)
    today_str = datetime.now().strftime("%Y-%m-%d")
    result["recent_changes"] = [
        {"date": h["date"], "change": h["change"]}
        for h in history if h["date"] != today_str and h.get("change") is not None
    ][:3]

    # 收盘后仍用真实净值替代
    if _is_market_closed():
        # v5.22: ETF路径也从盘中缓存取估值
        _cached = _intraday_estimation_cache.get(fund_code)
        _est_raw = (
            _cached["est"]
            if _cached and _cached["date"] == today_str and _cached["est"] is not None
            else pct  # fallback: 用当前ETF实时值
        )
        today_nav = next(
            (h for h in history if h["date"] == today_str and h.get("change") is not None),
            None
        )
        if today_nav:
            result["estimation_change"] = round(today_nav["change"], 4)
            result["_source"] = "nav"
            result["_nav_date"] = today_str
            result["notes"].append(f"收盘后使用真实净值 {today_str}")
            # v5.22: ETF路径也记录偏差（0%也是有效值）
            if _est_raw is not None:
                result["_estimation_change_raw"] = _est_raw
                record_deviation(fund_code, today_str, _est_raw, today_nav["change"])
        # v5.23: 次日补录——缓存日期非今日但NAV历史中有该日净值，补录偏差
        if _cached and _cached["date"] != today_str and _cached["est"] is not None:
            _cache_date = _cached["date"]
            _cache_nav = next(
                (h for h in history if h["date"] == _cache_date and h.get("change") is not None),
                None
            )
            if _cache_nav:
                record_deviation(fund_code, _cache_date, _cached["est"], _cache_nav["change"])
        elif result["recent_changes"]:
            latest = result["recent_changes"][0]
            if latest["change"] is not None:
                result["estimation_change"] = round(latest["change"], 4)
                result["_source"] = "nav"
                result["_nav_date"] = latest["date"]
                result["notes"].append(f"收盘后使用真实净值 {latest['date']}")
    else:
        # v5.22: 盘中缓存ETF估值
        _intraday_estimation_cache[fund_code] = {
            "date": today_str,
            "est": pct
        }

    return result


def _try_fundgz_fallback(result: dict, fund_code: str) -> dict:
    """当持仓穿透失败时，尝试用天天基金 fundgz 盘中估值接口。
    适用于QDII、指数基金等天天基金有官方估值但无持仓数据的基金。
    返回填充后的 result（原地修改），estimation_change 仍为 None 表示不可用。"""
    from .providers import get_fundgz_estimation, get_fund_nav_history

    gz = get_fundgz_estimation(fund_code)
    if not gz:
        return result

    try:
        gszzl = float(gz["gszzl"])
    except (ValueError, TypeError, KeyError):
        return result

    gztime = gz.get("gztime", "")
    fund_name = gz.get("name", "")

    # 检查估值是否过期：如果估值日期不是今天也不是昨天（QDII凌晨更新），可能是陈旧数据
    # QDII基金的gztime可能是凌晨（如 "2026-02-28 05:00"），属于正常
    if gztime:
        try:
            gz_date = datetime.strptime(gztime[:10], "%Y-%m-%d")
            days_old = (datetime.now() - gz_date).days
            if days_old > 3:
                # 超过3天的估值太旧，不使用
                return result
        except Exception:
            pass

    result["estimation_change"] = round(gszzl, 4)
    result["asof_time"] = gztime
    result["confidence"] = 0.6  # fundgz 可靠度中等（天天基金官方估值）
    result["_source"] = "fundgz"
    result["fund_name"] = fund_name or result.get("fund_name")
    result["notes"].append(f"天天基金估值: {gszzl:+.2f}% ({gztime})")

    # 附带近几日真实涨跌幅
    history = get_fund_nav_history(fund_code, 5)
    today_str = datetime.now().strftime("%Y-%m-%d")
    result["recent_changes"] = [
        {"date": h["date"], "change": h["change"]}
        for h in history if h.get("change") is not None
    ][:5]

    # 收盘后优先用真实净值
    if _is_market_closed():
        # v5.22: fundgz路径也从盘中缓存取估值
        _cached = _intraday_estimation_cache.get(fund_code)
        _est_raw = (
            _cached["est"]
            if _cached and _cached["date"] == today_str and _cached["est"] is not None
            else gszzl  # fallback: 用当前fundgz估值
        )
        today_nav = next(
            (h for h in history if h["date"] == today_str and h.get("change") is not None),
            None
        )
        if today_nav:
            result["estimation_change"] = round(today_nav["change"], 4)
            result["_source"] = "nav"
            result["_nav_date"] = today_str
            result["notes"].append(f"收盘后使用真实净值 {today_str}")
            # v5.22: fundgz路径也记录偏差（0%也是有效值）
            if _est_raw is not None:
                result["_estimation_change_raw"] = _est_raw
                record_deviation(fund_code, today_str, _est_raw, today_nav["change"])
        # v5.23: 次日补录——缓存日期非今日但NAV历史中有该日净值，补录偏差
        if _cached and _cached["date"] != today_str and _cached["est"] is not None:
            _cache_date = _cached["date"]
            _cache_nav = next(
                (h for h in history if h["date"] == _cache_date and h.get("change") is not None),
                None
            )
            if _cache_nav:
                record_deviation(fund_code, _cache_date, _cached["est"], _cache_nav["change"])
    else:
        # v5.22: 盘中缓存fundgz估值
        _intraday_estimation_cache[fund_code] = {
            "date": today_str,
            "est": gszzl
        }

    return result


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

    # 0. 商品类ETF联接基金（黄金/原油等）：跳过持仓穿透，直接用目标ETF实时行情
    etf_result = _try_etf_realtime_fallback(result, fund_code)
    if etf_result.get("estimation_change") is not None:
        return etf_result

    # 1. 获取持仓
    holdings = get_holdings(fund_code)

    if holdings.get("error"):
        result["notes"].append(f"持仓获取失败: {holdings['error']}")
        # 尝试 fundgz 估值 → 再降级到纯净值
        result = _try_fundgz_fallback(result, fund_code)
        if result.get("estimation_change") is not None:
            return result
        return _try_nav_fallback(result, fund_code)

    if not holdings.get("positions"):
        result["notes"].append("无持仓数据")
        # 尝试 fundgz 估值 → 再降级到纯净值
        result = _try_fundgz_fallback(result, fund_code)
        if result.get("estimation_change") is not None:
            return result
        return _try_nav_fallback(result, fund_code)

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
        result = _try_fundgz_fallback(result, fund_code)
        if result.get("estimation_change") is not None:
            return result
        return _try_nav_fallback(result, fund_code)

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
    from .providers import get_fund_nav_history
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
    if _is_market_closed():
        # v5.22: intraday_cache 已由 batch 入口的 _ensure_intraday_cache_loaded() 统一加载
        # v5.20: 收盘后优先从盘中缓存取估值（比当前重算的更准确）
        _cached = _intraday_estimation_cache.get(fund_code)
        _est_raw = (
            _cached["est"]
            if _cached and _cached["date"] == today_str and _cached["est"] is not None
            else result["estimation_change"]  # fallback: 用当前重算的值
        )

        # 优先查找今天的真实净值（收盘后基金公司已公布当日净值）
        today_nav_entry = next(
            (h for h in history if h["date"] == today_str and h.get("change") is not None),
            None
        )
        if today_nav_entry is not None:
            result["estimation_change"] = round(today_nav_entry["change"], 4)
            result["_source"] = "nav"
            result["_nav_date"] = today_str
            result["notes"].append(f"使用真实净值涨跌 {today_str}")
            # v5.22: 记录偏差（仅当盘中有过有效估值时，0%也是有效值）
            if _est_raw is not None:
                result["_estimation_change_raw"] = _est_raw
                record_deviation(fund_code, today_str, _est_raw, today_nav_entry["change"])
        # v5.23: 次日补录——缓存日期非今日但NAV历史中有该日净值，补录偏差
        if _cached and _cached["date"] != today_str and _cached["est"] is not None:
            _cache_date = _cached["date"]
            _cache_nav = next(
                (h for h in history if h["date"] == _cache_date and h.get("change") is not None),
                None
            )
            if _cache_nav:
                record_deviation(fund_code, _cache_date, _cached["est"], _cache_nav["change"])
        if today_nav_entry is None and result["recent_changes"]:
            # 今天的净值尚未公布，退回到最近一个交易日的真实净值
            latest = result["recent_changes"][0]
            if latest["change"] is not None:
                result["estimation_change"] = round(latest["change"], 4)
                result["_source"] = "nav"
                result["_nav_date"] = latest["date"]
                result["notes"].append(f"使用真实净值涨跌 {latest['date']}")
            else:
                result["_source"] = "estimation"
        elif today_nav_entry is None:
            result["_source"] = "estimation"
    else:
        result["_source"] = "estimation"
        # v5.20: 盘中缓存当前估值，供收盘后偏差记录使用
        # v5.22: 仅写内存缓存，文件持久化由 calculate_valuation_batch 统一完成
        if result["estimation_change"] is not None:
            _intraday_estimation_cache[fund_code] = {
                "date": today_str,
                "est": result["estimation_change"]
            }

    # v5.20: 附带校准后的置信度（供前端和 engine 统一使用）
    result["calibrated_confidence"] = calibrate_confidence(fund_code, result["confidence"])

    return result


def _is_market_closed() -> bool:
    """判断当天是否已收盘（或非交易日）
    True = 可以安全用真实净值替代估值
    盘前/盘中/午休 都返回 False，只有15:05后和非交易日返回 True
    节假日检测：最近一个有净值的交易日距今>3天则视为长假期间
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
    # 工作日盘中时段，额外检查是否为法定节假日
    # 正常情况：周一最近净值=周五(隔2天)，周二~五=前一天(隔1天)
    # 若最近净值距今>=4天，说明中间有连续非交易日（法定假期）
    try:
        from .providers import get_fund_nav_history
        hist = get_fund_nav_history("000300", 3)  # 沪深300
        if hist:
            latest_date = datetime.strptime(hist[0]["date"], "%Y-%m-%d")
            gap = (now - latest_date).days
            # 周一允许gap=3(周五→周一)，其余工作日gap>=4必为假期
            max_normal_gap = 3 if weekday == 0 else 2
            if gap > max_normal_gap:
                return True
    except Exception:
        pass
    return False  # 盘中或午休，继续用估值

def calculate_valuation_batch(fund_codes: List[str]) -> List[dict]:
    from concurrent.futures import ThreadPoolExecutor
    from .providers import get_fund_5day_change, get_fund_nav_history

    def _calc_20day_change(code):
        """用近20+1日净值计算20日复利涨跌幅"""
        try:
            hist = get_fund_nav_history(code, 22)
            changes = [h["change"] for h in hist if h.get("change") is not None][:20]
            if len(changes) < 5:
                return None
            product = 1.0
            for c in changes:
                product *= (1 + c / 100)
            return round((product - 1) * 100, 2)
        except Exception:
            return None

    # v5.22: 在并发计算之前统一加载盘中缓存（进程重启恢复场景）
    _ensure_intraday_cache_loaded()

    # 1. 并发计算估值
    results_map = {}
    week_data = {}
    month_data = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        # 估值并发
        val_futures = {pool.submit(calculate_valuation, code): code for code in fund_codes}
        # 5日涨幅并发
        week_futures = {pool.submit(get_fund_5day_change, code): code for code in fund_codes}
        # 20日涨幅并发
        month_futures = {pool.submit(_calc_20day_change, code): code for code in fund_codes}

        for future in val_futures:
            code = val_futures[future]
            try:
                results_map[code] = future.result(timeout=30)
            except:
                results_map[code] = {"fund_code": code, "error": "估值计算超时"}

        for future in week_futures:
            code = week_futures[future]
            try:
                week_data[code] = future.result(timeout=20)
            except:
                week_data[code] = None

        for future in month_futures:
            code = month_futures[future]
            try:
                month_data[code] = future.result(timeout=20)
            except:
                month_data[code] = None

    # 2. 按原始顺序组装结果并合并
    results = [results_map.get(code, {"fund_code": code}) for code in fund_codes]
    for r in results:
        r["week_change"] = week_data.get(r["fund_code"])
        r["month_change"] = month_data.get(r["fund_code"])

    # v5.22: 所有并发估值计算完成后，一次性将缓冲区偏差写入文件
    # 避免并发逐只写文件导致 load 到不完整数据覆盖历史
    flush_deviations()

    # v5.22: 盘中缓存也在并发结束后统一持久化一次（而非每只基金写一次）
    if _intraday_estimation_cache:
        try:
            _ensure_data_dir()
            tmp = _INTRADAY_CACHE_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(_intraday_estimation_cache, f, ensure_ascii=False, indent=2)
            tmp.replace(_INTRADAY_CACHE_FILE)
        except Exception:
            pass

    return results

def calculate_valuation_by_state() -> dict:
    # v5.22: 统一加载盘中缓存
    _ensure_intraday_cache_loaded()
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

    # v5.22: 非batch路径也统一flush偏差
    flush_deviations()

    return result


if __name__ == "__main__":
    print("=== 测试单基金估值 ===")
    v = calculate_valuation("017193")
    print(json.dumps(v, ensure_ascii=False, indent=2))