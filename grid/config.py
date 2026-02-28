"""
grid/config.py - 低频网格策略：配置、常量、状态管理

包含: 信号历史 | 波动率敏感度 | 行情模式(v5.17) | 适配性评分 | 全部策略常量
"""
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from positions import load_positions, save_positions, parse_fund_key
from valuation.core import calculate_valuation
from valuation.providers import get_fund_nav_history

# ============================================================
# 信号历史记录（持久化到 data/signal_history.json）
# ============================================================

DATA_DIR = Path(__file__).parent.parent / "data"
HISTORY_FILE = DATA_DIR / "signal_history.json"
_hist_lock = threading.Lock()
MAX_HISTORY_PER_FUND = 90


def _load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_history(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _hist_lock:
        try:
            tmp = HISTORY_FILE.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp.replace(HISTORY_FILE)
        except Exception as e:
            print(f"[Strategy] 保存信号历史失败: {e}")


def _append_signal_history(fund_code: str, signal: dict, market: dict):
    """追加一条信号记录，同一天同一来源覆盖"""
    history = _load_history()
    records = history.setdefault(fund_code, [])

    today_str = datetime.now().strftime("%Y-%m-%d")
    source = signal.get("_source") or market.get("_source") or "estimation"
    entry = {
        "date": today_str,
        "time": datetime.now().strftime("%H:%M:%S"),
        "source": source,
        "signal_name": signal.get("signal_name"),
        "action": signal.get("action"),
        "priority": signal.get("priority"),
        "reason": signal.get("reason"),
        "amount": signal.get("amount"),
        "sell_pct": signal.get("sell_pct"),
        "today_change": market.get("today_change"),
        "total_profit_pct": market.get("total_profit_pct"),
        "current_nav": market.get("current_nav"),
        "nav_at_signal": market.get("current_nav"),
        "outcome_t3": None,
        "outcome_t5": None,
        "outcome_t10": None,
    }

    records = [r for r in records
               if not (r.get("date") == today_str and r.get("source", "estimation") == source)]
    records.append(entry)

    if len(records) > MAX_HISTORY_PER_FUND:
        records = records[-MAX_HISTORY_PER_FUND:]

    history[fund_code] = records
    _save_history(history)


def backfill_signal_outcomes():
    """回填历史信号的 outcome 字段。建议收盘后调用一次。"""
    history = _load_history()
    updated = False

    for fund_code, records in history.items():
        nav_hist = get_fund_nav_history(fund_code, 30)
        nav_by_date = {h["date"]: h["nav"] for h in nav_hist if h.get("nav")}
        trade_dates = sorted(nav_by_date.keys())

        for rec in records:
            if rec.get("nav_at_signal") is None:
                continue
            sig_date = rec["date"]
            nav_at = rec["nav_at_signal"]
            if nav_at <= 0:
                continue

            for offset, field in [(3, "outcome_t3"), (5, "outcome_t5"), (10, "outcome_t10")]:
                if rec.get(field) is not None:
                    continue
                future_dates = [d for d in trade_dates if d > sig_date]
                if len(future_dates) >= offset:
                    target_date = future_dates[offset - 1]
                    target_nav = nav_by_date.get(target_date)
                    if target_nav:
                        rec[field] = round((target_nav / nav_at - 1) * 100, 2)
                        updated = True

    if updated:
        _save_history(history)
    return updated


def get_signal_history(fund_code: str = None, limit: int = 30) -> dict:
    history = _load_history()
    if fund_code:
        records = history.get(fund_code, [])
        return {fund_code: records[-limit:]}
    return {code: recs[-limit:] for code, recs in history.items()}


def calc_signal_win_rate(fund_code: str = None, lookback: int = 30) -> dict:
    """计算信号胜率统计"""
    history = _load_history()
    codes = [fund_code] if fund_code else list(history.keys())

    buy_outcomes = []
    sell_outcomes = []

    for code in codes:
        for rec in history.get(code, [])[-lookback:]:
            if rec.get("outcome_t5") is None:
                continue
            if rec.get("action") == "buy":
                buy_outcomes.append(rec["outcome_t5"])
            elif rec.get("action") == "sell":
                sell_outcomes.append(rec["outcome_t5"])

    buy_win_rate = (sum(1 for o in buy_outcomes if o > 0) / len(buy_outcomes)
                    if buy_outcomes else None)
    sell_accuracy = (sum(1 for o in sell_outcomes if o < 0) / len(sell_outcomes)
                     if sell_outcomes else None)
    avg_buy_t5 = (sum(buy_outcomes) / len(buy_outcomes) if buy_outcomes else None)

    return {
        "buy_win_rate": round(buy_win_rate, 3) if buy_win_rate is not None else None,
        "sell_accuracy": round(sell_accuracy, 3) if sell_accuracy is not None else None,
        "avg_buy_outcome_t5": round(avg_buy_t5, 2) if avg_buy_t5 is not None else None,
        "buy_sample_count": len(buy_outcomes),
        "sell_sample_count": len(sell_outcomes),
    }


# ============================================================
# v5.3: 波动率灵敏度自动校准（修复空壳实现）
# ============================================================

DEFAULT_VOL_SENSITIVITY = 1.0
# 自动校准结果缓存有效期（秒），避免每次信号生成都重新计算
_VOL_SENS_CACHE_TTL = 3600 * 6  # 6小时


def _get_vol_sensitivity(fund_code: str) -> tuple:
    """
    获取基金的波动率灵敏度系数。优先级：
    1. positions.json 中用户手动配置的 vol_sensitivity（用户显式覆盖）
    2. positions.json 中缓存的自动校准值 vol_sensitivity_auto（带时间戳）
    3. 实时计算自动校准值并缓存
    4. 默认 1.0

    返回 (sensitivity_value, source_str) 元组，source_str ∈ {"manual", "auto", "default"}
    """
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return DEFAULT_VOL_SENSITIVITY, "default"

    # 1. 用户手动配置（最高优先级）
    if fund.get("vol_sensitivity") is not None:
        return max(0.5, min(1.5, fund["vol_sensitivity"])), "manual"

    # 2. 检查缓存的自动校准值是否还在有效期内
    cached = fund.get("vol_sensitivity_auto")
    cached_at = fund.get("vol_sensitivity_auto_at")
    if cached is not None and cached_at:
        try:
            ts = datetime.strptime(cached_at, "%Y-%m-%d %H:%M:%S")
            if (datetime.now() - ts).total_seconds() < _VOL_SENS_CACHE_TTL:
                return max(0.5, min(1.5, cached)), "auto"
        except (ValueError, TypeError):
            pass

    # 3. 实时计算并缓存
    calibrated = auto_calibrate_vol_sensitivity(fund_code)
    if calibrated is not None:
        fund["vol_sensitivity_auto"] = calibrated
        fund["vol_sensitivity_auto_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        save_positions(data)
        return max(0.5, min(1.5, calibrated)), "auto"

    return DEFAULT_VOL_SENSITIVITY, "default"


def auto_calibrate_vol_sensitivity(fund_code: str) -> Optional[float]:
    """
    自动校准波动率灵敏度。

    由于没有盘中估值历史数据，无法直接比较"估值波动率 vs 真实波动率"。
    改用以下可观测指标的综合判断：

    1. 尾部厚度因子（tail_ratio = stddev / MAD×1.4826）
       - 正态分布 ≈ 1.0
       - 厚尾（极端波动频繁）> 1.2 → 阈值应更宽 → sensitivity ↑
       - 轻尾（波动集中在均值附近）< 0.8 → 阈值可收紧 → sensitivity ↓

    2. 持仓覆盖率修正（从估值结果推断）
       - 覆盖率低的基金，盘中估值可能系统性偏离真实净值
       - 覆盖率 < 70% → sensitivity × 1.1（估值不可靠，放宽阈值）

    3. 近期波动率变化率（regime detection）
       - 近5日波动率 / 近20日波动率
       - > 1.5 = 波动率放大期 → sensitivity × 1.05
       - < 0.6 = 波动率收缩期 → sensitivity × 0.95

    返回建议的 sensitivity 值（0.7~1.3），None 表示数据不足
    """
    real_code, _ = parse_fund_key(fund_code)
    nav_hist = get_fund_nav_history(real_code, 30)

    if len(nav_hist) < 15:
        return None

    real_changes = [h["change"] for h in nav_hist if h.get("change") is not None]
    if len(real_changes) < 10:
        return None

    mean_real = sum(real_changes) / len(real_changes)
    var_real = sum((c - mean_real) ** 2 for c in real_changes) / len(real_changes)
    vol_real = var_real ** 0.5

    if vol_real < 0.1:
        return 1.0  # 极低波动（货币/纯债）

    # --- 因子1: 尾部厚度 ---
    sorted_changes = sorted(real_changes)
    n = len(sorted_changes)
    median = sorted_changes[n // 2] if n % 2 else (sorted_changes[n // 2 - 1] + sorted_changes[n // 2]) / 2
    abs_devs = sorted([abs(c - median) for c in real_changes])
    m = len(abs_devs)
    mad = abs_devs[m // 2] if m % 2 else (abs_devs[m // 2 - 1] + abs_devs[m // 2]) / 2
    vol_robust = mad * 1.4826

    if vol_robust > 0:
        tail_ratio = vol_real / vol_robust
        tail_factor = max(0.8, min(1.2, tail_ratio))
    else:
        tail_factor = 1.0

    # --- 因子2: 波动率变化率（regime detection）---
    regime_factor = 1.0
    if len(real_changes) >= 10:
        vol_5d = (sum((c - sum(real_changes[:5]) / 5) ** 2 for c in real_changes[:5]) / 5) ** 0.5
        vol_all = vol_real
        if vol_all > 0:
            regime_ratio = vol_5d / vol_all
            if regime_ratio > 1.5:
                regime_factor = 1.05  # 波动放大期
            elif regime_ratio < 0.6:
                regime_factor = 0.95  # 波动收缩期

    # --- 因子3: 持仓覆盖率修正（从估值置信度推断）---
    coverage_factor = 1.0
    try:
        from valuation.core import calculate_valuation
        val = calculate_valuation(real_code)
        coverage = val.get("coverage", {})
        stock_total = coverage.get("stock_total_weight", 0)
        covered = coverage.get("covered_weight", 0)
        if stock_total > 0:
            cov_ratio = covered / stock_total
            if cov_ratio < 0.5:
                coverage_factor = 1.15
            elif cov_ratio < 0.7:
                coverage_factor = 1.10  # 覆盖率低→估值不可靠→放宽
    except Exception:
        pass

    # --- 综合 ---
    sensitivity = round(tail_factor * regime_factor * coverage_factor, 2)
    sensitivity = max(0.7, min(1.3, sensitivity))

    return sensitivity


def update_vol_sensitivity(fund_code: str, sensitivity: float) -> bool:
    """用户手动设置波动率灵敏度（覆盖自动校准值）"""
    sensitivity = max(0.5, min(1.5, sensitivity))
    data = load_positions()
    funds = data.setdefault("funds", {})
    if fund_code not in funds:
        return False
    funds[fund_code]["vol_sensitivity"] = round(sensitivity, 2)
    save_positions(data)
    return True


def clear_vol_sensitivity(fund_code: str) -> bool:
    """清除手动设置和自动缓存，下次信号生成时重新校准"""
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return False
    fund.pop("vol_sensitivity", None)
    fund.pop("vol_sensitivity_auto", None)
    fund.pop("vol_sensitivity_auto_at", None)
    save_positions(data)
    return True


def get_vol_sensitivity_info(fund_code: str) -> dict:
    """获取灵敏度完整信息（供API返回）"""
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code, {})

    manual = fund.get("vol_sensitivity")
    auto_cached = fund.get("vol_sensitivity_auto")
    auto_at = fund.get("vol_sensitivity_auto_at")

    # 当前生效值
    effective, source = _get_vol_sensitivity(fund_code)

    return {
        "fund_code": fund_code,
        "effective": effective,
        "source": source,
        "manual": manual,
        "auto_cached": auto_cached,
        "auto_cached_at": auto_at,
        "default": DEFAULT_VOL_SENSITIVITY,
        "range": {"min": 0.5, "max": 1.5},
    }


# ============================================================
# 核心阈值常量
# ============================================================

# ============================================================
# v5.13: 行情模式切换系统（market regime）
# ============================================================
# 三种模式：bull(牛市), neutral(震荡/默认), bear(熊市)
# 通过参数覆盖影响策略行为，不修改核心逻辑代码

# 行情模式参数定义
REGIME_PARAMS = {
    # ── v5.16: 铁律驱动的行情模式参数 ─────────────────────────────
    #
    # v5.14/v5.15/v5.13.3/v5.13.4 四轮失败验证的三条铁律:
    #   铁律1: 止损参数不可调 → L2↔L3水床效应完美对冲
    #   铁律2: 止盈抑制不可一刀切 → 扭亏交易受伤最重
    #   铁律3: 仓位增加边际递减 → +2pp仓位仅+0.3pp年化
    #
    # v5.16 设计原则:
    #   ✅ 只调买入端(入场时机/回补节奏)
    #   ❌ 不碰止盈参数(tp_suppress锁定65)
    #   ❌ 不碰止损参数(l2_stop_loss锁定35)
    #   ❌ 不碰补仓上限(supplement_max锁定3)
    #   ❌ 不碰仓位上限(size_mul_cap锁定1.40)
    #
    # v5.13.4后验: 纯熊市组10只全部正向 → 熊市买入端收紧有效
    #              牛市组全面负向 → 牛市参数违反铁律1/2/3
    # ──────────────────────────────────────────────────────────────
    "bull": {  # v5.17: =基线(铁律3验证: 牛市组39只净亏-1.4%, 取消)
        "first_build_ratio": 0.70,
        "size_mul_cap": 1.40,
        "tp_suppress_threshold": 65,
        "supplement_max_count": 3,
        "l2_stop_loss_base": 35,
        "rebuy_discount": 0.015,
    },
    "neutral": {                         # v5.13 帕累托最优基线，全部锁定
        "first_build_ratio": 0.70,
        "size_mul_cap": 1.40,
        "tp_suppress_threshold": 65,
        "supplement_max_count": 3,
        "l2_stop_loss_base": 35,
        "rebuy_discount": 0.015,
    },
    "bear": {
        "first_build_ratio": 0.55,       # 买入端: 熊市保守建仓
        "size_mul_cap": 1.40,            # 锁定基线(铁律3)
        "tp_suppress_threshold": 65,     # 锁定基线(铁律2)
        "supplement_max_count": 3,       # 锁定基线(v5.13.3教训: 砍到2→扭亏-66k)
        "l2_stop_loss_base": 35,         # 锁定基线(铁律1)
        "rebuy_discount": 0.030,         # 买入端: 熊市要求更大安全边际才回补
    },
}

# 自动识别结果缓存有效期（秒）
_REGIME_CACHE_TTL = 3600 * 6  # 6小时

def _get_market_regime() -> str:
    """
    读取positions.json的行情模式设置。
    返回 "bull" / "neutral" / "bear"
    """
    try:
        data = load_positions()
        regime = data.get("market_regime", "neutral")
        if regime in ("bull", "neutral", "bear"):
            return regime
    except Exception:
        pass
    return "neutral"


def _auto_detect_regime(trend_ctx: dict) -> str:
    """
    基于trend_ctx自动识别行情模式。

    v5.16: 铁律驱动的极端行情检测
    ─────────────────────────────────────────────────
    v5.13.3: OR连接 long>5%/-3% → 33/33/33% → 净亏-36k
    v5.13.4: AND确认 long>12%/-8% → 8/88/4%  → 净亏-14k
    v5.16:   AND确认 long>15%/-10% → 目标≤5/≥90/≤5%
             + 只调买入端参数 → 预期净正向

    关键洞察: v5.13.4中纯熊市组10只全部正向(+0.01%~+2.08%)
    但牛市组因违反铁律2(tp_suppress=67)导致净亏-13k
    v5.16锁定止盈/止损=基线 → 牛熊模式只影响入场节奏
    → 即使误判也不会伤害核心盈利能力
    """
    long_20d = trend_ctx.get("long_20d")
    trend_label = trend_ctx.get("trend_label", "震荡")

    # v5.17: 只检测熊市（牛市参数=基线，检测无意义）
    # 熊市: 20日跌<-10% 且 趋势标签确认（极端弱势才触发）
    if (long_20d is not None and long_20d < -10
            and trend_label in ("连跌", "中期走弱")):
        return "bear"

    return "neutral"


def _resolve_regime(trend_ctx: dict = None) -> str:
    """
    解析当前生效的行情模式。
    优先级: 手动指认(regime_manual) > 自动识别(默认开启) > 默认neutral
    """
    try:
        data = load_positions()
    except Exception:
        return "neutral"

    # 1. 手动指认优先级最高
    regime_manual = data.get("regime_manual")
    if regime_manual in ("bull", "neutral", "bear"):
        return regime_manual

    # 2. 自动识别（默认开启）
    regime_auto = data.get("regime_auto", True)

    if regime_auto and trend_ctx is not None:
        # 检查缓存
        cached_regime = data.get("regime_auto_result")
        cached_at = data.get("regime_auto_at")
        if cached_regime and cached_at:
            try:
                ts = datetime.strptime(cached_at, "%Y-%m-%d %H:%M:%S")
                if (datetime.now() - ts).total_seconds() < _REGIME_CACHE_TTL:
                    return cached_regime
            except (ValueError, TypeError):
                pass

        # 自动识别并缓存
        detected = _auto_detect_regime(trend_ctx)
        data["regime_auto_result"] = detected
        data["regime_auto_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data["market_regime"] = detected
        save_positions(data)
        return detected

    return data.get("market_regime", "neutral")


def _get_regime_params(regime: str) -> dict:
    """获取指定行情模式的参数字典"""
    return REGIME_PARAMS.get(regime, REGIME_PARAMS["neutral"]).copy()


def set_market_regime(regime: str, auto: bool = True, manual: bool = False) -> dict:
    """
    设置行情模式（供API调用）。
    参数:
        regime: "neutral" / "bear"（v5.17: bull已禁用，自动映射为neutral）
        auto: 是否开启自动识别（默认开启）
        manual: 是否为手动指认（手动优先级高于自动）
    """
    if regime == "bull" or regime not in ("neutral", "bear"):
        regime = "neutral"
    data = load_positions()

    if manual:
        # 手动指认: 设置 regime_manual，覆盖自动识别
        data["regime_manual"] = regime
        data["market_regime"] = regime
    else:
        # 切换回自动: 清除手动覆盖
        data.pop("regime_manual", None)
        data["regime_auto"] = auto
        data["market_regime"] = regime
        if auto:
            data.pop("regime_auto_result", None)
            data.pop("regime_auto_at", None)

    save_positions(data)
    return {
        "regime": regime,
        "auto": data.get("regime_auto", True),
        "manual": data.get("regime_manual") is not None,
        "manual_regime": data.get("regime_manual"),
        "params": _get_regime_params(regime),
    }


def get_market_regime_info() -> dict:
    """获取当前行情模式完整信息（供API返回）"""
    data = load_positions()
    regime = data.get("market_regime", "neutral")
    auto = data.get("regime_auto", True)
    auto_result = data.get("regime_auto_result")
    auto_at = data.get("regime_auto_at")
    manual_regime = data.get("regime_manual")
    effective = manual_regime if manual_regime in ("neutral", "bear") else (auto_result or regime)
    if effective == "bull":  # v5.17: bull已禁用
        effective = "neutral"
    return {
        "regime": effective,
        "auto": auto,
        "auto_result": auto_result,
        "auto_at": auto_at,
        "manual": manual_regime is not None,
        "manual_regime": manual_regime,
        "params": _get_regime_params(effective),
    }


# ============================================================
# v5.13: 适配评分缓存（回测生成，前端展示）
# ============================================================
FITNESS_CACHE_FILE = DATA_DIR / "fitness_cache.json"


def get_fitness_scores() -> dict:
    """读取适配评分缓存（由backtest.py写入）"""
    if not FITNESS_CACHE_FILE.exists():
        return {}
    try:
        with open(FITNESS_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_fund_fitness(fund_code: str) -> dict:
    """获取单只基金的适配评分"""
    real_code, _ = parse_fund_key(fund_code)
    cache = get_fitness_scores()
    return cache.get(fund_code) or cache.get(real_code) or {}


# --- 以波动率倍数表达的核心阈值 ---
DIP_BUY_VOL_MULTIPLE = 1.8
SUPPLEMENT_TRIGGER_VOL_MULTIPLE = 1.2
SUPPLEMENT_LOSS_VOL_MULTIPLE = 2.2
CONSECUTIVE_DIP_VOL_MULTIPLE = 0.7
STOP_LOSS_VOL_MULTIPLE = 6.0  # v5.5 优化: 大幅放宽止损线（原4.5→6.0），正常波动不应触发止损，回测显示36%交易是止损在失血
TAKE_PROFIT_VOL_MULTIPLE = 2.5  # v5.5 优化: 提高止盈门槛（原1.5→2.5），让利润奔跑，避免小止盈频繁触发
TREND_WEAK_VOL_MULTIPLE = 1.5
DISASTER_LOSS_VOL_MULTIPLE = 6.5  # v5.5 优化: 灾难线适度放宽（原5.0→6.5），与止损线保持合理间距
DISASTER_DAILY_VOL_MULTIPLE = 3.0

# --- 固定默认值（波动率数据不足时兜底）---
DEFAULT_DIP_THRESHOLD = -2.5
DEFAULT_TAKE_PROFIT_TRIGGER = 2.0
DEFAULT_STOP_LOSS_BASE = -7.0  # v5.5 优化: 默认止损线从-5.0→-7.0，给正常波动更多空间
DEFAULT_SUPPLEMENT_TRIGGER = -1.5
DEFAULT_SUPPLEMENT_LOSS_MIN = -3.0
DEFAULT_CONSECUTIVE_DIP_TRIGGER = -1.0
DEFAULT_TREND_WEAK_CUMULATIVE = -2.0
DEFAULT_DISASTER_LOSS = -12.0  # v5.5 优化: 灾难线从-9.0→-12.0
DEFAULT_DISASTER_DAILY_DROP = -6.0  # v5.5 优化: 从-5.0→-6.0

# --- 向后兼容别名 ---
TAKE_PROFIT_TRIGGER = DEFAULT_TAKE_PROFIT_TRIGGER
STOP_LOSS_BASE = DEFAULT_STOP_LOSS_BASE
SUPPLEMENT_TRIGGER = DEFAULT_SUPPLEMENT_TRIGGER
SUPPLEMENT_LOSS_MIN = DEFAULT_SUPPLEMENT_LOSS_MIN
CONSECUTIVE_DIP_TRIGGER = DEFAULT_CONSECUTIVE_DIP_TRIGGER
TREND_WEAK_CUMULATIVE = DEFAULT_TREND_WEAK_CUMULATIVE
DISASTER_LOSS_THRESHOLD = DEFAULT_DISASTER_LOSS
DISASTER_DAILY_DROP = DEFAULT_DISASTER_DAILY_DROP

COOLDOWN_DAYS = 0  # v5.5 优化: 冷却期从1天→0天，卖出后立即可重新入场，避免牛市中错过反弹
SUPPLEMENT_MAX_COUNT_DEFAULT = 3
SUPPLEMENT_MAX_COUNT_HARD_CAP = 5

# 补仓档位：(次数, 预算比例, 当日跌幅vol倍数, 浮亏vol倍数)
# v5.5 优化: 首次建仓60%+补仓比例上调，目标平均仓位从12%→35-45%
SUPPLEMENT_TIERS_VOL = [
    (0, 0.70, 1.0, 1.8),   # v5.13: 首次建仓70%（v5.11为60%），加快入场速度
    (1, 0.35, 1.4, 3.0),   # v5.5: 第一次补仓35%（原25%），门槛也降低
    (2, 0.25, 1.8, 4.5),   # v5.5: 第二次补仓25%（原20%）
    (3, 0.15, 2.2, 6.0),   # v5.5: 第三次补仓15%（原10%）
    (4, 0.10, 2.6, 7.5),   # v5.5: 第四次补仓10%（原5%）
]
# v5.5 优化: 同步更新固定补仓档位
SUPPLEMENT_TIERS = [
    (0, 0.70, -1.2, -2.5),   # v5.13: 首次建仓70%，加快入场速度
    (1, 0.35, -1.8, -4.0),
    (2, 0.25, -2.2, -6.5),
    (3, 0.15, -2.8, -9.0),
    (4, 0.10, -3.2, -11.0),
]

SUPPLEMENT_CAP_RATIO = 0.35  # v5.5 优化: 单次补仓上限从20%→35%，允许更大笔补仓快速建仓

# 扭亏止盈档位
# v5.5 优化: 扭亏止盈门槛大幅提高——原来赚0.3%就卖（平均只赚0.8元），现在至少2%才卖
TOTAL_PROFIT_SELL_TIERS_VOL = [
    (7.0, 50),   # v5.9 重构P1: 从5.0→7.0（v5.8扭亏止盈膨胀+27%至8681次，需大幅提高门槛）
    (5.0, 30),   # v5.9 重构P1: 从3.5→5.0，至少赚到5%才开始卖
]
TOTAL_PROFIT_SELL_TIERS = [
    (10.0, 50),  # v5.9 重构P1: 从7.0→10.0，固定版门槛同步大幅提高
    (7.0, 30),   # v5.9 重构P1: 从5.0→7.0
]

TREND_BUILD_TRIGGER_5D = -3.0
TREND_BUILD_TRIGGER_10D = -5.0

# v5.5 优化: 提高止盈门槛，让利润奔跑，避免赚3.5%就开始卖
TAKE_PROFIT_TIERS = [
    (12.0, 100),  # v5.5: 原8.0→12.0，只有大赚才清仓
    (8.0, 70),    # v5.5: 原5.0→8.0
    (5.0, 50),    # v5.5: 原3.5→5.0
]

# v5.5 优化: 慢涨止盈门槛也同步提高
SLOW_PROFIT_TIERS = [
    (12.0, 70),   # v5.5: 原8.0→12.0
    (8.0, 50),    # v5.5: 原5.0→8.0
    (6.0, 30),    # v5.5: 原4.0→6.0
]

DISASTER_CONSECUTIVE_DOWN = 3
DISASTER_SELL_PCT_EXTREME = 50
DISASTER_SELL_PCT_DAILY = 30

SUPPLEMENT_MIN_GAP_TRADE_DAYS = 2  # v5.8 重构P3: 补仓间隔从3天→2天，提高补仓执行率
SUPPLEMENT_REBUY_STEP_PCT = 1.0

# 回撤止盈
TRAIL_PROFIT_ACTIVATE = 5.0   # v5.5 优化: 激活线从3.5→5.0，低盈利时不启动回撤止盈
TRAIL_DD_BASE = 2.5           # v5.5 优化: 回撤容忍从1.8→2.5，允许更大回撤空间
TRAIL_DD_MIN = 1.8            # v5.5 优化: 最小回撤从1.2→1.8
TRAIL_DD_MAX = 5.0            # v5.5 优化: 最大回撤从4.0→5.0
TRAIL_PROFIT_SELL_TIERS = [
    (12.0, 70),   # v5.5: 原8.0→12.0
    (8.0, 50),    # v5.5: 原5.0→8.0
    (5.0, 30),    # v5.5: 原3.5→5.0
]

# FIFO穿透降级
PASSTHROUGH_LOSS_DOWNGRADE = -50.0
PASSTHROUGH_MIN_NET_PROFIT_RATIO = 0.002
PASSTHROUGH_MIN_NET_PROFIT_ABS = 30.0
PASSTHROUGH_LOSS_RATIO_THRESHOLD = 0.6

# 组合级
DAILY_BUY_CAP_RATIO_BASE = 0.20        # v5.5 优化: 日买入上限从10%→20%，让资金能真正入场
DAILY_BUY_CAP_RATIO_CONSERVATIVE = 0.12  # v5.5 优化: 保守模式从6%→12%
DAILY_BUY_CAP_RATIO_AGGRESSIVE = 0.30    # v5.5 优化: 激进模式从15%→30%

# 波动率状态机
VOL_LOW = 0.8
VOL_NORMAL_HIGH = 1.8
VOL_EXTREME = 3.0

# 止损分级
STOP_LOSS_L1_FACTOR = 0.7
STOP_LOSS_L2_SELL_PCT_BASE = 35  # v5.5 优化: L2止损基准从50%→35%，减少每次止损出血量（回测显示L2占11%交易量，平均亏6.5元）
STOP_LOSS_L3_FACTOR = 1.5
STOP_LOSS_L3_CONSEC_DOWN = 7  # v5.5 优化: 从5→7天，需要更长连跌才触发极端止损

# 同赛道约束
SECTOR_BUY_CAP_RATIO = 0.60  # v5.5 优化: 赛道集中度从40%→60%，减少对买入的限制

# 信号胜率自适应
WIN_RATE_TIGHTEN_THRESHOLD = 0.40
WIN_RATE_TIGHTEN_FACTOR = 1.10

# 流动性溢价
LIQUIDITY_PREMIUM_EXTRA_PCT = 15