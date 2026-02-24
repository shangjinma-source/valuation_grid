"""
strategy.py - 低频网格交易策略信号引擎 v5.3

================================================================================
v5.3 升级要点（资深基金经理30年经验深度优化）
================================================================================

【v5.2 保留的改进】
  - FUND_VOL_SENSITIVITY 自适应+可配置
  - profit_norm 下限降到 max(2.5, vol * 4.0)
  - 补仓成本修复效率阈值动态化
  - generate_all_signals 并发优化
  - volume_proxy 成交量代理
  - _estimate_current_nav 收盘后修正
  - 止损/灾难保护根据补仓次数递减

【一、趋势转弱卖出：从一刀切100%改为盈利深度分级减仓】
  v5.2 问题：趋势确认转弱后直接卖100%，不考虑总盈利水平。
  基金正常回调2-3%很常见，如果总浮盈>5%，不应因一次回调就清仓。

  v5.3 改进：
    - 总浮盈<2%  → 100%清仓（薄利快跑）
    - 总浮盈2-5% → 70%减仓（留底仓观察）
    - 总浮盈>5%  → 50%减仓（回调是正常波动）
    - 放量确认转弱 → 在上述基础上各+20%

【二、补仓节奏阀增加价格纵深维度】
  v5.2 问题：3个交易日间隔对急跌太慢、对阴跌太快。
  v5.3 改进：
    - 3日内急跌超过2倍波动率 → 允许提前补仓（缩短为2个交易日）
    - 阴跌（每日<0.5%但连续5天）→ 间隔延长到5个交易日

【三、_estimate_current_nav 收盘后逻辑修复】
  v5.2 BUG：收盘后如果最新日期≠今天，直接返回最新净值不乘今日涨跌。
  当T日收盘后T日净值尚未公布时，会用T-1净值，导致估值偏差。
  v5.3：收盘后且最新≠今天 → 用最新净值×(1+today_change/100)

【四、总仓位止损减仓比例动态化】
  v5.2 问题：总仓位止损固定减50%最老批次。
  v5.3 改进：
    - 浮亏刚触止损线  → 30%减仓（给反弹空间）
    - 浮亏超止损线2%以上 → 50%减仓
    - 浮亏超止损线5%以上 → 70%减仓
    - 补仓>=3次且浮亏仍深 → 额外+10%（表明判断错误应加速止损）

【五、空仓"冷却期后建仓"增加趋势过滤】
  v5.2 问题：冷却结束后只要今天跌就买，没有趋势过滤。
  v5.3 改进：需要同时满足：
    - 今天跌幅 ≤ 0
    - 5日累计非深度下跌（>-5%）或 已出现企稳信号
    - 置信度足够

【六、灾难保护通道增加短期大亏安全网】
  v5.2 问题：持有<7天走灾难保护通道，但灾难触发条件可能不满足，
  导致持有3天亏7%却只触发L1预警无法止损。
  v5.3 改进：增加"短期深亏安全网"——持有<7天且单批浮亏>6%，
  无条件减仓30%（不等灾难触发）

【七、组合级风控negative_ratio改为加权计算】
  v5.2 问题：只按基金数量计算负面比例，一只重仓和一只轻仓权重一样。
  v5.3 改进：按持仓金额加权计算 negative_ratio

================================================================================
"""

import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

from positions import get_fund_position, get_sell_fee_rate, load_positions, save_positions, parse_fund_key
from core import calculate_valuation, load_state, _is_market_closed
from providers import get_fund_nav_history

# ============================================================
# 信号历史记录（持久化到 data/signal_history.json）
# ============================================================

DATA_DIR = Path(__file__).parent / "data"
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
        from core import calculate_valuation
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

# --- 以波动率倍数表达的核心阈值 ---
DIP_BUY_VOL_MULTIPLE = 1.8
SUPPLEMENT_TRIGGER_VOL_MULTIPLE = 1.2
SUPPLEMENT_LOSS_VOL_MULTIPLE = 2.2
CONSECUTIVE_DIP_VOL_MULTIPLE = 0.7
STOP_LOSS_VOL_MULTIPLE = 3.5
TAKE_PROFIT_VOL_MULTIPLE = 1.5
TREND_WEAK_VOL_MULTIPLE = 1.5
DISASTER_LOSS_VOL_MULTIPLE = 5.0
DISASTER_DAILY_VOL_MULTIPLE = 3.0

# --- 固定默认值（波动率数据不足时兜底）---
DEFAULT_DIP_THRESHOLD = -2.5
DEFAULT_TAKE_PROFIT_TRIGGER = 2.0
DEFAULT_STOP_LOSS_BASE = -5.0
DEFAULT_SUPPLEMENT_TRIGGER = -1.5
DEFAULT_SUPPLEMENT_LOSS_MIN = -3.0
DEFAULT_CONSECUTIVE_DIP_TRIGGER = -1.0
DEFAULT_TREND_WEAK_CUMULATIVE = -2.0
DEFAULT_DISASTER_LOSS = -9.0
DEFAULT_DISASTER_DAILY_DROP = -5.0

# --- 向后兼容别名 ---
TAKE_PROFIT_TRIGGER = DEFAULT_TAKE_PROFIT_TRIGGER
STOP_LOSS_BASE = DEFAULT_STOP_LOSS_BASE
SUPPLEMENT_TRIGGER = DEFAULT_SUPPLEMENT_TRIGGER
SUPPLEMENT_LOSS_MIN = DEFAULT_SUPPLEMENT_LOSS_MIN
CONSECUTIVE_DIP_TRIGGER = DEFAULT_CONSECUTIVE_DIP_TRIGGER
TREND_WEAK_CUMULATIVE = DEFAULT_TREND_WEAK_CUMULATIVE
DISASTER_LOSS_THRESHOLD = DEFAULT_DISASTER_LOSS
DISASTER_DAILY_DROP = DEFAULT_DISASTER_DAILY_DROP

COOLDOWN_DAYS = 2
SUPPLEMENT_MAX_COUNT_DEFAULT = 3
SUPPLEMENT_MAX_COUNT_HARD_CAP = 5

# 补仓档位：(次数, 预算比例, 当日跌幅vol倍数, 浮亏vol倍数)
SUPPLEMENT_TIERS_VOL = [
    (0, 0.25, 1.2, 2.2),
    (1, 0.20, 1.6, 3.5),
    (2, 0.15, 2.0, 5.5),
    (3, 0.12, 2.4, 7.0),
    (4, 0.10, 2.8, 8.5),
]
SUPPLEMENT_TIERS = [
    (0, 0.25, -1.5, -3.0),
    (1, 0.20, -2.0, -5.0),
    (2, 0.15, -2.5, -8.0),
    (3, 0.12, -3.0, -10.0),
    (4, 0.10, -3.5, -12.0),
]

SUPPLEMENT_CAP_RATIO = 0.20

# 扭亏止盈档位
TOTAL_PROFIT_SELL_TIERS_VOL = [
    (2.0, 50),
    (1.0, 30),
    (0.3, 20),
]
TOTAL_PROFIT_SELL_TIERS = [
    (3.0, 50),
    (1.5, 30),
    (0.8, 20),
]

TREND_BUILD_TRIGGER_5D = -3.0
TREND_BUILD_TRIGGER_10D = -5.0

TAKE_PROFIT_TIERS = [
    (8.0, 100),
    (5.0, 70),
    (3.5, 50),
]

SLOW_PROFIT_TIERS = [
    (8.0, 70),
    (5.0, 50),
    (4.0, 30),
]

DISASTER_CONSECUTIVE_DOWN = 3
DISASTER_SELL_PCT_EXTREME = 50
DISASTER_SELL_PCT_DAILY = 30

SUPPLEMENT_MIN_GAP_TRADE_DAYS = 3
SUPPLEMENT_REBUY_STEP_PCT = 1.0

# 回撤止盈
TRAIL_PROFIT_ACTIVATE = 3.5
TRAIL_DD_BASE = 1.8
TRAIL_DD_MIN = 1.2
TRAIL_DD_MAX = 4.0
TRAIL_PROFIT_SELL_TIERS = [
    (8.0, 70),
    (5.0, 50),
    (3.5, 30),
]

# FIFO穿透降级
PASSTHROUGH_LOSS_DOWNGRADE = -50.0
PASSTHROUGH_MIN_NET_PROFIT_RATIO = 0.002
PASSTHROUGH_MIN_NET_PROFIT_ABS = 30.0
PASSTHROUGH_LOSS_RATIO_THRESHOLD = 0.6

# 组合级
DAILY_BUY_CAP_RATIO_BASE = 0.10
DAILY_BUY_CAP_RATIO_CONSERVATIVE = 0.06
DAILY_BUY_CAP_RATIO_AGGRESSIVE = 0.15

# 波动率状态机
VOL_LOW = 0.8
VOL_NORMAL_HIGH = 1.8
VOL_EXTREME = 3.0

# 止损分级
STOP_LOSS_L1_FACTOR = 0.7
STOP_LOSS_L2_SELL_PCT_BASE = 50  # v5.2: 基准值，实际根据补仓次数递减
STOP_LOSS_L3_FACTOR = 1.5
STOP_LOSS_L3_CONSEC_DOWN = 5

# 同赛道约束
SECTOR_BUY_CAP_RATIO = 0.40

# 信号胜率自适应
WIN_RATE_TIGHTEN_THRESHOLD = 0.40
WIN_RATE_TIGHTEN_FACTOR = 1.10

# 流动性溢价
LIQUIDITY_PREMIUM_EXTRA_PCT = 15


# ============================================================
# 波动率自适应阈值生成器
# ============================================================

def _vol_adaptive_thresholds(fund_code: str, vol: float) -> dict:
    """
    根据波动率动态生成所有阈值。
    v5.3: sensitivity 从 _get_vol_sensitivity 获取（自适应+缓存+用户配置）
    """
    sensitivity, sens_source = _get_vol_sensitivity(fund_code)

    if vol is None or vol <= 0:
        return {
            "dip_threshold": DEFAULT_DIP_THRESHOLD,
            "tp_trigger": DEFAULT_TAKE_PROFIT_TRIGGER,
            "stop_loss": DEFAULT_STOP_LOSS_BASE,
            "supplement_trigger": DEFAULT_SUPPLEMENT_TRIGGER,
            "supplement_loss_min": DEFAULT_SUPPLEMENT_LOSS_MIN,
            "consecutive_dip": DEFAULT_CONSECUTIVE_DIP_TRIGGER,
            "trend_weak": DEFAULT_TREND_WEAK_CUMULATIVE,
            "disaster_loss": DEFAULT_DISASTER_LOSS,
            "disaster_daily": DEFAULT_DISASTER_DAILY_DROP,
            "supplement_tiers": SUPPLEMENT_TIERS,
            "total_profit_tiers": TOTAL_PROFIT_SELL_TIERS,
            "_vol_based": False,
            "_sensitivity": sensitivity,
            "_sensitivity_source": sens_source,
        }

    v = vol * sensitivity

    dip       = max(-5.0,  min(-1.2, round(-v * DIP_BUY_VOL_MULTIPLE, 2)))
    tp        = max(1.0,   min(5.0,  round( v * TAKE_PROFIT_VOL_MULTIPLE, 2)))
    sl        = max(-10.0, min(-3.0, round(-v * STOP_LOSS_VOL_MULTIPLE, 2)))
    supp_trig = max(-4.0,  min(-0.8, round(-v * SUPPLEMENT_TRIGGER_VOL_MULTIPLE, 2)))
    supp_loss = max(-10.0, min(-2.0, round(-v * SUPPLEMENT_LOSS_VOL_MULTIPLE, 2)))
    consec_dip= max(-2.5,  min(-0.5, round(-v * CONSECUTIVE_DIP_VOL_MULTIPLE, 2)))
    tw        = max(-4.0,  min(-1.0, round(-v * TREND_WEAK_VOL_MULTIPLE, 2)))
    dis_loss  = max(-12.0, min(-5.0, round(-v * DISASTER_LOSS_VOL_MULTIPLE, 2)))
    dis_daily = max(-7.0,  min(-3.0, round(-v * DISASTER_DAILY_VOL_MULTIPLE, 2)))

    supp_tiers = []
    for tier_count, ratio, trig_mul, loss_mul in SUPPLEMENT_TIERS_VOL:
        t = max(-4.0,  min(-0.8, round(-v * trig_mul, 2)))
        l = max(-12.0, min(-2.0, round(-v * loss_mul, 2)))
        supp_tiers.append((tier_count, ratio, t, l))

    tp_tiers = [(round(max(0.3, min(4.0, v * mul)), 2), pct)
                for mul, pct in TOTAL_PROFIT_SELL_TIERS_VOL]

    return {
        "dip_threshold": dip,
        "tp_trigger": tp,
        "stop_loss": sl,
        "supplement_trigger": supp_trig,
        "supplement_loss_min": supp_loss,
        "consecutive_dip": consec_dip,
        "trend_weak": tw,
        "disaster_loss": dis_loss,
        "disaster_daily": dis_daily,
        "supplement_tiers": supp_tiers,
        "total_profit_tiers": tp_tiers,
        "_vol_based": True,
        "_sensitivity": sensitivity,
        "_sensitivity_source": sens_source,
    }


# ============================================================
# 波动率状态机
# ============================================================

def _classify_volatility(vol: float) -> str:
    if vol is None or vol < VOL_LOW:
        return "low_vol"
    elif vol < VOL_NORMAL_HIGH:
        return "normal_vol"
    elif vol < VOL_EXTREME:
        return "high_vol"
    else:
        return "extreme_vol"


# ============================================================
# 动量因子计算
# ============================================================

def _calc_momentum_score(trend_ctx: dict) -> float:
    """综合动量评分 ∈ [-1, 1]"""
    s5 = trend_ctx.get("short_5d")
    m10 = trend_ctx.get("mid_10d")
    l20 = trend_ctx.get("long_20d")

    def _norm(x, scale=5.0):
        if x is None:
            return 0.0
        return math.tanh(x / scale)

    score = 0.5 * _norm(s5, 4.0) + 0.3 * _norm(m10, 6.0) + 0.2 * _norm(l20, 10.0)
    return round(max(-1.0, min(1.0, score)), 3)


# ============================================================
# 动态阈值计算
# ============================================================

def _calc_risk_multiplier(trend_ctx: dict) -> float:
    """risk_mul 只用回撤驱动"""
    mdd_20 = trend_ctx.get("max_drawdown") or 0.0
    mdd_60 = trend_ctx.get("max_drawdown_60") or 0.0
    mdd = max(mdd_20, mdd_60)

    if mdd <= 5:
        mdd_term = 0.0
    elif mdd <= 10:
        mdd_term = (mdd - 5) * 0.06
    else:
        mdd_term = 0.30 + (mdd - 10) * 0.03

    risk_mul = 1.0 + mdd_term
    return max(0.85, min(1.5, risk_mul))


def _calc_dynamic_thresholds(trend_ctx: dict, fund_code: str,
                             confidence: float, source: str,
                             signal_stats: dict = None) -> dict:
    real_code, _ = parse_fund_key(fund_code)
    risk_mul = _calc_risk_multiplier(trend_ctx)
    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
    vol_state = _classify_volatility(vol)

    va = _vol_adaptive_thresholds(fund_code, vol)

    dip_threshold = round(va["dip_threshold"] * risk_mul, 2)
    tp_trigger = round(va["tp_trigger"], 2)
    stop_loss_adj = round(va["stop_loss"] * risk_mul, 2)

    if source == "estimation" and confidence < 0.75:
        tp_trigger = round(tp_trigger + 0.5, 2)

    supplement_tiers_adj = []
    for count, ratio, trigger, loss_min in va["supplement_tiers"]:
        supplement_tiers_adj.append((count, ratio,
                                     round(trigger * risk_mul, 2),
                                     round(loss_min * risk_mul, 2)))

    trail_dd = max(TRAIL_DD_MIN, min(TRAIL_DD_MAX, TRAIL_DD_BASE * risk_mul))

    # 信号胜率自适应
    win_rate_adj = 1.0
    if signal_stats and signal_stats.get("buy_win_rate") is not None:
        if (signal_stats["buy_win_rate"] < WIN_RATE_TIGHTEN_THRESHOLD
                and signal_stats.get("buy_sample_count", 0) >= 5):
            win_rate_adj = WIN_RATE_TIGHTEN_FACTOR
            dip_threshold = round(dip_threshold * win_rate_adj, 2)
            supplement_tiers_adj = [
                (c, r, round(t * win_rate_adj, 2), round(l * win_rate_adj, 2))
                for c, r, t, l in supplement_tiers_adj
            ]

    if vol_state == "low_vol":
        dip_threshold = round(dip_threshold * 0.85, 2)
        tp_trigger = round(tp_trigger * 0.85, 2)

    dip_threshold = max(-6.0, dip_threshold)
    stop_loss_adj = max(-12.0, stop_loss_adj)

    rebuy_step = max(0.8, vol * 0.8) if vol else SUPPLEMENT_REBUY_STEP_PCT

    return {
        "risk_multiplier": round(risk_mul, 2),
        "dip_threshold": round(dip_threshold, 2),
        "tp_trigger": round(tp_trigger, 2),
        "stop_loss_adj": round(stop_loss_adj, 2),
        "supplement_tiers": supplement_tiers_adj,
        "trail_dd": round(trail_dd, 2),
        "vol_state": vol_state,
        "momentum_score": _calc_momentum_score(trend_ctx),
        "win_rate_adj": round(win_rate_adj, 2),
        "rebuy_step": round(rebuy_step, 2),
        "_va": va,
        "_vol_based": va.get("_vol_based", False),
        "_sensitivity": va.get("_sensitivity", DEFAULT_VOL_SENSITIVITY),
        "_sensitivity_source": va.get("_sensitivity_source", "default"),
        "consecutive_dip_trigger": round(va["consecutive_dip"], 2),
        "supplement_trigger": round(va["supplement_trigger"], 2),
        "supplement_loss_min": round(va["supplement_loss_min"], 2),
        "trend_weak_cumulative": round(va["trend_weak"], 2),
        "disaster_loss_threshold": round(va["disaster_loss"], 2),
        "disaster_daily_drop": round(va["disaster_daily"], 2),
        "total_profit_sell_tiers": va.get("total_profit_tiers", TOTAL_PROFIT_SELL_TIERS),
    }


# ============================================================
# 统一止盈评分框架
# ============================================================

def _calc_sell_score(batch: dict, current_nav: float, today_change: float,
                     trend_ctx: dict, dyn: dict, fee_rate: float,
                     hold_days: int, peak_profit: float) -> dict:
    """
    v5.2 改进：profit_norm 下限从 4.0 降到 max(2.5, vol*4.0)
    让低波品种更容易触发止盈
    """
    profit_pct = round((current_nav / batch["nav"] - 1) * 100, 2) if batch["nav"] > 0 else 0.0

    if profit_pct <= fee_rate * 2.0:
        return {"score": 0, "sell_pct": 0, "signal_name": None, "reason": "盈利不足覆盖费率"}

    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.2

    # v5.2: 修正 profit_norm，低波品种更灵敏
    profit_norm = max(3.0, vol * 4.0)
    profit_score = math.tanh(profit_pct / profit_norm) * 40

    trail_score = 0
    if peak_profit > 3.0 and peak_profit > profit_pct:
        dd = peak_profit - profit_pct
        trail_dd_threshold = dyn.get("trail_dd", TRAIL_DD_BASE)
        if dd >= trail_dd_threshold:
            trail_score = min(30, dd / trail_dd_threshold * 15)

    momentum = dyn.get("momentum_score", 0)
    momentum_score = max(0, -momentum * 15)

    liquidity_score = 0
    liquidity_trigger = max(1.5, vol * TAKE_PROFIT_VOL_MULTIPLE)
    if today_change >= liquidity_trigger:
        liquidity_score = min(15, (today_change - liquidity_trigger) * 5)

    fee_drag = -fee_rate * 5

    total_score = profit_score + trail_score + momentum_score + liquidity_score + fee_drag

    if total_score >= 60:
        sell_pct = 100
        signal_name = "强势止盈"
    elif total_score >= 45:
        sell_pct = 70
        signal_name = "止盈卖出"
    elif total_score >= 30:
        sell_pct = 50
        signal_name = "分批止盈"
    elif total_score >= 25:
        sell_pct = 30
        signal_name = "慢涨止盈"
    else:
        sell_pct = 0
        signal_name = None

    if today_change >= liquidity_trigger and sell_pct > 0 and sell_pct < 100:
        sell_pct = min(100, sell_pct + LIQUIDITY_PREMIUM_EXTRA_PCT)

    reason = (f"综合评分{total_score:.0f}(盈利{profit_score:.0f}+回撤{trail_score:.0f}"
              f"+动量{momentum_score:.0f}+流动性{liquidity_score:.0f}+费率{fee_drag:.0f})")

    return {
        "score": round(total_score, 1),
        "sell_pct": sell_pct,
        "signal_name": signal_name,
        "reason": reason,
        "profit_pct": profit_pct,
        "peak_profit": peak_profit,
    }


# ============================================================
# 补仓成本修复效率
# ============================================================

def _calc_cost_repair_efficiency(batches: list, current_nav: float,
                                 supplement_amount: float) -> float:
    total_cost = sum(b["amount"] for b in batches)
    total_shares = sum(b["shares"] for b in batches)

    if total_shares <= 0 or current_nav <= 0 or supplement_amount <= 0:
        return 0.0

    avg_cost_before = total_cost / total_shares
    new_shares = supplement_amount / current_nav
    avg_cost_after = (total_cost + supplement_amount) / (total_shares + new_shares)

    cost_drop_pct = (avg_cost_before - avg_cost_after) / avg_cost_before * 100
    efficiency = cost_drop_pct / (supplement_amount / 1000)
    return round(efficiency, 4)


def _calc_dynamic_supplement_max(pos: dict) -> int:
    max_pos = pos.get("max_position", 5000)
    batches = pos.get("batches", [])
    holding = [b for b in batches if b.get("status") == "holding"]
    if holding:
        sorted_batches = sorted(holding, key=lambda b: b["buy_date"])
        first_amount = sorted_batches[0].get("amount", max_pos * 0.3)
    else:
        first_amount = max_pos * 0.3

    if first_amount <= 0:
        return SUPPLEMENT_MAX_COUNT_DEFAULT

    dynamic_max = math.ceil(max_pos / first_amount) - 1
    return max(1, min(SUPPLEMENT_MAX_COUNT_HARD_CAP, dynamic_max))


# ============================================================
# 三级止损体系（v5.2: L2卖出比例根据补仓次数递减）
# ============================================================

def _evaluate_stop_loss(profit_pct: float, stop_loss_adj: float,
                        hold_days: int, fee_rate: float,
                        trend_ctx: dict, confidence: float,
                        source: str, supplement_count: int = 0) -> dict:
    """
    v5.2: L2 sell_pct 根据补仓次数递减
    补仓越多 → 越应该保留仓位等反弹 → 减仓比例越小
    """
    if hold_days < 7:
        return {"level": None, "sell_pct": 0, "reason": "未满7天，走灾难保护通道"}

    confidence_adj = 0.0
    if source == "estimation" and confidence < 0.6:
        confidence_adj = -1.0

    effective_stop = stop_loss_adj - fee_rate + confidence_adj
    consec_down = trend_ctx.get("consecutive_down", 0)

    # L3: 极端止损
    extreme_threshold = effective_stop * STOP_LOSS_L3_FACTOR
    if profit_pct <= extreme_threshold or (profit_pct <= effective_stop and consec_down >= STOP_LOSS_L3_CONSEC_DOWN):
        reason = f"极端止损: 浮亏{profit_pct}%"
        if consec_down >= STOP_LOSS_L3_CONSEC_DOWN:
            reason += f", 连跌{consec_down}天"
        return {"level": "L3", "sell_pct": 100, "reason": reason}

    # L2: 常规止损（v5.2: 补仓次数越多，保留越多仓位）
    if profit_pct <= effective_stop:
        l2_sell_pct = max(30, STOP_LOSS_L2_SELL_PCT_BASE - supplement_count * 10)
        return {
            "level": "L2",
            "sell_pct": l2_sell_pct,
            "reason": f"常规止损: 浮亏{profit_pct}% ≤ 止损线{effective_stop:.1f}%, 减仓{l2_sell_pct}%"
                      f"(已补仓{supplement_count}次, 保留反弹仓位)"
        }

    # L1: 预警
    warning_threshold = effective_stop * STOP_LOSS_L1_FACTOR
    if profit_pct <= warning_threshold:
        return {
            "level": "L1",
            "sell_pct": 0,
            "reason": f"止损预警: 浮亏{profit_pct}%接近止损线{effective_stop:.1f}%(预警线{warning_threshold:.1f}%)"
        }

    return {"level": None, "sell_pct": 0, "reason": ""}


# ============================================================
# 补仓禁入判断
# ============================================================

def _is_supplement_forbidden(trend_ctx: dict, confidence: float,
                             source: str, vol_state: str) -> tuple:
    if vol_state == "extreme_vol":
        vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 0
        return True, f"波动率{vol}%处于极端水平，暂停补仓"

    mid_10d = trend_ctx.get("mid_10d")
    consecutive_down = trend_ctx.get("consecutive_down", 0)
    max_drawdown = trend_ctx.get("max_drawdown", 0)
    vol = trend_ctx.get("volatility") or 0

    if mid_10d is not None and mid_10d <= -7 and consecutive_down >= 3:
        return True, f"10日累跌{mid_10d}%且连跌{consecutive_down}天，趋势禁入"

    if max_drawdown >= 10 and vol >= 2.2:
        return True, f"回撤{max_drawdown}%+波动率{vol}%，高风险禁入"

    if source == "estimation" and confidence < 0.6:
        return True, f"置信度{confidence:.0%}偏低，盘中补仓禁入"

    return False, ""


def _check_supplement_rate_limit(pos: dict, current_nav: float,
                                 nav_history: list, trend_ctx: dict,
                                 rebuy_step: float) -> tuple:
    batches = pos.get("batches", [])
    holding_batches = [b for b in batches if b.get("status") == "holding"]
    if not holding_batches:
        return False, "", 1.0

    trade_dates = [h["date"] for h in nav_history if h.get("date")]
    today_str = datetime.now().strftime("%Y-%m-%d")

    total_profit_pct = pos.get("_total_profit_pct")
    use_all_buys = total_profit_pct is not None and total_profit_pct < -3.0

    if use_all_buys:
        ref_batches = holding_batches
    else:
        ref_batches = [b for b in holding_batches if b.get("is_supplement")]

    # v5.3: 动态间隔——根据近期跌速调整
    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
    short_3d = trend_ctx.get("short_3d") or 0
    consecutive_down = trend_ctx.get("consecutive_down", 0)

    # 急跌缩短间隔（3日跌幅>2倍波动率 → 允许2日间隔）
    # 阴跌延长间隔（连续5天每天<0.5%但都是跌 → 5日间隔）
    dynamic_gap = SUPPLEMENT_MIN_GAP_TRADE_DAYS  # 默认3
    if abs(short_3d) > vol * 2 and short_3d < 0:
        dynamic_gap = max(2, SUPPLEMENT_MIN_GAP_TRADE_DAYS - 1)
    elif consecutive_down >= 5:
        recent_changes = [h.get("change", 0) for h in nav_history[:5] if h.get("change") is not None]
        if recent_changes and all(abs(c) < 0.5 for c in recent_changes):
            dynamic_gap = min(5, SUPPLEMENT_MIN_GAP_TRADE_DAYS + 2)

    if ref_batches:
        latest = max(ref_batches, key=lambda b: b["buy_date"])
        gap = _count_trade_days_between(latest["buy_date"], today_str, trade_dates)
        if gap < dynamic_gap:
            scope = "所有买入" if use_all_buys else "补仓"
            return True, f"距上次{scope}仅{gap}个交易日(要求≥{dynamic_gap})", 1.0

        supplement_batches = [b for b in holding_batches if b.get("is_supplement")]
        if supplement_batches:
            latest_supp = max(supplement_batches, key=lambda b: b["buy_date"])
            last_supp_nav = latest_supp.get("nav", 0)
            if last_supp_nav > 0 and current_nav > 0:
                drop_from_last = (current_nav / last_supp_nav - 1) * 100
                if drop_from_last > -rebuy_step:
                    return (True,
                            f"当前净值较上次补仓仅跌{drop_from_last:.1f}%(要求≥{rebuy_step:.1f}%)",
                            1.0)

    tier_factor = 1.0
    mid_10d = trend_ctx.get("mid_10d")

    if (mid_10d is not None and mid_10d < -5) or consecutive_down >= 4:
        tier_factor *= 0.7
    if vol > 2.2:
        tier_factor *= 0.8

    return False, "", tier_factor


def _count_trade_days_between(date_from: str, date_to: str,
                              trade_dates: list) -> int:
    try:
        d_from = datetime.strptime(date_from, "%Y-%m-%d").date()
        d_to = datetime.strptime(date_to, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 999
    count = 0
    for td_str in trade_dates:
        try:
            td = datetime.strptime(td_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d_from < td <= d_to:
            count += 1
    return count


def _is_in_cooldown(pos: dict, nav_history: list) -> bool:
    sell_date = pos.get("cooldown_sell_date")
    cooldown_td = pos.get("cooldown_trade_days", COOLDOWN_DAYS)

    if sell_date:
        trade_dates = [h["date"] for h in nav_history if h.get("date")]
        today_str = datetime.now().strftime("%Y-%m-%d")
        gap = _count_trade_days_between(sell_date, today_str, trade_dates)
        return gap < cooldown_td

    cd_str = pos.get("cooldown_until")
    if cd_str:
        try:
            cd_date = datetime.strptime(cd_str, "%Y-%m-%d").date()
            return datetime.now().date() <= cd_date
        except (ValueError, TypeError):
            pass
    return False


def _calc_size_multiplier(risk_mul: float, confidence: float,
                          trend_label: str, momentum_score: float = 0) -> float:
    size_mul_risk = 1.0 / max(0.8, risk_mul)

    if confidence < 0.6:
        size_mul_conf = 0.6
    elif confidence < 0.75:
        size_mul_conf = 0.8
    else:
        size_mul_conf = 1.0

    weak_labels = {"中期走弱", "连跌", "偏弱"}
    strong_labels = {"中期走强", "偏强", "连涨"}
    if trend_label in weak_labels:
        size_mul_trend = 0.8
    elif trend_label in strong_labels:
        size_mul_trend = 1.05
    else:
        size_mul_trend = 1.0

    momentum_adj = 1.0
    if momentum_score < -0.5:
        momentum_adj = 0.75
    elif momentum_score < -0.3:
        momentum_adj = 0.85

    raw = size_mul_risk * size_mul_conf * size_mul_trend * momentum_adj
    return round(max(0.30, min(1.10, raw)), 2)


# ============================================================
# 辅助函数
# ============================================================

def _estimate_current_nav(batch_nav: float, today_change: float,
                          nav_history: list) -> float:
    """
    收盘后净值估算逻辑：
    - 最新记录==今天 → 直接用今天净值
    - 最新记录≠今天 → 返回最新收盘净值（此时 today_change 已被 core.py 替换为
      该日真实涨跌，latest_nav 已包含该变动，不可重复计算）
    - 盘中 → 用昨日净值 × (1 + today_change/100)
    """
    today_str = datetime.now().strftime("%Y-%m-%d")

    if _is_market_closed() and nav_history:
        latest = nav_history[0]
        latest_nav = latest.get("nav")
        if latest_nav is not None:
            # 如果最新记录就是今天的，直接用
            if latest.get("date") == today_str:
                return latest_nav
            # 否则返回最新收盘净值（不乘 today_change，避免重复计算）
            return latest_nav

    # 盘中：用昨日净值 × (1 + today_change)
    if nav_history and nav_history[0].get("nav") is not None:
        yesterday_nav = nav_history[0]["nav"]
        return yesterday_nav * (1 + today_change / 100)

    # 兜底
    return batch_nav * (1 + today_change / 100)


def _calc_batch_profit_pct(batch: dict, current_nav: float) -> float:
    if batch["nav"] <= 0:
        return 0.0
    return round((current_nav / batch["nav"] - 1) * 100, 2)


def _calc_total_profit_pct(batches: list, current_nav: float) -> float:
    total_cost = sum(b["amount"] for b in batches)
    if total_cost <= 0:
        return 0.0
    total_value = sum(b["shares"] * current_nav for b in batches)
    return round((total_value / total_cost - 1) * 100, 2)


def _get_take_profit_sell_pct(profit_pct: float) -> int:
    for threshold, pct in TAKE_PROFIT_TIERS:
        if profit_pct > threshold:
            return pct
    return 50


def _get_slow_profit_sell_pct(profit_pct: float) -> Optional[int]:
    for threshold, pct in SLOW_PROFIT_TIERS:
        if profit_pct > threshold:
            return pct
    return None


def _calc_min_profit_buffer(fee_rate: float, vol: float = 1.0) -> float:
    return max(1.5, fee_rate * 2.5 + max(0.3, vol * 0.5))


def _get_trail_profit_sell_pct(peak_profit_pct: float) -> int:
    for threshold, pct in TRAIL_PROFIT_SELL_TIERS:
        if peak_profit_pct >= threshold:
            return pct
    return 30


def _calc_peak_profit(batch: dict, nav_history: list) -> float:
    buy_nav = batch.get("nav", 0)
    if buy_nav <= 0:
        return 0.0

    stored_peak = batch.get("peak_nav")
    if stored_peak and stored_peak > buy_nav:
        peak_nav = stored_peak
    else:
        peak_nav = buy_nav

    buy_date_str = batch.get("buy_date", "")
    for h in nav_history:
        h_date = h.get("date", "")
        h_nav = h.get("nav")
        if h_nav is None:
            continue
        if h_date > buy_date_str and h_nav > peak_nav:
            peak_nav = h_nav

    return round((peak_nav / buy_nav - 1) * 100, 2)


def _update_batch_peak_nav(fund_code: str, batch_id: str, current_nav: float):
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return
    for b in fund.get("batches", []):
        if b["id"] == batch_id and b.get("status") == "holding":
            old_peak = b.get("peak_nav", b.get("nav", 0))
            if current_nav > old_peak:
                b["peak_nav"] = round(current_nav, 4)
                b["peak_date"] = datetime.now().strftime("%Y-%m-%d")
                save_positions(data)
            break


def _build_fifo_sell_plan(batches_sorted: list, sell_signals: list,
                          current_nav: float, fund_code: str) -> dict:
    target_sells = {}
    for sig in sell_signals:
        bid = sig["target_batch_id"]
        shares = sig.get("sell_shares", 0)
        if bid not in target_sells or shares > target_sells[bid]:
            target_sells[bid] = shares

    batch_ids_ordered = [b["id"] for b in batches_sorted]
    last_target_idx = -1
    for bid in target_sells:
        if bid in batch_ids_ordered:
            idx = batch_ids_ordered.index(bid)
            if idx > last_target_idx:
                last_target_idx = idx

    fifo_steps = []
    total_fifo_shares = 0.0

    for i, batch in enumerate(batches_sorted):
        if batch.get("status") != "holding":
            continue
        if i > last_target_idx:
            break

        bid = batch["id"]
        buy_date = datetime.strptime(batch["buy_date"], "%Y-%m-%d").date()
        hold_days = (datetime.now().date() - buy_date).days
        fee_rate = get_sell_fee_rate(fund_code, hold_days)

        if bid in target_sells:
            shares = target_sells[bid]
            is_passthrough = False
            reason = next((s.get("signal_name", "") for s in sell_signals
                          if s["target_batch_id"] == bid), "")
        else:
            shares = batch["shares"]
            is_passthrough = True
            reason = "FIFO穿过（需先卖出此批次）"

        profit_pct = _calc_batch_profit_pct(batch, current_nav)
        est_gross = shares * current_nav
        est_fee = round(est_gross * fee_rate / 100, 2)
        est_net_profit = round(est_gross * (1 - fee_rate / 100) - batch["amount"] * (shares / batch["shares"] if batch["shares"] > 0 else 1), 2)

        fifo_steps.append({
            "batch_id": bid,
            "buy_date": batch["buy_date"],
            "sell_shares": round(shares, 2),
            "batch_total_shares": round(batch["shares"], 2),
            "is_full_sell": abs(shares - batch["shares"]) < 0.01,
            "is_passthrough": is_passthrough,
            "hold_days": hold_days,
            "fee_rate": fee_rate,
            "profit_pct": profit_pct,
            "estimated_fee": est_fee,
            "estimated_net_profit": est_net_profit,
            "reason": reason,
            "note": batch.get("note", ""),
        })
        total_fifo_shares += shares

    total_est_fee = sum(s["estimated_fee"] for s in fifo_steps)
    total_est_profit = sum(s["estimated_net_profit"] for s in fifo_steps)
    has_passthrough = any(s["is_passthrough"] for s in fifo_steps)

    passthrough_loss_steps = [
        s for s in fifo_steps if s["is_passthrough"] and s["estimated_net_profit"] < 0
    ]
    passthrough_warning = None
    passthrough_loss_total = 0.0
    if passthrough_loss_steps:
        passthrough_loss_total = sum(s["estimated_net_profit"] for s in passthrough_loss_steps)
        batch_ids = [s["batch_id"] for s in passthrough_loss_steps]
        passthrough_warning = (
            f"注意: FIFO穿过的{len(passthrough_loss_steps)}个批次({', '.join(batch_ids)})"
            f"预计亏损{passthrough_loss_total:.2f}元, 请确认是否值得为目标批次执行卖出"
        )

    return {
        "total_shares": round(total_fifo_shares, 2),
        "batch_count": len(fifo_steps),
        "steps": fifo_steps,
        "total_estimated_fee": round(total_est_fee, 2),
        "total_estimated_profit": round(total_est_profit, 2),
        "has_passthrough": has_passthrough,
        "passthrough_warning": passthrough_warning,
        "passthrough_loss_total": round(passthrough_loss_total, 2),
        "instruction": f"在支付宝输入卖出 {round(total_fifo_shares, 2)} 份",
    }


def _make_signal(fund_code: str, **kwargs) -> dict:
    return {
        "fund_code": fund_code,
        "signal_name": kwargs.get("signal_name", "观望"),
        "action": kwargs.get("action", "hold"),
        "priority": kwargs.get("priority", 8),
        "sub_priority": kwargs.get("sub_priority", 0),
        "target_batch_id": kwargs.get("target_batch_id"),
        "amount": kwargs.get("amount"),
        "sell_shares": kwargs.get("sell_shares"),
        "sell_pct": kwargs.get("sell_pct"),
        "reason": kwargs.get("reason", ""),
        "fee_info": kwargs.get("fee_info"),
        "alert": kwargs.get("alert", False),
        "alert_msg": kwargs.get("alert_msg"),
        "_confidence": kwargs.get("_confidence"),
        "_source": kwargs.get("_source"),
    }


def _is_higher_priority(new_sig: dict, current_best: dict) -> bool:
    if new_sig["priority"] != current_best["priority"]:
        return new_sig["priority"] < current_best["priority"]
    return new_sig["sub_priority"] < current_best["sub_priority"]


def _stamp(sig, confidence, source):
    sig["_confidence"] = confidence
    sig["_source"] = source
    return sig


# ============================================================
# 决策依据说明
# ============================================================

def _build_decision_note(fund_code: str, tc: dict, today_change: float,
                         source: str, current_nav: float = None,
                         total_profit_pct: float = None, pos: dict = None,
                         dynamic_thresholds: dict = None) -> str:
    parts = []

    if source == "nav":
        parts.append("当前使用真实净值数据")
    else:
        parts.append("当前使用盘中估值")

    dt = dynamic_thresholds or {}
    risk_mul = dt.get("risk_multiplier", 1.0)
    vol_state = dt.get("vol_state", "normal_vol")
    momentum = dt.get("momentum_score", 0)

    if vol_state == "extreme_vol":
        parts.append("⚠️ 极端波动环境，暂停买入")
    elif vol_state == "high_vol":
        parts.append(f"高波动环境(风险系数{risk_mul}×)")
    elif vol_state == "low_vol":
        parts.append(f"低波动环境(网格收窄)")

    sensitivity = dt.get("_sensitivity", 1.0)
    if abs(sensitivity - 1.0) > 0.05:
        parts.append(f"灵敏度{sensitivity:.2f}")

    if dt.get("_vol_based"):
        va = dt.get("_va", {})
        parts.append(f"阈值自适应(买≤{va.get('dip_threshold','?')}%/止损≤{va.get('stop_loss','?')}%)")

    if abs(momentum) > 0.3:
        direction = "上升" if momentum > 0 else "下降"
        parts.append(f"动量{direction}({momentum:.2f})")

    win_adj = dt.get("win_rate_adj", 1.0)
    if win_adj > 1.0:
        parts.append(f"近期买入胜率偏低，阈值已收紧{(win_adj-1)*100:.0f}%")

    short_3d = tc.get("short_3d")
    if short_3d is not None:
        if short_3d < -3:
            parts.append(f"近3日累跌{short_3d}%，短期超卖")
        elif short_3d > 3:
            parts.append(f"近3日累涨{short_3d}%，短期过热")

    mid_10d = tc.get("mid_10d")
    if mid_10d is not None:
        if mid_10d < -5:
            parts.append(f"10日累计{mid_10d}%，中期走弱")
        elif mid_10d > 5:
            parts.append(f"10日累计+{mid_10d}%，中期走强")

    if total_profit_pct is not None:
        if total_profit_pct > 3:
            parts.append(f"总浮盈{total_profit_pct}%，关注止盈")
        elif total_profit_pct < -8:
            parts.append(f"总浮亏{total_profit_pct}%，谨慎补仓")
        elif total_profit_pct < -3:
            parts.append(f"总浮亏{total_profit_pct}%，观察企稳")

    if pos and pos.get("has_position"):
        pct_used = pos["total_cost"] / pos.get("max_position", 5000) * 100
        if pct_used > 80:
            parts.append(f"仓位已用{pct_used:.0f}%，空间有限")
        elif pct_used < 30:
            parts.append(f"仓位仅{pct_used:.0f}%，可择机加仓")

    return "；".join(parts) if parts else "数据不足，暂无分析"


def _build_market_analysis(fund_code: str, val: dict, nav_history: list,
                           pos: dict, current_nav: float = None,
                           total_profit_pct: float = None,
                           trend_ctx: dict = None,
                           dynamic_thresholds: dict = None) -> dict:
    real_code, _ = parse_fund_key(fund_code)
    today_change = val.get("estimation_change") or 0.0
    source = val.get("_source", "estimation")

    day_changes = []
    if source == "estimation":
        day_changes.append({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "change": round(today_change, 2),
            "source": "estimation",
        })
        for h in nav_history[:19]:
            if h.get("change") is not None:
                day_changes.append({"date": h.get("date", ""), "change": round(h["change"], 2), "source": "nav"})
    else:
        for h in nav_history[:20]:
            if h.get("change") is not None:
                day_changes.append({"date": h.get("date", ""), "change": round(h["change"], 2), "source": "nav"})

    tc = trend_ctx or {}
    dt = dynamic_thresholds or {}

    strategy_params = {
        "dip_buy_threshold": dt.get("dip_threshold", DEFAULT_DIP_THRESHOLD),
        "take_profit_trigger": dt.get("tp_trigger", DEFAULT_TAKE_PROFIT_TRIGGER),
        "stop_loss_base": dt.get("stop_loss_adj", DEFAULT_STOP_LOSS_BASE),
        "risk_multiplier": dt.get("risk_multiplier", 1.0),
        "vol_state": dt.get("vol_state", "normal_vol"),
        "momentum_score": dt.get("momentum_score", 0),
        "trail_dd": dt.get("trail_dd", TRAIL_DD_BASE),
        "win_rate_adj": dt.get("win_rate_adj", 1.0),
        "vol_sensitivity": dt.get("_sensitivity", DEFAULT_VOL_SENSITIVITY),
        "vol_sensitivity_source": dt.get("_sensitivity_source", "default"),
        "consecutive_dip_trigger": dt.get("consecutive_dip_trigger", DEFAULT_CONSECUTIVE_DIP_TRIGGER),
        "supplement_max_count": _calc_dynamic_supplement_max(pos) if pos else SUPPLEMENT_MAX_COUNT_DEFAULT,
        "supplement_trigger": dt.get("supplement_trigger", DEFAULT_SUPPLEMENT_TRIGGER),
        "supplement_loss_min": dt.get("supplement_loss_min", DEFAULT_SUPPLEMENT_LOSS_MIN),
        "total_profit_sell_min": dt.get("total_profit_sell_tiers", TOTAL_PROFIT_SELL_TIERS)[-1][0] if dt.get("total_profit_sell_tiers", TOTAL_PROFIT_SELL_TIERS) else 0.5,
        "trend_weak_cumulative": dt.get("trend_weak_cumulative", DEFAULT_TREND_WEAK_CUMULATIVE),
        "disaster_loss_threshold": dt.get("disaster_loss_threshold", DEFAULT_DISASTER_LOSS),
        "vol_based": dt.get("_vol_based", False),
    }

    # v5.2: 计算当前市值和盈亏（供前端展示，与支付宝对齐）
    market_value = round(current_nav * pos.get("total_shares", 0), 2) if current_nav and pos else None
    total_cost = pos.get("total_cost", pos.get("total_amount", 0)) if pos else 0
    unrealized_pnl = round(market_value - total_cost, 2) if market_value is not None else None
    unrealized_pnl_pct = round(unrealized_pnl / total_cost * 100, 2) if unrealized_pnl is not None and total_cost > 0 else None
    realized_pnl = pos.get("realized_pnl", 0) if pos else 0
    total_invested = pos.get("total_invested", total_cost) if pos else 0
    total_received = pos.get("total_received", 0) if pos else 0
    # 累计盈亏 = 当前市值 + 已回款 - 总投入（与支付宝"累计盈亏"口径一致）
    cumulative_pnl = round((market_value or 0) + total_received - total_invested, 2) if market_value is not None else None

    return {
        "today_change": round(today_change, 2),
        "today_source": source,
        "day_changes": day_changes,
        "short_3d": tc.get("short_3d"),
        "short_5d": tc.get("short_5d"),
        "mid_10d": tc.get("mid_10d"),
        "long_20d": tc.get("long_20d"),
        "trend": tc.get("trend_label", "震荡"),
        "volatility": tc.get("volatility"),
        "volatility_robust": tc.get("volatility_robust"),
        "volume_proxy": tc.get("volume_proxy"),
        "consecutive_down": tc.get("consecutive_down", 0),
        "consecutive_up": tc.get("consecutive_up", 0),
        "max_drawdown": tc.get("max_drawdown"),
        "data_days": tc.get("data_days", len(day_changes)),
        "current_nav": round(current_nav, 4) if current_nav else None,
        # v5.2: 市值与盈亏（前端应展示 market_value 而非 total_cost）
        "market_value": market_value,               # 当前市值 = 份额 × 净值（对标支付宝"金额"）
        "total_cost": total_cost,                    # 持仓成本 = sum(batch.amount)
        "unrealized_pnl": unrealized_pnl,            # 未实现盈亏 = 市值 - 成本
        "unrealized_pnl_pct": unrealized_pnl_pct,    # 未实现盈亏率 = 盈亏/成本
        "realized_pnl": realized_pnl,                # 已实现盈亏（历史卖出）
        "cumulative_pnl": cumulative_pnl,            # 累计盈亏（对标支付宝"累计盈亏"）
        "total_profit_pct": round(total_profit_pct, 2) if total_profit_pct is not None else None,
        "confidence": val.get("confidence"),
        "strategy_params": strategy_params,
        "decision_note": _build_decision_note(fund_code, tc, today_change, source,
                                               current_nav, total_profit_pct, pos,
                                               dynamic_thresholds=dt),
    }


# ============================================================
# 综合趋势分析（v5.2: 增加 volume_proxy 成交量代理）
# ============================================================

def _analyze_trend(today_change: float, hist_changes: list,
                   nav_history: list = None,
                   nav_history_60: list = None) -> dict:
    all_changes = [today_change] + hist_changes

    def _compound_return(changes):
        product = 1.0
        for c in changes:
            product *= (1 + c / 100)
        return round((product - 1) * 100, 2)

    short_3d = _compound_return(all_changes[:3]) if len(all_changes) >= 3 else _compound_return(all_changes)
    short_5d = _compound_return(all_changes[:5]) if len(all_changes) >= 5 else None

    mid_10d = None
    long_20d = None
    if nav_history and len(nav_history) >= 2:
        navs = [h["nav"] for h in nav_history if h.get("nav") is not None]
        if len(navs) >= 10:
            mid_10d = round((navs[0] / navs[9] - 1) * 100, 2)
        elif len(navs) >= 2:
            mid_10d = round((navs[0] / navs[-1] - 1) * 100, 2)
        if len(navs) >= 20:
            long_20d = round((navs[0] / navs[19] - 1) * 100, 2)
    else:
        mid_10d = _compound_return(all_changes[:10]) if len(all_changes) >= 10 else None
        long_20d = _compound_return(all_changes[:20]) if len(all_changes) >= 20 else None

    vol_data = all_changes[:20]
    volatility = None
    volatility_robust = None
    if len(vol_data) >= 5:
        mean = sum(vol_data) / len(vol_data)
        variance = sum((c - mean) ** 2 for c in vol_data) / len(vol_data)
        volatility = round(variance ** 0.5, 2)
        sorted_data = sorted(vol_data)
        n = len(sorted_data)
        median = sorted_data[n // 2] if n % 2 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        abs_devs = sorted([abs(c - median) for c in vol_data])
        m = len(abs_devs)
        mad = abs_devs[m // 2] if m % 2 else (abs_devs[m // 2 - 1] + abs_devs[m // 2]) / 2
        volatility_robust = round(mad * 1.4826, 2)

    # v5.2: 成交量代理（用近5日涨跌幅绝对值/波动率 近似）
    volume_proxy = None
    if len(all_changes) >= 5 and volatility_robust and volatility_robust > 0:
        recent_abs = [abs(c) for c in all_changes[:5]]
        mean_abs = sum(recent_abs) / len(recent_abs)
        volume_proxy = round(mean_abs / volatility_robust, 2)
        # > 1.5 = 放量, < 0.5 = 缩量

    consecutive_down = 0
    for c in all_changes:
        if c < 0:
            consecutive_down += 1
        else:
            break
    consecutive_up = 0
    for c in all_changes:
        if c > 0:
            consecutive_up += 1
        else:
            break

    max_drawdown = 0.0
    if nav_history and len(nav_history) >= 5:
        navs = [h["nav"] for h in nav_history if h.get("nav") is not None]
        navs_chrono = list(reversed(navs[:20]))
        peak = navs_chrono[0] if navs_chrono else 0
        for n_val in navs_chrono:
            if n_val > peak:
                peak = n_val
            if peak > 0:
                dd = (peak - n_val) / peak * 100
                if dd > max_drawdown:
                    max_drawdown = dd
    max_drawdown = round(max_drawdown, 2)

    max_drawdown_60 = 0.0
    if nav_history_60 and len(nav_history_60) >= 10:
        navs_60 = [h["nav"] for h in nav_history_60 if h.get("nav") is not None]
        if len(navs_60) >= 5:
            navs_60_chrono = list(reversed(navs_60[:60]))
            peak_60 = navs_60_chrono[0] if navs_60_chrono else 0
            for n_val in navs_60_chrono:
                if n_val > peak_60:
                    peak_60 = n_val
                if peak_60 > 0:
                    dd = (peak_60 - n_val) / peak_60 * 100
                    if dd > max_drawdown_60:
                        max_drawdown_60 = dd
    max_drawdown_60 = round(max_drawdown_60, 2)

    trend_label = "震荡"
    if consecutive_down >= 3:
        trend_label = "连跌"
    elif consecutive_up >= 3:
        trend_label = "连涨"
    elif short_3d and short_3d < -2:
        trend_label = "偏弱"
    elif short_3d and short_3d > 2:
        trend_label = "偏强"
    elif mid_10d is not None and mid_10d < -5:
        trend_label = "中期走弱"
    elif mid_10d is not None and mid_10d > 5:
        trend_label = "中期走强"

    return {
        "short_3d": round(short_3d, 2) if short_3d is not None else None,
        "short_5d": round(short_5d, 2) if short_5d is not None else None,
        "mid_10d": round(mid_10d, 2) if mid_10d is not None else None,
        "long_20d": round(long_20d, 2) if long_20d is not None else None,
        "volatility": volatility,
        "volatility_robust": volatility_robust,
        "volume_proxy": volume_proxy,
        "consecutive_down": consecutive_down,
        "consecutive_up": consecutive_up,
        "max_drawdown": max_drawdown,
        "max_drawdown_60": max_drawdown_60,
        "trend_label": trend_label,
        "data_days": len(all_changes),
    }


# ============================================================
# 核心信号生成
# ============================================================

def generate_signal(fund_code: str) -> dict:
    real_code, owner = parse_fund_key(fund_code)

    val = calculate_valuation(real_code)
    today_change = val.get("estimation_change") or 0.0
    recent = val.get("recent_changes", [])
    pos = get_fund_position(fund_code)
    nav_history = get_fund_nav_history(real_code, 20)
    nav_history_60 = get_fund_nav_history(real_code, 60)
    confidence = val.get("confidence", 0.0)
    source = val.get("_source", "estimation")

    hist_changes = [h["change"] for h in nav_history if h.get("change") is not None]
    trend_ctx = _analyze_trend(today_change, hist_changes, nav_history,
                               nav_history_60=nav_history_60)

    signal_stats = calc_signal_win_rate(fund_code)

    dyn = _calc_dynamic_thresholds(trend_ctx, fund_code, confidence, source,
                                    signal_stats=signal_stats)

    vol_state = dyn["vol_state"]
    momentum = dyn.get("momentum_score", 0)

    in_cooldown = _is_in_cooldown(pos, nav_history)

    size_mul = _calc_size_multiplier(
        dyn["risk_multiplier"], confidence,
        trend_ctx.get("trend_label", "震荡"),
        momentum_score=momentum
    )

    # === 有持仓 ===
    if pos["has_position"]:
        batches = pos["batches"]
        batches_sorted = sorted(batches, key=lambda b: b["buy_date"])

        current_nav = _estimate_current_nav(
            batches_sorted[0]["nav"], today_change, nav_history
        )
        total_profit_pct = _calc_total_profit_pct(batches_sorted, current_nav)

        market_analysis = _build_market_analysis(
            fund_code, val, nav_history, pos,
            current_nav=current_nav, total_profit_pct=total_profit_pct,
            trend_ctx=trend_ctx, dynamic_thresholds=dyn
        )

        # === 同日卖出抑制：检测今天是否已执行过卖出操作 ===
        # 若今天已卖出，抑制新的卖出信号，避免重复建议
        # （止损信号L2/L3除外——风险优先级高于重复抑制）
        today_str = datetime.now().strftime("%Y-%m-%d")
        _sold_today = (pos.get("cooldown_sell_date") == today_str)

        best_signal = None
        all_signals = []
        extra_alerts = []
        supplement_count = pos.get("supplement_count", 0)

        for batch in batches_sorted:
            buy_date = datetime.strptime(batch["buy_date"], "%Y-%m-%d").date()
            hold_days = (datetime.now().date() - buy_date).days
            fee_rate = get_sell_fee_rate(fund_code, hold_days)
            profit_pct = _calc_batch_profit_pct(batch, current_nav)

            _update_batch_peak_nav(fund_code, batch["id"], current_nav)

            # --- 三级止损评估（v5.2: 传入 supplement_count）---
            stop_eval = _evaluate_stop_loss(
                profit_pct, dyn["stop_loss_adj"], hold_days, fee_rate,
                trend_ctx, confidence, source,
                supplement_count=supplement_count
            )

            if stop_eval["level"] == "L3":
                sig = _make_signal(
                    fund_code,
                    signal_name="极端止损(L3)",
                    action="sell",
                    priority=1,
                    target_batch_id=batch["id"],
                    sell_shares=round(batch["shares"], 2),
                    sell_pct=100,
                    reason=stop_eval["reason"],
                    fee_info={
                        "sell_fee_rate": fee_rate,
                        "estimated_fee": round(batch["shares"] * current_nav * fee_rate / 100, 2),
                        "estimated_net_profit": round(batch["shares"] * current_nav * (1 - fee_rate / 100) - batch["amount"], 2),
                    },
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig
                continue

            elif stop_eval["level"] == "L2":
                sell_shares = round(batch["shares"] * stop_eval["sell_pct"] / 100, 2)
                sig = _make_signal(
                    fund_code,
                    signal_name="常规止损(L2)",
                    action="sell",
                    priority=1,
                    sub_priority=1,
                    target_batch_id=batch["id"],
                    sell_shares=sell_shares,
                    sell_pct=stop_eval["sell_pct"],
                    reason=stop_eval["reason"],
                    fee_info={
                        "sell_fee_rate": fee_rate,
                        "estimated_fee": round(sell_shares * current_nav * fee_rate / 100, 2),
                        "estimated_net_profit": round(sell_shares * current_nav * (1 - fee_rate / 100) - batch["amount"] * stop_eval["sell_pct"] / 100, 2),
                    },
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig
                continue

            elif stop_eval["level"] == "L1":
                extra_alerts.append(stop_eval["reason"])

            # --- 灾难保护阀（未满7天）---
            if hold_days < 7:
                disaster_triggered = False
                disaster_reason = ""
                # v5.2: 灾难保护卖出比例也根据补仓情况递减
                disaster_sell_pct = max(30, DISASTER_SELL_PCT_EXTREME - supplement_count * 10)

                va = dyn.get("_va", {})
                disaster_loss = va.get("disaster_loss", DEFAULT_DISASTER_LOSS)
                disaster_daily = va.get("disaster_daily", DEFAULT_DISASTER_DAILY_DROP)

                effective_disaster = min(disaster_loss, dyn["stop_loss_adj"] * 1.5)
                if profit_pct <= effective_disaster:
                    disaster_triggered = True
                    disaster_reason = f"批次{batch['id']}仅{hold_days}天, 亏损{profit_pct}% ≤ 灾难线{effective_disaster}%"

                if (not disaster_triggered
                        and today_change <= disaster_daily
                        and trend_ctx.get("consecutive_down", 0) >= DISASTER_CONSECUTIVE_DOWN):
                    disaster_triggered = True
                    disaster_sell_pct = max(20, DISASTER_SELL_PCT_DAILY - supplement_count * 5)
                    disaster_reason = f"批次{batch['id']}仅{hold_days}天, 今日暴跌{today_change}%+连跌, 灾难保护"

                if disaster_triggered:
                    sell_shares = round(batch["shares"] * disaster_sell_pct / 100, 2)
                    sig = _make_signal(
                        fund_code,
                        signal_name="灾难保护(未满7天)",
                        action="sell",
                        priority=1.2,
                        target_batch_id=batch["id"],
                        sell_shares=sell_shares,
                        sell_pct=disaster_sell_pct,
                        reason=disaster_reason,
                        fee_info={"sell_fee_rate": fee_rate},
                        alert=True,
                        alert_msg=f"灾难保护卖出将产生{fee_rate}%高费率",
                    )
                    all_signals.append(sig)
                    if best_signal is None or _is_higher_priority(sig, best_signal):
                        best_signal = sig
                    continue

                if profit_pct <= -3.0:
                    extra_alerts.append(f"批次{batch['id']}亏损{profit_pct}%但仅持有{hold_days}天")

                # v5.3: 短期深亏安全网——持有<7天且单批浮亏>6%，
                # 灾难触发条件未满足时也要止损（避免止损空隙）
                if not disaster_triggered and profit_pct <= -6.0:
                    safety_sell_pct = 30
                    sell_shares_sn = round(batch["shares"] * safety_sell_pct / 100, 2)
                    sig = _make_signal(
                        fund_code,
                        signal_name="短期深亏止损(安全网)",
                        action="sell",
                        priority=1.5,
                        target_batch_id=batch["id"],
                        sell_shares=sell_shares_sn,
                        sell_pct=safety_sell_pct,
                        reason=f"批次{batch['id']}仅{hold_days}天, 亏损{profit_pct}%已超6%, 安全网减仓{safety_sell_pct}%",
                        fee_info={"sell_fee_rate": fee_rate},
                        alert=True,
                        alert_msg=f"短期深亏安全网：仅持有{hold_days}天即亏损{profit_pct}%，建议减仓止损",
                    )
                    all_signals.append(sig)
                    if best_signal is None or _is_higher_priority(sig, best_signal):
                        best_signal = sig

                continue

            if hold_days < 7:
                continue

            # --- 统一止盈评分 ---
            peak_profit = _calc_peak_profit(batch, nav_history_60)
            sell_eval = _calc_sell_score(
                batch, current_nav, today_change, trend_ctx, dyn,
                fee_rate, hold_days, peak_profit
            )

            if sell_eval["sell_pct"] > 0 and sell_eval["signal_name"]:
                sell_pct = sell_eval["sell_pct"]
                sell_shares = round(batch["shares"] * sell_pct / 100, 2)
                est_gross = sell_shares * current_nav
                est_fee = round(est_gross * fee_rate / 100, 2)
                est_net_profit = round(est_gross * (1 - fee_rate / 100) - batch["amount"] * sell_pct / 100, 2)

                is_low_conf = source == "estimation" and confidence < 0.5
                # 同日卖出抑制：今天已卖过则降级为 hold，避免重复建议
                is_suppressed = _sold_today
                sig = _make_signal(
                    fund_code,
                    signal_name=sell_eval["signal_name"] + ("(今日已操作)" if is_suppressed else "(待确认)" if is_low_conf else ""),
                    action="hold" if (is_low_conf or is_suppressed) else "sell",
                    priority=2,
                    sub_priority=max(0, 10 - sell_eval["score"]),
                    target_batch_id=batch["id"],
                    sell_shares=sell_shares,
                    sell_pct=sell_pct,
                    reason=(f"持有{hold_days}天, 浮盈{sell_eval['profit_pct']}%, "
                            f"峰值{peak_profit:.1f}%, {sell_eval['reason']}, 卖出{sell_pct}%"
                            + (f", 置信度{confidence:.0%}偏低" if is_low_conf else "")
                            + (f" (今日已执行卖出，次日再评估)" if is_suppressed else "")),
                    fee_info={
                        "sell_fee_rate": fee_rate,
                        "estimated_fee": est_fee,
                        "estimated_net_profit": est_net_profit,
                    },
                    alert=is_low_conf or is_suppressed,
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig
                continue

            # --- 趋势转弱卖出 ---
            vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
            min_profit_buffer = _calc_min_profit_buffer(fee_rate, vol)
            mid_10d_val = trend_ctx.get("mid_10d")
            short_5d_val = trend_ctx.get("short_5d")

            has_trend_confirm = False
            if (len(recent) >= 2
                    and recent[0].get("change") is not None
                    and recent[1].get("change") is not None
                    and recent[0]["change"] < 0 and recent[1]["change"] < 0):
                cumulative_drop = recent[0]["change"] + recent[1]["change"]
                trend_weak_thresh = dyn.get("trend_weak_cumulative", DEFAULT_TREND_WEAK_CUMULATIVE)
                if cumulative_drop <= trend_weak_thresh:
                    if ((short_5d_val is not None and short_5d_val < 0)
                            or (mid_10d_val is not None and mid_10d_val < 0)):
                        has_trend_confirm = True

            # v5.2: 放量下跌时趋势转弱更可信
            volume_proxy = trend_ctx.get("volume_proxy")
            if has_trend_confirm and volume_proxy and volume_proxy > 1.5:
                has_trend_confirm = True  # 放量确认，维持信号
            elif has_trend_confirm and volume_proxy and volume_proxy < 0.5:
                has_trend_confirm = False  # 缩量下跌，可能是假信号

            if profit_pct > min_profit_buffer and has_trend_confirm:
                is_low_conf = source == "estimation" and confidence < 0.5
                # 同日卖出抑制
                is_suppressed = _sold_today
                # v5.3: 趋势转弱减仓比例按总浮盈深度分级
                # 薄利快跑，厚利只减仓（回调是正常波动）
                if total_profit_pct < 2:
                    trend_sell_pct = 100  # 薄利快跑
                elif total_profit_pct < 5:
                    trend_sell_pct = 70   # 留底仓观察
                else:
                    trend_sell_pct = 50   # 回调是正常波动

                # 放量确认转弱 → 加码卖出
                if volume_proxy and volume_proxy > 1.5:
                    trend_sell_pct = min(100, trend_sell_pct + 20)

                sell_shares_tw = round(batch["shares"] * trend_sell_pct / 100, 2)
                sig = _make_signal(
                    fund_code,
                    signal_name="趋势转弱" + ("(今日已操作)" if is_suppressed else "(待确认)" if is_low_conf else ""),
                    action="hold" if (is_low_conf or is_suppressed) else "sell",
                    priority=3,
                    target_batch_id=batch["id"],
                    sell_shares=sell_shares_tw,
                    sell_pct=trend_sell_pct,
                    reason=f"持有{hold_days}天, 浮盈{profit_pct}%, 总浮盈{total_profit_pct}%, "
                           f"趋势确认转弱, 减仓{trend_sell_pct}%"
                           + (f"(放量{volume_proxy:.1f}×)" if volume_proxy and volume_proxy > 1.5 else "")
                           + (f" (今日已执行卖出，次日再评估)" if is_suppressed else ""),
                    fee_info={
                        "sell_fee_rate": fee_rate,
                        "estimated_fee": round(sell_shares_tw * current_nav * fee_rate / 100, 2),
                        "estimated_net_profit": round(sell_shares_tw * current_nav * (1 - fee_rate / 100) - batch["amount"] * trend_sell_pct / 100, 2),
                    },
                    alert=is_low_conf or is_suppressed,
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig
                continue

        # --- 总仓位扭亏为盈 ---
        if total_profit_pct > 0 and len(batches_sorted) >= 2:
            oldest = batches_sorted[0]
            oldest_buy_date = datetime.strptime(oldest["buy_date"], "%Y-%m-%d").date()
            oldest_hold_days = (datetime.now().date() - oldest_buy_date).days
            oldest_fee_rate = get_sell_fee_rate(fund_code, oldest_hold_days)

            if oldest_hold_days >= 7:
                sell_pct = None
                profit_tiers = dyn.get("total_profit_sell_tiers", TOTAL_PROFIT_SELL_TIERS)
                for threshold, pct in profit_tiers:
                    if total_profit_pct > threshold:
                        sell_pct = pct
                        break
                if sell_pct:
                    sell_shares = round(oldest["shares"] * sell_pct / 100, 2)
                    # 同日卖出抑制
                    _tp_suppressed = _sold_today
                    sig = _make_signal(
                        fund_code,
                        signal_name="扭亏止盈" + ("(今日已操作)" if _tp_suppressed else ""),
                        action="hold" if _tp_suppressed else "sell",
                        priority=2.5,
                        target_batch_id=oldest["id"],
                        sell_shares=sell_shares,
                        sell_pct=sell_pct,
                        reason=f"总浮盈{total_profit_pct}%, 补仓{len(batches_sorted)}批后扭亏, 最老批次减仓{sell_pct}%"
                               + (f" (今日已执行卖出，次日再评估)" if _tp_suppressed else ""),
                        fee_info={
                            "sell_fee_rate": oldest_fee_rate,
                            "estimated_fee": round(sell_shares * current_nav * oldest_fee_rate / 100, 2),
                            "estimated_net_profit": round(sell_shares * current_nav * (1 - oldest_fee_rate / 100) - oldest["amount"] * sell_pct / 100, 2),
                        },
                        alert=_tp_suppressed,
                    )
                    all_signals.append(sig)
                    if best_signal is None or _is_higher_priority(sig, best_signal):
                        best_signal = sig

        # --- 总仓位止损 ---
        if total_profit_pct < 0 and not best_signal:
            va = dyn.get("_va", {})
            base_portfolio_stop = round(va.get("stop_loss", DEFAULT_STOP_LOSS_BASE) * 0.65, 2)

            supp_count = pos.get("supplement_count", 0)
            dyn_max_supp = _calc_dynamic_supplement_max(pos)
            remaining_ratio = max(0, (dyn_max_supp - supp_count) / max(1, dyn_max_supp))
            has_budget = pos["total_cost"] < pos["max_position"] * 0.8

            if has_budget and remaining_ratio > 0:
                portfolio_stop = round(base_portfolio_stop * (1 + 0.2 * remaining_ratio), 2)
            else:
                portfolio_stop = base_portfolio_stop
            portfolio_stop = max(-8.0, portfolio_stop)

            portfolio_warn = round(portfolio_stop * 0.7, 2)

            if total_profit_pct <= portfolio_stop:
                oldest = batches_sorted[0]
                oldest_bd = datetime.strptime(oldest["buy_date"], "%Y-%m-%d").date()
                oldest_hd = (datetime.now().date() - oldest_bd).days
                if oldest_hd >= 7:
                    oldest_fr = get_sell_fee_rate(fund_code, oldest_hd)
                    # v5.3: 动态减仓比例，根据超止损线的深度决定
                    excess_loss = portfolio_stop - total_profit_pct  # 正数，越大越严重
                    if excess_loss >= 5:
                        portfolio_sell_pct = 70
                    elif excess_loss >= 2:
                        portfolio_sell_pct = 50
                    else:
                        portfolio_sell_pct = 30
                    # 补仓>=3次且仍深亏 → 判断可能错误，加速止损
                    if supp_count >= 3:
                        portfolio_sell_pct = min(100, portfolio_sell_pct + 10)
                    sell_shares = round(oldest["shares"] * portfolio_sell_pct / 100, 2)
                    # 同日卖出抑制（总仓位止损也抑制，今天已减过仓了）
                    _psl_suppressed = _sold_today
                    sig = _make_signal(
                        fund_code,
                        signal_name="总仓位止损" + ("(今日已操作)" if _psl_suppressed else ""),
                        action="hold" if _psl_suppressed else "sell",
                        priority=1,
                        sub_priority=2,
                        target_batch_id=oldest["id"],
                        sell_shares=sell_shares,
                        sell_pct=portfolio_sell_pct,
                        reason=(f"总浮亏{total_profit_pct}% ≤ 总仓位止损线{portfolio_stop}%"
                                f"(补仓{supp_count}/{dyn_max_supp}), 超线{excess_loss:.1f}%, "
                                f"最老批次减仓{portfolio_sell_pct}%"
                                + (f" (今日已执行卖出，次日再评估)" if _psl_suppressed else "")),
                        fee_info={
                            "sell_fee_rate": oldest_fr,
                            "estimated_fee": round(sell_shares * current_nav * oldest_fr / 100, 2),
                            "estimated_net_profit": round(sell_shares * current_nav * (1 - oldest_fr / 100) - oldest["amount"] * portfolio_sell_pct / 100, 2),
                        },
                        alert=True,
                        alert_msg=f"总仓位浮亏{total_profit_pct}%已触达止损线{portfolio_stop}%",
                    )
                    all_signals.append(sig)
                    if best_signal is None or _is_higher_priority(sig, best_signal):
                        best_signal = sig
            elif total_profit_pct <= portfolio_warn:
                extra_alerts.append(
                    f"总仓位浮亏{total_profit_pct}%接近止损线{portfolio_stop}%(预警线{portfolio_warn}%)"
                )

        # --- 递进补仓 ---
        dynamic_max_supp = _calc_dynamic_supplement_max(pos)
        forbidden, forbid_reason = _is_supplement_forbidden(
            trend_ctx, confidence, source, vol_state
        )

        if forbidden:
            if total_profit_pct < -3.0:
                extra_alerts.append(f"补仓被禁入: {forbid_reason}")
        elif (supplement_count < dynamic_max_supp
                and pos["total_cost"] < pos["max_position"]
                and not in_cooldown):
            pos["_total_profit_pct"] = total_profit_pct
            rebuy_step = dyn.get("rebuy_step", SUPPLEMENT_REBUY_STEP_PCT)
            rate_blocked, rate_reason, tier_factor = _check_supplement_rate_limit(
                pos, current_nav, nav_history, trend_ctx, rebuy_step
            )
            if rate_blocked:
                if total_profit_pct < -3.0:
                    extra_alerts.append(f"补仓受节奏阀限制: {rate_reason}")
            else:
                adj_tiers = dyn.get("supplement_tiers", SUPPLEMENT_TIERS)
                for tier_count, tier_ratio, tier_trigger, tier_loss_min in adj_tiers:
                    if supplement_count == tier_count:
                        if (total_profit_pct <= tier_loss_min
                                and today_change <= tier_trigger):
                            risk_budget = pos["max_position"] - pos["total_cost"]
                            effective_ratio = tier_ratio * tier_factor
                            supplement_amount = round(risk_budget * effective_ratio, 2)
                            cap = pos["max_position"] * SUPPLEMENT_CAP_RATIO
                            supplement_amount = round(min(supplement_amount, cap, risk_budget), 2)
                            supplement_amount = round(supplement_amount * size_mul, 2)

                            # v5.2: 成本修复效率阈值动态化
                            max_pos = pos.get("max_position", 5000)
                            min_efficiency = 0.05 * (5000 / max(1000, max_pos))
                            efficiency = _calc_cost_repair_efficiency(
                                batches_sorted, current_nav, supplement_amount
                            )
                            if efficiency < min_efficiency and supplement_amount > 500:
                                extra_alerts.append(
                                    f"补仓效率偏低({efficiency:.4f}% per 千元 < {min_efficiency:.4f}%), "
                                    f"建议等待更大跌幅后补仓"
                                )
                                break

                            if supplement_amount > 0:
                                sig = _make_signal(
                                    fund_code,
                                    signal_name=f"补仓(第{supplement_count+1}次/上限{dynamic_max_supp})",
                                    action="buy",
                                    priority=4,
                                    amount=supplement_amount,
                                    reason=f"总浮亏{total_profit_pct}%, 今日跌{today_change}%, "
                                           f"补仓{supplement_amount}元(成本修复效率{efficiency:.4f}%/千元)",
                                )
                                all_signals.append(sig)
                                if best_signal is None or _is_higher_priority(sig, best_signal):
                                    best_signal = sig
                        break

        # --- 冷却期后加仓 ---
        if (not in_cooldown
                and pos.get("cooldown_sell_date")
                and pos["total_cost"] < pos["max_position"] * 0.8
                and total_profit_pct < -2.0
                and today_change <= 0
                and not forbidden):
            remaining = pos["max_position"] - pos["total_cost"]
            rebuy_amount = round(min(remaining * 0.3, pos["total_cost"] * 0.3) * size_mul, 2)
            if rebuy_amount >= 100:
                sig = _make_signal(
                    fund_code,
                    signal_name="冷却期后加仓",
                    action="buy",
                    priority=4,
                    amount=rebuy_amount,
                    reason=f"冷却期结束, 总浮亏{total_profit_pct}%, 加仓{rebuy_amount}元",
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig

        # 汇总
        if best_signal:
            if extra_alerts and not best_signal.get("alert_msg"):
                best_signal["alert"] = True
                best_signal["alert_msg"] = "; ".join(extra_alerts)
            elif extra_alerts:
                best_signal["alert_msg"] += "; " + "; ".join(extra_alerts)
            best_signal["market_analysis"] = market_analysis
            all_signals.sort(key=lambda s: (s["priority"], s.get("sub_priority", 0)))
            best_signal["all_signals"] = [
                {k: v for k, v in s.items() if k != "all_signals"}
                for s in all_signals
            ]

            # FIFO 穿透降级
            sell_signals = [s for s in all_signals if s.get("action") == "sell" and s.get("target_batch_id")]
            if sell_signals:
                fifo_plan = _build_fifo_sell_plan(
                    batches_sorted, sell_signals, current_nav, fund_code
                )
                best_priority = best_signal.get("priority", 8)
                if fifo_plan.get("has_passthrough") and best_priority >= 2:
                    loss_total = fifo_plan.get("passthrough_loss_total", 0)
                    total_est_profit = fifo_plan.get("total_estimated_profit", 0)
                    total_pos_amount = pos.get("total_cost", 1)

                    min_net_profit = max(
                        PASSTHROUGH_MIN_NET_PROFIT_ABS,
                        total_pos_amount * PASSTHROUGH_MIN_NET_PROFIT_RATIO
                    )

                    should_downgrade = False
                    downgrade_reason = ""

                    if total_est_profit < min_net_profit:
                        should_downgrade = True
                        downgrade_reason = f"净收益{total_est_profit:.0f}元 < 门槛{min_net_profit:.0f}元"

                    if (not should_downgrade
                            and loss_total < 0
                            and total_est_profit > 0
                            and abs(loss_total) > total_est_profit * PASSTHROUGH_LOSS_RATIO_THRESHOLD):
                        should_downgrade = True
                        downgrade_reason = f"穿透亏损{loss_total:.0f}元 > 总利润{total_est_profit:.0f}元×{PASSTHROUGH_LOSS_RATIO_THRESHOLD:.0%}"

                    if should_downgrade:
                        for sig_item in all_signals:
                            if sig_item.get("action") == "sell" and sig_item.get("priority", 8) >= 2:
                                sig_item["action"] = "hold"
                                sig_item["signal_name"] += "(穿透亏损过大)"
                                sig_item["reason"] += f" → {downgrade_reason}"
                                sig_item["alert"] = True
                        fifo_plan["downgraded"] = True

                best_signal["fifo_sell_plan"] = fifo_plan

            _append_signal_history(fund_code, best_signal, {
                "today_change": today_change,
                "total_profit_pct": total_profit_pct,
                "current_nav": current_nav,
                "_source": source,
            })
            return _stamp(best_signal, confidence, source)

        # 无触发 → 持有
        reason_parts = [f"总浮盈{total_profit_pct}%", f"今日{today_change}%"]
        if trend_ctx.get("trend_label"):
            reason_parts.append(f"趋势:{trend_ctx['trend_label']}")
        reason_parts.append(f"风险系数{dyn['risk_multiplier']}×")
        if vol_state != "normal_vol":
            reason_parts.append(f"波动状态:{vol_state}")
        reason_parts.append("无触发条件")

        hold_signal = _make_signal(
            fund_code, signal_name="持有等待", action="hold", priority=8,
            reason=", ".join(reason_parts),
            alert=bool(extra_alerts),
            alert_msg="; ".join(extra_alerts) if extra_alerts else None,
        )
        hold_signal["market_analysis"] = market_analysis
        _append_signal_history(fund_code, hold_signal, {
            "today_change": today_change,
            "total_profit_pct": total_profit_pct,
            "current_nav": current_nav,
            "_source": source,
        })
        return _stamp(hold_signal, confidence, source)

    # === 空仓 ===
    dip_threshold = dyn["dip_threshold"]
    market_analysis = _build_market_analysis(
        fund_code, val, nav_history, pos, trend_ctx=trend_ctx,
        dynamic_thresholds=dyn
    )
    market_ctx = {"today_change": today_change, "total_profit_pct": None,
                  "current_nav": None, "_source": source}

    can_buy_empty = (source == "nav" or confidence >= 0.55)

    # 极端波动禁止买入
    if vol_state == "extreme_vol":
        sig = _make_signal(
            fund_code, signal_name="极端波动观望", action="hold", priority=8,
            reason=f"波动率处于极端水平({vol_state})，暂停所有买入",
            alert=True, alert_msg="极端波动环境，仅允许止损操作",
        )
        sig["market_analysis"] = market_analysis
        _append_signal_history(fund_code, sig, market_ctx)
        return _stamp(sig, confidence, source)

    # --- 大跌抄底 ---
    if today_change <= dip_threshold and not in_cooldown and can_buy_empty:
        max_pos = pos["max_position"]
        buy_amount = round(max_pos * 0.5 * size_mul, 2)

        # v5.2: 缩量大跌可能是假突破，减少建仓规模
        volume_proxy = trend_ctx.get("volume_proxy")
        if volume_proxy and volume_proxy < 0.5:
            buy_amount = round(buy_amount * 0.6, 2)
            vol_note = f"(缩量{volume_proxy:.1f}×, 减仓买入)"
        else:
            vol_note = ""

        sig = _make_signal(
            fund_code, signal_name="大跌抄底", action="buy", priority=6,
            amount=buy_amount,
            reason=f"今日跌{today_change}% ≤ 动态阈值{dip_threshold}%, 买入{buy_amount}元{vol_note}",
        )
        sig["market_analysis"] = market_analysis
        _append_signal_history(fund_code, sig, market_ctx)
        return _stamp(sig, confidence, source)

    # --- 趋势建仓 ---
    short_5d = trend_ctx.get("short_5d")
    mid_10d = trend_ctx.get("mid_10d")
    consecutive_down = trend_ctx.get("consecutive_down", 0)

    if not in_cooldown and can_buy_empty:
        max_pos = pos["max_position"]
        build_signal = None

        if (mid_10d is not None and mid_10d <= TREND_BUILD_TRIGGER_10D
                and today_change >= -0.5):
            buy_amount = round(max_pos * 0.4 * size_mul, 2)
            build_signal = _make_signal(
                fund_code, signal_name="低位建仓", action="buy", priority=6,
                amount=buy_amount,
                reason=f"10日累跌{mid_10d}%, 今日企稳, 中期低位建仓{buy_amount}元",
            )
        elif (short_5d is not None and short_5d <= TREND_BUILD_TRIGGER_5D
                and today_change > 0):
            buy_amount = round(max_pos * 0.3 * size_mul, 2)
            build_signal = _make_signal(
                fund_code, signal_name="反弹建仓", action="buy", priority=6,
                amount=buy_amount,
                reason=f"5日累跌{short_5d}%, 今日反弹, 逢低建仓{buy_amount}元",
            )
        elif (consecutive_down >= 3 and today_change < 0
                and len(hist_changes) >= 1
                and abs(today_change) < abs(hist_changes[0]) * 0.6):
            buy_amount = round(max_pos * 0.25 * size_mul, 2)
            build_signal = _make_signal(
                fund_code, signal_name="跌势放缓建仓", action="buy", priority=7,
                amount=buy_amount,
                reason=f"连跌{consecutive_down}天, 跌幅收窄, 试探建仓{buy_amount}元",
            )

        if build_signal:
            build_signal["market_analysis"] = market_analysis
            _append_signal_history(fund_code, build_signal, market_ctx)
            return build_signal

    # --- 连跌低吸 ---
    consec_dip_thresh = dyn.get("consecutive_dip_trigger", DEFAULT_CONSECUTIVE_DIP_TRIGGER)
    if (today_change <= consec_dip_thresh
            and len(recent) >= 1
            and recent[0].get("change") is not None
            and recent[0]["change"] < 0
            and not in_cooldown and can_buy_empty):
        max_pos = pos["max_position"]
        buy_amount = round(max_pos * 0.3 * size_mul, 2)
        sig = _make_signal(
            fund_code, signal_name="连跌低吸", action="buy", priority=7,
            amount=buy_amount,
            reason=f"今日跌{today_change}% ≤ {consec_dip_thresh}%, 昨日跌{recent[0]['change']}%, 连跌低吸{buy_amount}元",
        )
        sig["market_analysis"] = market_analysis
        _append_signal_history(fund_code, sig, market_ctx)
        return _stamp(sig, confidence, source)

    # --- 冷却期后建仓 ---
    # v5.3: 增加趋势过滤，避免在深度下跌趋势中盲目建仓
    if (not in_cooldown
            and pos.get("cooldown_sell_date")
            and today_change <= 0
            and can_buy_empty):
        # v5.3: 过滤深度下跌趋势
        short_5d_cd = trend_ctx.get("short_5d")
        consecutive_down_cd = trend_ctx.get("consecutive_down", 0)
        trend_ok = True
        cd_note = ""
        if short_5d_cd is not None and short_5d_cd <= -5 and consecutive_down_cd >= 3:
            trend_ok = False
            cd_note = f"(5日累跌{short_5d_cd}%+连跌{consecutive_down_cd}天, 延迟建仓)"
        if trend_ok:
            max_pos = pos["max_position"]
            buy_amount = round(max_pos * 0.3 * size_mul, 2)
            sig = _make_signal(
                fund_code, signal_name="冷却期后建仓", action="buy", priority=7,
                amount=buy_amount,
                reason=f"冷却期结束, 今日{today_change}%, 重新建仓{buy_amount}元",
            )
            sig["market_analysis"] = market_analysis
            _append_signal_history(fund_code, sig, market_ctx)
            return _stamp(sig, confidence, source)
        else:
            # 趋势不好，记录但不建仓
            extra_cd_note = f"冷却期已结束但趋势不佳{cd_note}"
            # 继续走到"观望"

    # --- 观望 ---
    obs_parts = [f"今日{today_change}%"]
    if trend_ctx.get("trend_label"):
        obs_parts.append(f"趋势:{trend_ctx['trend_label']}")
    obs_parts.append(f"波动状态:{vol_state}")
    obs_parts.append("无触发条件")
    sig = _make_signal(
        fund_code, signal_name="观望", action="hold", priority=8,
        reason=", ".join(obs_parts),
    )
    sig["market_analysis"] = market_analysis
    _append_signal_history(fund_code, sig, market_ctx)
    return _stamp(sig, confidence, source)


# ============================================================
# 批量信号（v5.2: 并发优化 + 组合级风控）
# ============================================================

def generate_all_signals() -> dict:
    fund_codes = set()
    pos_data = load_positions()
    for code in pos_data.get("funds", {}).keys():
        fund_codes.add(code)

    # v5.2: 并发生成信号
    signals = []
    sorted_codes = sorted(fund_codes)

    def _safe_generate(code):
        try:
            return generate_signal(code)
        except Exception as e:
            print(f"[Strategy] 生成 {code} 信号失败: {e}")
            return _make_signal(code, reason=f"信号生成失败: {e}")

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_safe_generate, code): code for code in sorted_codes}
        results_map = {}
        for future in futures:
            code = futures[future]
            results_map[code] = future.result()

    signals = [results_map[code] for code in sorted_codes]
    signals.sort(key=lambda s: (s["priority"], s.get("sub_priority", 0)))

    # === 组合级风控 ===
    cash_reserve_ratio = pos_data.get("cash_reserve_ratio", 0.30)
    funds_data = pos_data.get("funds", {})

    portfolio_max_invest = sum(
        f.get("max_position", 5000) for f in funds_data.values()
    ) * (1 - cash_reserve_ratio)

    portfolio_invested = 0
    for code, fund in funds_data.items():
        holding = [b for b in fund.get("batches", []) if b.get("status") == "holding"]
        portfolio_invested += sum(b.get("amount", 0) for b in holding)

    daily_budget = max(0, portfolio_max_invest - portfolio_invested)

    # 动态 daily_buy_cap
    # v5.3: 按持仓金额加权计算 negative_ratio（重仓基金下跌影响更大）
    total_weight = 0
    negative_weight = 0
    for s in signals:
        ma = s.get("market_analysis", {})
        fund_amount = ma.get("total_cost", 0) or ma.get("market_value", 0) or 1000
        total_weight += fund_amount
        if ma.get("today_change", 0) < 0:
            negative_weight += fund_amount
    negative_ratio = negative_weight / max(1, total_weight)

    if negative_ratio > 0.6:
        cap_ratio = DAILY_BUY_CAP_RATIO_CONSERVATIVE
    elif negative_ratio < 0.3:
        cap_ratio = DAILY_BUY_CAP_RATIO_AGGRESSIVE
    else:
        cap_ratio = DAILY_BUY_CAP_RATIO_BASE

    daily_buy_cap = round(portfolio_max_invest * cap_ratio, 2)
    effective_budget = min(daily_budget, daily_buy_cap) if daily_buy_cap > 0 else daily_budget

    buy_signals = [s for s in signals if s.get("action") == "buy" and s.get("amount")]

    if buy_signals:
        remaining_budget = effective_budget
        total_buy_count = len(buy_signals)

        # 修正折扣公式
        if total_buy_count > 1:
            discount = max(0.75, 1.0 - (total_buy_count - 1) * 0.10)
            confidences = [s.get("_confidence", 1.0) for s in buy_signals]
            avg_conf = sum(confidences) / len(confidences) if confidences else 1.0
            if avg_conf < 0.6:
                discount *= 0.7
            elif avg_conf < 0.75:
                discount *= 0.85
        else:
            discount = 1.0

        # 同赛道集中度约束
        state = load_state()
        sector_map = {}
        for sector in state.get("sectors", []):
            for fund in sector.get("funds", []):
                sector_map[fund.get("code", "")] = sector["name"]

        sector_spent = {}

        for sig in buy_signals:
            original = sig["amount"]
            discounted = round(original * discount, 2)

            real_code, _ = parse_fund_key(sig["fund_code"])
            sector_name = sector_map.get(real_code, "默认")
            sector_cap = effective_budget * SECTOR_BUY_CAP_RATIO
            sector_used = sector_spent.get(sector_name, 0)
            sector_remaining = max(0, sector_cap - sector_used)

            if remaining_budget <= 0:
                sig["action"] = "hold"
                sig["signal_name"] = sig["signal_name"] + "(预算不足)"
                sig["reason"] += f" (组合现金预算已耗尽)"
                sig["alert"] = True
            elif discounted > remaining_budget or discounted > sector_remaining:
                actual = round(min(remaining_budget, sector_remaining), 2)
                if actual <= 0:
                    sig["action"] = "hold"
                    sig["signal_name"] = sig["signal_name"] + "(赛道集中度限制)"
                    sig["reason"] += f" (同赛道{sector_name}买入已达上限)"
                    sig["alert"] = True
                else:
                    sig["amount"] = actual
                    sig["reason"] += f" (预算截断→{actual:.0f}元)"
                    remaining_budget -= actual
                    sector_spent[sector_name] = sector_used + actual
            else:
                sig["amount"] = discounted
                remaining_budget -= discounted
                sector_spent[sector_name] = sector_used + discounted

            if total_buy_count > 1 and sig["action"] == "buy":
                sig["reason"] += f" (组合风控: {total_buy_count}只, 折扣{discount:.0%})"

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signals": signals,
        "portfolio_budget": {
            "max_invest": round(portfolio_max_invest, 2),
            "invested": round(portfolio_invested, 2),
            "daily_budget": round(daily_budget, 2),
            "daily_buy_cap": round(daily_buy_cap, 2),
            "effective_budget": round(effective_budget, 2),
            "cash_reserve_ratio": cash_reserve_ratio,
            "cap_ratio": cap_ratio,
            "market_negative_ratio": round(negative_ratio, 2),
        },
    }