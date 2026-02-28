"""
grid/engine.py - 低频网格策略：信号引擎

包含: 决策备注 | 市场分析 | 趋势分析 | generate_signal | generate_all_signals
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from positions import (
    get_fund_position, get_sell_fee_rate, load_positions, parse_fund_key
)
from valuation.core import calculate_valuation, load_state
from valuation.providers import get_fund_nav_history

from .config import (
    _append_signal_history, _resolve_regime, _get_regime_params,
    get_fitness_scores, calc_signal_win_rate,
    # 常量
    DEFAULT_DIP_THRESHOLD, DEFAULT_TAKE_PROFIT_TRIGGER,
    DEFAULT_STOP_LOSS_BASE, DEFAULT_SUPPLEMENT_TRIGGER,
    DEFAULT_SUPPLEMENT_LOSS_MIN, DEFAULT_CONSECUTIVE_DIP_TRIGGER,
    DEFAULT_TREND_WEAK_CUMULATIVE, DEFAULT_DISASTER_LOSS,
    DEFAULT_DISASTER_DAILY_DROP, DEFAULT_VOL_SENSITIVITY,
    SUPPLEMENT_MAX_COUNT_DEFAULT, SUPPLEMENT_TIERS, SUPPLEMENT_CAP_RATIO,
    SUPPLEMENT_REBUY_STEP_PCT, TOTAL_PROFIT_SELL_TIERS,
    DAILY_BUY_CAP_RATIO_BASE, DAILY_BUY_CAP_RATIO_CONSERVATIVE,
    DAILY_BUY_CAP_RATIO_AGGRESSIVE,
    SECTOR_BUY_CAP_RATIO,
    PASSTHROUGH_LOSS_RATIO_THRESHOLD, PASSTHROUGH_MIN_NET_PROFIT_ABS,
    PASSTHROUGH_MIN_NET_PROFIT_RATIO,
    DISASTER_CONSECUTIVE_DOWN, DISASTER_SELL_PCT_DAILY, DISASTER_SELL_PCT_EXTREME,
    TRAIL_DD_BASE,
)
from .helpers import (
    _vol_adaptive_thresholds, _classify_volatility,
    _calc_momentum_score, _calc_risk_multiplier, _calc_dynamic_thresholds,
    _calc_sell_score, _calc_cost_repair_efficiency, _calc_dynamic_supplement_max,
    _evaluate_stop_loss,
    _is_supplement_forbidden, _check_supplement_rate_limit,
    _count_trade_days_between, _is_in_cooldown, _calc_size_multiplier,
    _estimate_current_nav, _calc_batch_profit_pct, _calc_total_profit_pct,
    _get_take_profit_sell_pct, _get_slow_profit_sell_pct,
    _calc_min_profit_buffer,
    _get_trail_profit_sell_pct, _calc_peak_profit, _update_batch_peak_nav,
    _build_fifo_sell_plan, _make_signal, _is_higher_priority, _stamp,
)

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
                           dynamic_thresholds: dict = None,
                           regime: str = None,
                           regime_source: str = "auto") -> dict:
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
        "market_regime": regime or "neutral",  # v5.13: 当前行情模式
        "regime_source": regime_source,  # v5.13: 模式来源 (manual/auto)
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

    # === v5.13: 行情模式参数注入 ===
    regime = _resolve_regime(trend_ctx)
    regime_params = _get_regime_params(regime)
    # 判断模式来源
    try:
        _pos_global = load_positions()
        _regime_source = "manual" if _pos_global.get("regime_manual") else "auto"
    except Exception:
        _regime_source = "auto"

    signal_stats = calc_signal_win_rate(fund_code)

    dyn = _calc_dynamic_thresholds(trend_ctx, fund_code, confidence, source,
                                    signal_stats=signal_stats)

    # v5.13: 行情模式覆盖补仓档位首次建仓比例
    if regime != "neutral":
        _regime_first_ratio = regime_params["first_build_ratio"]
        _adj_tiers = dyn.get("supplement_tiers", [])
        if _adj_tiers:
            _adj_tiers[0] = (_adj_tiers[0][0], _regime_first_ratio, _adj_tiers[0][2], _adj_tiers[0][3])
            dyn["supplement_tiers"] = _adj_tiers

    vol_state = dyn["vol_state"]
    momentum = dyn.get("momentum_score", 0)

    in_cooldown = _is_in_cooldown(pos, nav_history)

    size_mul = _calc_size_multiplier(
        dyn["risk_multiplier"], confidence,
        trend_ctx.get("trend_label", "震荡"),
        momentum_score=momentum
    )
    # v5.13: 行情模式覆盖size_mul上限
    size_mul = min(size_mul, regime_params["size_mul_cap"])

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
            trend_ctx=trend_ctx, dynamic_thresholds=dyn,
            regime=regime, regime_source=_regime_source
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
                supplement_count=supplement_count,
                today_change=today_change,  # v5.8 重构P0: 传入当日跌幅用于L3分级判断
                l2_sell_pct_base=regime_params["l2_stop_loss_base"]  # v5.13: 行情模式覆盖
            )

            if stop_eval["level"] == "L3":
                l3_sell_pct = stop_eval["sell_pct"]  # v5.8 重构P0: 使用分级减仓比例（70%或100%）
                l3_sell_shares = round(batch["shares"] * l3_sell_pct / 100, 2)
                sig = _make_signal(
                    fund_code,
                    signal_name="极端止损(L3)",
                    action="sell",
                    priority=1,
                    target_batch_id=batch["id"],
                    sell_shares=l3_sell_shares,
                    sell_pct=l3_sell_pct,
                    reason=stop_eval["reason"],
                    fee_info={
                        "sell_fee_rate": fee_rate,
                        "estimated_fee": round(l3_sell_shares * current_nav * fee_rate / 100, 2),
                        "estimated_net_profit": round(l3_sell_shares * current_nav * (1 - fee_rate / 100) - batch["amount"] * l3_sell_pct / 100, 2),
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

                # v5.5 优化: 短期深亏安全网门槛从-6%→-8%，减少过度敏感
                if not disaster_triggered and profit_pct <= -8.0:
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

            # v5.10 重构P0: 扭亏奖励分——将扭亏止盈纳入评分系统
            # 当total_profit_pct>0且≥2批时，为最老批次增加"扭亏奖励分"
            # 这样扭亏止盈必须通过评分门槛才能触发，受上升趋势抑制等规则约束
            # 彻底消除独立于评分系统的"逃逸通道"
            _nz_bonus = 0.0
            if (total_profit_pct > 0 and len(batches_sorted) >= 2
                    and batch["id"] == batches_sorted[0]["id"]):  # 仅最老批次获得扭亏加分
                # v5.11 重构P1: 降低nz_bonus——v5.10的min(20,tpp*3)过于慷慨导致扭亏18362次
                # 降低后: min(12,tpp*2)，需要更高浮盈才能凑够评分门槛
                _nz_bonus = min(12.0, total_profit_pct * 2.0)
                # 持仓时间修正: 最老批次持仓<15天时减半奖励（与v5.7时间门槛对齐）
                _oldest_bd = datetime.strptime(batches_sorted[0]["buy_date"], "%Y-%m-%d").date()
                _oldest_hd = (datetime.now().date() - _oldest_bd).days
                if _oldest_hd < 15:
                    _nz_bonus *= 0.5

            sell_eval = _calc_sell_score(
                batch, current_nav, today_change, trend_ctx, dyn,
                fee_rate, hold_days, peak_profit,
                nz_bonus=_nz_bonus,  # v5.10 P0: 传入扭亏奖励分
                tp_suppress_threshold=regime_params["tp_suppress_threshold"]  # v5.13: 行情模式覆盖
            )

            if sell_eval["sell_pct"] > 0 and sell_eval["signal_name"]:
                sell_pct = sell_eval["sell_pct"]

                # v5.10 重构P0: 扭亏止盈走评分通道出去时，标记信号名称
                # 便于回测统计区分普通止盈和扭亏驱动的止盈
                _effective_signal_name = sell_eval["signal_name"]
                if _nz_bonus > 0 and batch["id"] == batches_sorted[0]["id"]:
                    _effective_signal_name = f"扭亏{sell_eval['signal_name']}"

                sell_shares = round(batch["shares"] * sell_pct / 100, 2)
                est_gross = sell_shares * current_nav
                est_fee = round(est_gross * fee_rate / 100, 2)
                est_net_profit = round(est_gross * (1 - fee_rate / 100) - batch["amount"] * sell_pct / 100, 2)

                is_low_conf = source == "estimation" and confidence < 0.5
                # 同日卖出抑制：今天已卖过则降级为 hold，避免重复建议
                is_suppressed = _sold_today
                sig = _make_signal(
                    fund_code,
                    signal_name=_effective_signal_name + ("(今日已操作)" if is_suppressed else "(待确认)" if is_low_conf else ""),
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
            # v5.5 优化: 趋势转弱需要更强确认——需3天连跌（原2天）+5日/10日均为负
            # 回测显示趋势转弱信号质量一般，需要更严格过滤
            if (len(recent) >= 3
                    and all(recent[i].get("change") is not None and recent[i]["change"] < 0 for i in range(3))):
                cumulative_drop = sum(recent[i]["change"] for i in range(3))
                trend_weak_thresh = dyn.get("trend_weak_cumulative", DEFAULT_TREND_WEAK_CUMULATIVE) * 1.5  # v5.5: 门槛提高50%
                if cumulative_drop <= trend_weak_thresh:
                    if ((short_5d_val is not None and short_5d_val < -2)  # v5.5: 原<0→<-2
                            or (mid_10d_val is not None and mid_10d_val < -3)):  # v5.5: 原<0→<-3
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
                # v5.5 优化: 趋势转弱减仓比例整体降低——留更多仓位等反弹
                # 回测显示频繁清仓是牛市捕获率低的主因
                if total_profit_pct < 1:
                    trend_sell_pct = 70   # v5.5: 原100→70，薄利也留30%底仓
                elif total_profit_pct < 3:
                    trend_sell_pct = 50   # v5.5: 原70→50
                else:
                    trend_sell_pct = 30   # v5.5: 原50→30，厚利更要留仓位

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

        # --- v5.10 重构P0: 扭亏止盈已纳入评分系统，不再独立触发 ---
        # 旧逻辑(v5.4~v5.9): total_profit_pct>0 + ≥2批 → 独立触发扭亏止盈
        # 这个逻辑是六版以来反复出现的"水床效应"根源——每次压制评分止盈，
        # 都导致更多交易走扭亏通道逃逸(v5.9: 12277次占TOP1)
        # v5.10: 扭亏止盈改为评分系统的"nz_bonus加分项"，必须通过评分门槛才能卖出
        # 这样上升趋势抑制(<65)、持仓不足15日等规则全部对扭亏生效

        # --- 总仓位风控（v5.6 重构: 废除固定止损线死循环，改为"持仓时间豁免+趋势确认减仓+灾难保底"三层架构） ---
        if total_profit_pct < 0 and not best_signal:
            oldest = batches_sorted[0]
            oldest_bd = datetime.strptime(oldest["buy_date"], "%Y-%m-%d").date()
            oldest_hd = (datetime.now().date() - oldest_bd).days
            supp_count = pos.get("supplement_count", 0)
            dyn_max_supp = _calc_dynamic_supplement_max(pos)
            dyn_max_supp = min(dyn_max_supp, regime_params["supplement_max_count"])  # v5.13: 行情模式覆盖

            # v5.6 重构: 第一层——灾难保底（total_profit_pct <= -15%无条件减仓，不受时间/趋势限制）
            # 原因: 真正的灾难不能等，但-15%在正常波动中几乎不会触发，消除死循环
            if total_profit_pct <= -15.0 and oldest_hd >= 7:
                oldest_fr = get_sell_fee_rate(fund_code, oldest_hd)
                # 灾难级别：按严重程度分级减仓
                if total_profit_pct <= -25.0:
                    portfolio_sell_pct = 70  # v5.6: 极端灾难
                elif total_profit_pct <= -20.0:
                    portfolio_sell_pct = 50  # v5.6: 严重灾难
                else:
                    portfolio_sell_pct = 35  # v5.6: 灾难保底
                sell_shares = round(oldest["shares"] * portfolio_sell_pct / 100, 2)
                _psl_suppressed = _sold_today
                sig = _make_signal(
                    fund_code,
                    signal_name="灾难保底减仓" + ("(今日已操作)" if _psl_suppressed else ""),
                    action="hold" if _psl_suppressed else "sell",
                    priority=1,
                    sub_priority=2,
                    target_batch_id=oldest["id"],
                    sell_shares=sell_shares,
                    sell_pct=portfolio_sell_pct,
                    reason=(f"总浮亏{total_profit_pct}% ≤ -15%灾难线, "
                            f"最老批次减仓{portfolio_sell_pct}%"
                            + (f" (今日已执行卖出，次日再评估)" if _psl_suppressed else "")),
                    fee_info={
                        "sell_fee_rate": oldest_fr,
                        "estimated_fee": round(sell_shares * current_nav * oldest_fr / 100, 2),
                        "estimated_net_profit": round(sell_shares * current_nav * (1 - oldest_fr / 100) - oldest["amount"] * portfolio_sell_pct / 100, 2),
                    },
                    alert=True,
                    alert_msg=f"总仓位浮亏{total_profit_pct}%触发灾难保底减仓",
                )
                all_signals.append(sig)
                if best_signal is None or _is_higher_priority(sig, best_signal):
                    best_signal = sig

            # v5.6 重构: 第二层——持仓时间豁免（最老批次<20天完全不触发总仓位减仓）
            # 原因: 新建仓需要时间度过正常波动，<20天内的浮亏是网格策略的正常成本
            elif oldest_hd < 20:
                # 仅记录预警，不触发卖出
                if total_profit_pct <= -8.0:
                    extra_alerts.append(
                        f"总仓位浮亏{total_profit_pct}%，但最老批次仅持有{oldest_hd}天(<20天豁免期)，暂不减仓"
                    )

            # v5.6 重构: 第三层——趋势确认减仓（持仓≥20天后，需要趋势恶化双确认才减仓）
            # 原因: 不再用固定止损线，改用趋势判断——只有确认趋势转坏才减仓，避免正常波动反复触发
            elif oldest_hd >= 20:
                trend_label = trend_ctx.get("trend_label", "震荡")
                consecutive_down = trend_ctx.get("consecutive_down", 0)
                short_5d_val = trend_ctx.get("short_5d")
                mid_10d_val = trend_ctx.get("mid_10d")
                long_20d_val = trend_ctx.get("long_20d")

                # v5.6 重构: 趋势恶化需要多重确认，避免单一指标误判
                trend_deteriorating = False
                trend_confirm_reasons = []

                # 条件A: 连跌确认（≥6天连续下跌，说明不是正常波动而是趋势性下跌）
                if consecutive_down >= 6:  # v5.13: 5→6，收紧确认门槛
                    trend_confirm_reasons.append(f"连跌{consecutive_down}天")

                # 条件B: 中期走弱确认（10日累跌≥-10%，说明中期趋势已转坏）
                if mid_10d_val is not None and mid_10d_val <= -10:  # v5.13: -8→-10
                    trend_confirm_reasons.append(f"10日累跌{mid_10d_val}%")

                # 条件C: 趋势标签确认（趋势分析引擎判定为"连跌"或"中期走弱"）
                if trend_label in ("连跌", "中期走弱"):
                    trend_confirm_reasons.append(f"趋势:{trend_label}")

                # v5.6 重构: 至少满足2个条件才确认趋势恶化（双确认机制）
                if len(trend_confirm_reasons) >= 2:
                    trend_deteriorating = True

                # v5.6 重构: 补仓已用尽仍深亏，额外降低确认门槛（说明"越跌越买"策略已失败）
                remaining_supp = max(0, dyn_max_supp - supp_count)
                if remaining_supp == 0 and total_profit_pct <= -8.0 and len(trend_confirm_reasons) >= 1:
                    trend_deteriorating = True
                    trend_confirm_reasons.append(f"补仓{supp_count}/{dyn_max_supp}已用尽")

                if trend_deteriorating and oldest_hd >= 7:
                    oldest_fr = get_sell_fee_rate(fund_code, oldest_hd)
                    # v5.6 重构: 按亏损严重程度分级减仓
                    if total_profit_pct <= -12.0:
                        portfolio_sell_pct = 50  # 深亏+趋势恶化
                    elif total_profit_pct <= -8.0:
                        portfolio_sell_pct = 35  # 中度亏损+趋势恶化
                    elif total_profit_pct <= -6.0:  # v5.13: -5→-6，收紧浅亏减仓
                        portfolio_sell_pct = 25  # 轻度亏损+趋势恶化
                    else:
                        portfolio_sell_pct = 0   # 浅亏不减仓，等趋势进一步确认
                    if portfolio_sell_pct > 0:
                        sell_shares = round(oldest["shares"] * portfolio_sell_pct / 100, 2)
                        _psl_suppressed = _sold_today
                        sig = _make_signal(
                            fund_code,
                            signal_name="趋势确认减仓" + ("(今日已操作)" if _psl_suppressed else ""),
                            action="hold" if _psl_suppressed else "sell",
                            priority=1,
                            sub_priority=2,
                            target_batch_id=oldest["id"],
                            sell_shares=sell_shares,
                            sell_pct=portfolio_sell_pct,
                            reason=(f"总浮亏{total_profit_pct}%, 持仓{oldest_hd}天, "
                                    f"趋势恶化确认({'+'.join(trend_confirm_reasons)}), "
                                    f"最老批次减仓{portfolio_sell_pct}%"
                                    + (f" (今日已执行卖出，次日再评估)" if _psl_suppressed else "")),
                            fee_info={
                                "sell_fee_rate": oldest_fr,
                                "estimated_fee": round(sell_shares * current_nav * oldest_fr / 100, 2),
                                "estimated_net_profit": round(sell_shares * current_nav * (1 - oldest_fr / 100) - oldest["amount"] * portfolio_sell_pct / 100, 2),
                            },
                            alert=True,
                            alert_msg=f"总仓位浮亏{total_profit_pct}%+趋势恶化，确认减仓",
                        )
                        all_signals.append(sig)
                        if best_signal is None or _is_higher_priority(sig, best_signal):
                            best_signal = sig
                elif total_profit_pct <= -8.0:
                    # v5.6 重构: 深亏但趋势未完全确认恶化，仅预警
                    extra_alerts.append(
                        f"总仓位浮亏{total_profit_pct}%, 持仓{oldest_hd}天, "
                        f"趋势信号({','.join(trend_confirm_reasons) if trend_confirm_reasons else '无'}), 暂未触发减仓"
                    )

        # --- 递进补仓 ---
        dynamic_max_supp = _calc_dynamic_supplement_max(pos)
        dynamic_max_supp = min(dynamic_max_supp, regime_params["supplement_max_count"])  # v5.13: 行情模式覆盖
        forbidden, forbid_reason = _is_supplement_forbidden(
            trend_ctx, confidence, source, vol_state,
            batches_sorted=batches_sorted  # v5.7: 传入批次信息用于豁免期判断
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
                            # v5.8 重构P3: 效率门槛降低50%（0.05→0.025），深亏时跳过效率检查
                            max_pos = pos.get("max_position", 5000)
                            min_efficiency = 0.025 * (5000 / max(1000, max_pos))  # v5.8 重构P3: 0.05→0.025
                            efficiency = _calc_cost_repair_efficiency(
                                batches_sorted, current_nav, supplement_amount
                            )
                            # v5.8 重构P3: 深亏时(total_profit_pct<=-5%)跳过效率检查，直接补仓
                            if efficiency < min_efficiency and supplement_amount > 500 and total_profit_pct > -5.0:
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
                and today_change <= 0.3  # v5.13: 从<=0放宽到<=0.3
                and not forbidden):
            remaining = pos["max_position"] - pos["total_cost"]
            rebuy_amount = round(min(remaining * 0.5, pos["total_cost"] * 0.5) * size_mul, 2)  # v5.5 优化: 加仓比例从30%→50%
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

            # v5.11 重构P0: 延迟回补——止盈后不立即买入，记录回补触发价
            # v5.10直接回补导致: 买在止盈价(无安全边际)→正常波动触发L2→中位持仓3.5天
            # v5.11改为: 记录rebuy_trigger_nav，后续交易日净值跌到此价格才回补
            # v5.11 重构P3: 撤回弱趋势回补（下跌趋势中买入注定被止损）
            _rebuy_signal_names = {"分批止盈", "慢涨止盈", "止盈卖出", "强势止盈",
                                   "扭亏分批止盈", "扭亏慢涨止盈", "扭亏止盈卖出", "扭亏强势止盈"}
            if (best_signal.get("action") == "sell"
                    and any(best_signal.get("signal_name", "").startswith(sn) for sn in _rebuy_signal_names)):
                _rebuy_trend = trend_ctx.get("trend_label", "震荡")
                _rebuy_ratio = 0.0
                _rebuy_discount = 0.0  # 回补折价（安全边际）
                _regime_rebuy_discount = regime_params["rebuy_discount"]  # v5.13: 行情模式覆盖
                if _regime_rebuy_discount <= 0:
                    pass  # v5.13: 熊市模式不回补
                elif _rebuy_trend in ("连涨", "偏强", "中期走强"):
                    _rebuy_ratio = 0.70   # v5.13: 60%→70%，强趋势更积极回补
                    _rebuy_discount = _regime_rebuy_discount  # v5.13: 由行情模式控制
                elif _rebuy_trend in ("震荡",) and today_change >= -0.5:
                    _rebuy_ratio = 0.40   # v5.13: 35%→40%
                    _rebuy_discount = _regime_rebuy_discount + 0.010  # v5.13: 中性加0.01安全边际
                # v5.11 P3: 弱趋势不回补（撤回v5.10的20%，下跌趋势中买入注定被止损）

                if _rebuy_ratio > 0:
                    _sell_amount_est = best_signal.get("sell_shares", 0) * current_nav if current_nav else 0
                    _rebuy_amount = round(_sell_amount_est * _rebuy_ratio, 2)
                    _rebuy_cap = pos["max_position"] - pos["total_cost"] + _sell_amount_est  # 卖出后释放的空间
                    _rebuy_amount = round(min(_rebuy_amount, _rebuy_cap * 0.8), 2)
                    if _rebuy_amount >= 100:
                        _trigger_nav = round(current_nav * (1 - _rebuy_discount), 4)
                        best_signal["rebuy_recommendation"] = {
                            "action": "buy",
                            "amount": _rebuy_amount,
                            "ratio": _rebuy_ratio,
                            "trend": _rebuy_trend,
                            "trigger_nav": _trigger_nav,
                            "discount": _rebuy_discount,
                            "window_days": 20,  # 20天内有效
                            "reason": (f"v5.11延迟回补: 趋势{_rebuy_trend}, "
                                      f"净值回落{_rebuy_discount:.0%}至{_trigger_nav:.4f}时"
                                      f"回补{_rebuy_ratio:.0%}={_rebuy_amount:.0f}元(20天内有效)"),
                        }
                        best_signal["reason"] += (f" → 延迟回补{_rebuy_amount:.0f}元"
                                                  f"(净值≤{_trigger_nav:.4f}时触发)")
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
        dynamic_thresholds=dyn, regime=regime, regime_source=_regime_source
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
        buy_amount = round(max_pos * 0.80 * size_mul, 2)  # v5.13: 大跌抄底从70%→80%，加大抄底力度

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
            buy_amount = round(max_pos * 0.55 * size_mul, 2)  # v5.5 优化: 低位建仓从40%→55%
            build_signal = _make_signal(
                fund_code, signal_name="低位建仓", action="buy", priority=6,
                amount=buy_amount,
                reason=f"10日累跌{mid_10d}%, 今日企稳, 中期低位建仓{buy_amount}元",
            )
        elif (short_5d is not None and short_5d <= TREND_BUILD_TRIGGER_5D
                and today_change > 0):
            buy_amount = round(max_pos * 0.45 * size_mul, 2)  # v5.5 优化: 反弹建仓从30%→45%
            build_signal = _make_signal(
                fund_code, signal_name="反弹建仓", action="buy", priority=6,
                amount=buy_amount,
                reason=f"5日累跌{short_5d}%, 今日反弹, 逢低建仓{buy_amount}元",
            )
        elif (consecutive_down >= 3 and today_change < 0
                and len(hist_changes) >= 1
                and abs(today_change) < abs(hist_changes[0]) * 0.6):
            buy_amount = round(max_pos * 0.35 * size_mul, 2)  # v5.5 优化: 跌势放缓建仓从25%→35%
            build_signal = _make_signal(
                fund_code, signal_name="跌势放缓建仓", action="buy", priority=7,
                amount=buy_amount,
                reason=f"连跌{consecutive_down}天, 跌幅收窄, 试探建仓{buy_amount}元",
            )

        if build_signal:
            build_signal["market_analysis"] = market_analysis
            _append_signal_history(fund_code, build_signal, market_ctx)
            return build_signal

    # --- 温和回调建仓 (v5.13新增) ---
    # 空仓时3日累跌 > 1倍波动率 + 非极端环境 → 45%建仓
    if not in_cooldown and can_buy_empty:
        vol_for_mild = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
        short_3d = trend_ctx.get("short_3d")
        if (short_3d is not None
                and short_3d < 0
                and abs(short_3d) > vol_for_mild
                and vol_state != "extreme_vol"):
            max_pos = pos["max_position"]
            buy_amount = round(max_pos * 0.45 * size_mul, 2)
            sig = _make_signal(
                fund_code, signal_name="温和回调建仓", action="buy", priority=7,
                amount=buy_amount,
                reason=f"3日累跌{short_3d}%>波动率{vol_for_mild:.1f}%, 温和回调建仓{buy_amount}元",
            )
            sig["market_analysis"] = market_analysis
            _append_signal_history(fund_code, sig, market_ctx)
            return _stamp(sig, confidence, source)

    # --- 连跌低吸 ---
    consec_dip_thresh = dyn.get("consecutive_dip_trigger", DEFAULT_CONSECUTIVE_DIP_TRIGGER)
    if (today_change <= consec_dip_thresh
            and len(recent) >= 1
            and recent[0].get("change") is not None
            and recent[0]["change"] < 0
            and not in_cooldown and can_buy_empty):
        max_pos = pos["max_position"]
        buy_amount = round(max_pos * 0.45 * size_mul, 2)  # v5.5 优化: 连跌低吸从30%→45%
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
            and today_change <= 0.3  # v5.13: 从<=0放宽到<=0.3
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
            buy_amount = round(max_pos * 0.50 * size_mul, 2)  # v5.5 优化: 冷却期后建仓从30%→50%
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

    # v5.13: 注入适配评分缓存（由回测生成 data/fitness_cache.json）
    fitness_cache = get_fitness_scores()
    if fitness_cache:
        for sig in signals:
            code = sig.get("fund_code", "")
            real_code, _ = parse_fund_key(code)
            fi = fitness_cache.get(code) or fitness_cache.get(real_code)
            if fi and sig.get("market_analysis"):
                sig["market_analysis"]["fitness_score"] = fi.get("score")
                sig["market_analysis"]["fitness_grade"] = fi.get("grade")

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
            discount = max(0.85, 1.0 - (total_buy_count - 1) * 0.05)  # v5.5 优化: 折扣更温和（原0.75/0.10→0.85/0.05），允许更多资金入场
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