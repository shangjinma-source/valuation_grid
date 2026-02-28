"""
grid/helpers.py - 低频网格策略：计算辅助函数

包含: 波动率自适应 | 卖出评分 | 止损评估(L1/L2/L3) | 补仓控制 | NAV/利润计算 | FIFO卖出计划 | 信号构建
"""

import math
from datetime import datetime
from typing import Optional, List, Tuple

from positions import get_sell_fee_rate, load_positions, save_positions, parse_fund_key
from valuation.core import _is_market_closed

from .config import (
    _get_vol_sensitivity, DEFAULT_VOL_SENSITIVITY,
    # 波动率倍数
    DIP_BUY_VOL_MULTIPLE, SUPPLEMENT_TRIGGER_VOL_MULTIPLE,
    SUPPLEMENT_LOSS_VOL_MULTIPLE, CONSECUTIVE_DIP_VOL_MULTIPLE,
    STOP_LOSS_VOL_MULTIPLE, TAKE_PROFIT_VOL_MULTIPLE,
    TREND_WEAK_VOL_MULTIPLE, DISASTER_LOSS_VOL_MULTIPLE,
    DISASTER_DAILY_VOL_MULTIPLE,
    # 固定默认值
    DEFAULT_DIP_THRESHOLD, DEFAULT_TAKE_PROFIT_TRIGGER,
    DEFAULT_STOP_LOSS_BASE, DEFAULT_SUPPLEMENT_TRIGGER,
    DEFAULT_SUPPLEMENT_LOSS_MIN, DEFAULT_CONSECUTIVE_DIP_TRIGGER,
    DEFAULT_TREND_WEAK_CUMULATIVE, DEFAULT_DISASTER_LOSS,
    DEFAULT_DISASTER_DAILY_DROP,
    # 档位/常量
    COOLDOWN_DAYS, SUPPLEMENT_MAX_COUNT_DEFAULT, SUPPLEMENT_MAX_COUNT_HARD_CAP,
    SUPPLEMENT_TIERS_VOL, SUPPLEMENT_TIERS, SUPPLEMENT_MIN_GAP_TRADE_DAYS,
    SUPPLEMENT_REBUY_STEP_PCT,
    TOTAL_PROFIT_SELL_TIERS_VOL, TOTAL_PROFIT_SELL_TIERS,
    TAKE_PROFIT_TIERS, SLOW_PROFIT_TIERS,
    TRAIL_DD_BASE, TRAIL_DD_MIN, TRAIL_DD_MAX, TRAIL_PROFIT_SELL_TIERS,
    VOL_LOW, VOL_NORMAL_HIGH, VOL_EXTREME,
    STOP_LOSS_L1_FACTOR, STOP_LOSS_L2_SELL_PCT_BASE,
    STOP_LOSS_L3_FACTOR, STOP_LOSS_L3_CONSEC_DOWN,
    WIN_RATE_TIGHTEN_THRESHOLD, WIN_RATE_TIGHTEN_FACTOR,
    LIQUIDITY_PREMIUM_EXTRA_PCT,
    PASSTHROUGH_LOSS_DOWNGRADE, PASSTHROUGH_MIN_NET_PROFIT_RATIO,
    PASSTHROUGH_MIN_NET_PROFIT_ABS, PASSTHROUGH_LOSS_RATIO_THRESHOLD,
)

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
                     hold_days: int, peak_profit: float,
                     nz_bonus: float = 0.0,
                     tp_suppress_threshold: int = 65) -> dict:
    """
    v5.10 改进：
    1. 撤销v5.9的hold_time_bonus（灾难性副作用：止盈卖出-67%，强势止盈-100%）
    2. 新增nz_bonus参数：扭亏奖励分（P0将扭亏止盈纳入评分系统）
    """
    profit_pct = round((current_nav / batch["nav"] - 1) * 100, 2) if batch["nav"] > 0 else 0.0

    # v5.5 优化: 持仓不足15个交易日不触发止盈（除非盈利>10%），给趋势更多发展空间
    if hold_days < 15 and profit_pct < 10.0:
        return {"score": 0, "sell_pct": 0, "signal_name": None, "reason": "持仓不足15日且盈利<10%，暂不止盈"}

    if profit_pct <= fee_rate * 2.0:
        return {"score": 0, "sell_pct": 0, "signal_name": None, "reason": "盈利不足覆盖费率"}

    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.2

    # v5.5 优化: profit_norm大幅提高（原max(5.0, vol*6.0)→max(8.0, vol*8.0)），需要更高盈利才产生有效评分
    # 回测显示分批止盈平均只赚10元，慢涨止盈只赚9元，提高门槛让利润奔跑
    profit_norm = max(8.0, vol * 8.0)
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

    # v5.10 重构P1: 撤销v5.9的hold_time_bonus——这是v5.9全部恶化的罪魁祸首
    # v5.9 hold_time_bonus导致: 止盈卖出-67%(15566→5195), 强势止盈-100%(774→0)
    # 止盈利润从887k暴跌到715k(-172k), 同时扭亏止盈膨胀+41%至12277次

    # v5.10 重构P0: 扭亏奖励分（将扭亏止盈纳入评分系统）
    # nz_bonus > 0 表示当前处于扭亏状态（total_profit>0且≥2批），为评分加分
    # 这样扭亏止盈必须通过评分门槛才能触发，不再是独立的逃逸通道

    total_score = profit_score + trail_score + momentum_score + liquidity_score + fee_drag + nz_bonus

    # v5.5 优化: 大幅提高评分阈值（原65/50/40/35→75/60/50/45），减少低质量止盈
    # 回测显示强势止盈(79元/笔)才是正确卖法，应减少分批止盈(10元)和慢涨止盈(9元)
    if total_score >= 75:
        sell_pct = 100
        signal_name = "强势止盈"
    elif total_score >= 60:
        sell_pct = 70
        signal_name = "止盈卖出"
    elif total_score >= 55:  # v5.8 重构P2: 分批止盈门槛从50→55，让更多交易发展到更高级别
        sell_pct = 50
        signal_name = "分批止盈"
    elif total_score >= 53:  # v5.13: 慢涨止盈门槛从52→53
        sell_pct = 30
        signal_name = "慢涨止盈"
    else:
        sell_pct = 0
        signal_name = None

    if today_change >= liquidity_trigger and sell_pct > 0 and sell_pct < 100:
        sell_pct = min(100, sell_pct + LIQUIDITY_PREMIUM_EXTRA_PCT)

    # v5.13: 止盈抑制门槛由行情模式控制（默认65=v5.13基线）
    trend_label = trend_ctx.get("trend_label", "震荡")
    if trend_label in ("连涨", "偏强", "中期走强") and total_score < tp_suppress_threshold:
        sell_pct = 0
        signal_name = None
        reason = f"综合评分{total_score:.0f}但趋势为{trend_label}，强力抑制止盈让利润奔跑"

    reason = (f"综合评分{total_score:.0f}(盈利{profit_score:.0f}+回撤{trail_score:.0f}"
              f"+动量{momentum_score:.0f}+流动性{liquidity_score:.0f}+费率{fee_drag:.0f}"
              f"+扭亏{nz_bonus:.0f})")

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
                        source: str, supplement_count: int = 0,
                        today_change: float = 0.0,
                        l2_sell_pct_base: int = None) -> dict:  # v5.13: 行情模式覆盖L2基准
    """
    v5.8 重构P0: L3分级减仓 + L2时间梯度微调

    v5.7诊断: L3极端止损膨胀（第二次水床效应），2249次亏95元/次=214k
    v5.8修复:
      - L3分级: 持仓<20天+当日跌幅>-5%（非极端暴跌）→减仓70%而非清仓
      - L2时间梯度: <15天豁免→<10天豁免，回收5天的L2保护窗口
    """
    if l2_sell_pct_base is None:
        l2_sell_pct_base = STOP_LOSS_L2_SELL_PCT_BASE
    if hold_days < 7:
        return {"level": None, "sell_pct": 0, "reason": "未满7天，走灾难保护通道"}

    confidence_adj = 0.0
    if source == "estimation" and confidence < 0.6:
        confidence_adj = -1.0

    effective_stop = stop_loss_adj - fee_rate + confidence_adj
    consec_down = trend_ctx.get("consecutive_down", 0)

    # L3: 极端止损（v5.8 重构P0: 分级减仓，不再一律清仓100%）
    extreme_threshold = effective_stop * STOP_LOSS_L3_FACTOR
    if profit_pct <= extreme_threshold or (profit_pct <= effective_stop and consec_down >= STOP_LOSS_L3_CONSEC_DOWN):
        reason = f"极端止损: 浮亏{profit_pct}%"
        if consec_down >= STOP_LOSS_L3_CONSEC_DOWN:
            reason += f", 连跌{consec_down}天"

        # v5.9 重构P3: L3分级减仓窗口扩大——新仓定义从<20天→<25天，非暴跌从-5%→-4%
        # 原因: v5.8 L3单笔-81元仍超目标-70元，扩大分级窗口让更多L3走70%路径
        l3_sell_pct = 100  # 默认清仓
        if hold_days < 25 and today_change > -4.0:
            # v5.9 重构P3: 持仓<25天+当日跌幅未达-4%（非极端暴跌）→减仓70%
            l3_sell_pct = 70
            reason += f", 持仓{hold_days}天(<25天)+今日跌{today_change}%(>-4%), 减仓70%保留观察仓"
        else:
            reason += f", 持仓{hold_days}天, 清仓100%"

        return {"level": "L3", "sell_pct": l3_sell_pct, "reason": reason}

    # v5.8 重构P0: L2止损线时间梯度调整（<15天豁免→<10天豁免）
    l2_stop = effective_stop  # 基础止损线

    # v5.7 重构: 震荡市放宽（无方向波动不应触发方向性止损）
    trend_label = trend_ctx.get("trend_label", "震荡")
    if trend_label == "震荡":
        l2_stop = round(l2_stop * 1.25, 2)  # v5.7: 震荡市止损线放宽25%

    if hold_days < 10:  # v5.8 重构P0: 从<15天→<10天，回收5天的L2保护窗口
        # v5.8 重构P0: 持仓<10天，L2需趋势确认才触发
        # 原因: <15天太激进，把太多本应L2止损的交易推到了L3
        if profit_pct <= l2_stop:
            # 需要趋势确认: 连跌≥3天
            if consec_down >= 3:
                l2_sell_pct = max(30, l2_sell_pct_base - supplement_count * 10)
                return {
                    "level": "L2",
                    "sell_pct": l2_sell_pct,
                    "reason": f"常规止损: 浮亏{profit_pct}% ≤ 止损线{l2_stop:.1f}%, 持仓{hold_days}天, "
                              f"连跌{consec_down}天确认趋势恶化, 减仓{l2_sell_pct}%"
                              f"(已补仓{supplement_count}次)"
                }
            else:
                # 触及止损线但趋势未确认，仅预警
                return {
                    "level": "L1",
                    "sell_pct": 0,
                    "reason": f"止损预警: 浮亏{profit_pct}%触及止损线{l2_stop:.1f}%, 但持仓仅{hold_days}天"
                              f"且连跌{consec_down}天(<3天), 等待趋势确认"
                }
    elif hold_days <= 30:
        # v5.7 重构: 持仓10-30天，止损线放宽×1.3
        l2_stop = round(l2_stop * 1.3, 2)

    # L2: 常规止损（v5.2: 补仓次数越多，保留越多仓位）
    if profit_pct <= l2_stop:
        l2_sell_pct = max(30, l2_sell_pct_base - supplement_count * 10)
        return {
            "level": "L2",
            "sell_pct": l2_sell_pct,
            "reason": f"常规止损: 浮亏{profit_pct}% ≤ 止损线{l2_stop:.1f}%, 减仓{l2_sell_pct}%"
                      f"(已补仓{supplement_count}次, 保留反弹仓位)"
        }

    # L1: 预警
    warning_threshold = l2_stop * STOP_LOSS_L1_FACTOR
    if profit_pct <= warning_threshold:
        return {
            "level": "L1",
            "sell_pct": 0,
            "reason": f"止损预警: 浮亏{profit_pct}%接近止损线{l2_stop:.1f}%(预警线{warning_threshold:.1f}%)"
        }

    return {"level": None, "sell_pct": 0, "reason": ""}


# ============================================================
# 补仓禁入判断
# ============================================================

def _is_supplement_forbidden(trend_ctx: dict, confidence: float,
                             source: str, vol_state: str,
                             batches_sorted: list = None) -> tuple:
    if vol_state == "extreme_vol":
        vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 0
        return True, f"波动率{vol}%处于极端水平，暂停补仓"

    mid_10d = trend_ctx.get("mid_10d")
    consecutive_down = trend_ctx.get("consecutive_down", 0)
    max_drawdown = trend_ctx.get("max_drawdown", 0)
    vol = trend_ctx.get("volatility") or 0

    if mid_10d is not None and mid_10d <= -10 and consecutive_down >= 5:  # v5.5 优化: 放宽禁入条件（原-7/3→-10/5），允许更多补仓机会
        return True, f"10日累跌{mid_10d}%且连跌{consecutive_down}天，趋势禁入"

    if max_drawdown >= 15 and vol >= 2.5:  # v5.5 优化: 放宽（原10/2.2→15/2.5）
        return True, f"回撤{max_drawdown}%+波动率{vol}%，高风险禁入"

    if source == "estimation" and confidence < 0.6:
        return True, f"置信度{confidence:.0%}偏低，盘中补仓禁入"

    # v5.7 重构 P3: 豁免期内趋势恶化冻结补仓
    # v5.8 重构P3: 放宽冻结条件（连跌≥3天+5日累跌<-5% → 连跌≥4天+5日累跌<-7%）
    # 原因: 原条件太严格导致补仓执行率过低（第2次补仓仅49%）
    if batches_sorted:
        oldest = batches_sorted[0]
        oldest_buy_date = datetime.strptime(oldest["buy_date"], "%Y-%m-%d").date()
        oldest_hold_days = (datetime.now().date() - oldest_buy_date).days
        if oldest_hold_days < 20:  # 在20天豁免期内
            short_5d = trend_ctx.get("short_5d")
            if consecutive_down >= 4 and short_5d is not None and short_5d <= -7:  # v5.8 重构P3: 3/5→4/7
                return True, f"豁免期内({oldest_hold_days}天)趋势恶化(连跌{consecutive_down}天+5日累跌{short_5d}%), 冻结补仓"

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
    dynamic_gap = SUPPLEMENT_MIN_GAP_TRADE_DAYS  # v5.8: 默认2天
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
        size_mul_trend = 1.30  # v5.13: 上升趋势加码从1.20→1.30
    else:
        size_mul_trend = 1.0

    momentum_adj = 1.0
    if momentum_score < -0.5:
        momentum_adj = 0.75
    elif momentum_score < -0.3:
        momentum_adj = 0.85

    raw = size_mul_risk * size_mul_conf * size_mul_trend * momentum_adj
    return round(max(0.40, min(1.40, raw)), 2)  # v5.13: 上限从1.30→1.40


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