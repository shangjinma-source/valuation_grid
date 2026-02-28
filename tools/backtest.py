"""
backtest.py - 低频网格策略历史回测引擎 v1.0 (同步 strategy v5.13)

功能：
  1. 从天天基金拉取指定基金的完整历史净值数据（支持自定义时间范围）
  2. 用一万元初始本金，逐日模拟 strategy.py 中的低频网格策略
  3. 模拟买入、卖出、补仓、止损、止盈等全部信号逻辑
  4. 输出详细的交易记录、资金曲线、收益统计和策略评估指标
  5. 支持多基金批量回测 + 横向对比
  6. 支持行情模式参数（--regime bull/neutral/bear）

用法：
  python backtest.py                         # 使用默认基金列表回测
  python backtest.py --funds 017193 015968   # 指定基金
  python backtest.py --days 365              # 回测最近365天
  python backtest.py --capital 10000         # 初始资金1万
  python backtest.py --detail                # 打印每笔交易详情
  python backtest.py --regime bull           # 牛市模式回测

注意：
  - 本脚本完全独立，不修改任何现有文件
  - 不写入 data/ 目录，不影响你的真实持仓数据
  - 策略参数直接从 strategy.py 导入，确保和线上一致

依赖：
  - 与项目在同一目录下运行（需要 import strategy 中的常量和函数）
  - 网络连接（拉取天天基金历史净值API）
"""

import argparse
import json
import math
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.request import urlopen, Request

# ============================================================
# 从 strategy.py 导入核心常量和阈值函数（保持一致性）
# ============================================================
try:
    from grid.config import (
        # 核心阈值常量
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
        # 补仓档位
        SUPPLEMENT_TIERS_VOL, SUPPLEMENT_TIERS,
        SUPPLEMENT_CAP_RATIO, SUPPLEMENT_MAX_COUNT_DEFAULT,
        SUPPLEMENT_MAX_COUNT_HARD_CAP,
        # 止盈档位
        TAKE_PROFIT_TIERS, SLOW_PROFIT_TIERS,
        TOTAL_PROFIT_SELL_TIERS_VOL, TOTAL_PROFIT_SELL_TIERS,
        # 趋势建仓
        TREND_BUILD_TRIGGER_5D, TREND_BUILD_TRIGGER_10D,
        # 回撤止盈
        TRAIL_PROFIT_ACTIVATE, TRAIL_DD_BASE, TRAIL_DD_MIN, TRAIL_DD_MAX,
        TRAIL_PROFIT_SELL_TIERS,
        # 波动率状态
        VOL_LOW, VOL_NORMAL_HIGH, VOL_EXTREME,
        # 止损
        STOP_LOSS_L1_FACTOR, STOP_LOSS_L2_SELL_PCT_BASE,
        STOP_LOSS_L3_FACTOR, STOP_LOSS_L3_CONSEC_DOWN,
        # 灾难保护
        DISASTER_CONSECUTIVE_DOWN, DISASTER_SELL_PCT_EXTREME, DISASTER_SELL_PCT_DAILY,
        # 补仓间隔
        SUPPLEMENT_MIN_GAP_TRADE_DAYS, SUPPLEMENT_REBUY_STEP_PCT,
        # 冷却
        COOLDOWN_DAYS,
    )
    from grid.engine import (
        _analyze_trend, _calc_momentum_score, _calc_risk_multiplier,
        _classify_volatility,
    )
    STRATEGY_IMPORTED = True
    print("[Backtest] ✓ 成功导入 grid 模块参数")
except ImportError as e:
    print(f"[Backtest] ⚠ 无法导入 strategy.py ({e})，使用内置默认参数")
    STRATEGY_IMPORTED = False
    # 内置兜底默认值（与 strategy.py v5.5 一致）
    DIP_BUY_VOL_MULTIPLE = 1.8
    SUPPLEMENT_TRIGGER_VOL_MULTIPLE = 1.2
    SUPPLEMENT_LOSS_VOL_MULTIPLE = 2.2
    CONSECUTIVE_DIP_VOL_MULTIPLE = 0.7
    STOP_LOSS_VOL_MULTIPLE = 6.0          # v5.5
    TAKE_PROFIT_VOL_MULTIPLE = 2.5        # v5.5
    TREND_WEAK_VOL_MULTIPLE = 1.5
    DISASTER_LOSS_VOL_MULTIPLE = 6.5      # v5.5
    DISASTER_DAILY_VOL_MULTIPLE = 3.0
    DEFAULT_DIP_THRESHOLD = -2.5
    DEFAULT_TAKE_PROFIT_TRIGGER = 2.0
    DEFAULT_STOP_LOSS_BASE = -7.0         # v5.5
    DEFAULT_SUPPLEMENT_TRIGGER = -1.5
    DEFAULT_SUPPLEMENT_LOSS_MIN = -3.0
    DEFAULT_CONSECUTIVE_DIP_TRIGGER = -1.0
    DEFAULT_TREND_WEAK_CUMULATIVE = -2.0
    DEFAULT_DISASTER_LOSS = -12.0         # v5.5
    DEFAULT_DISASTER_DAILY_DROP = -6.0    # v5.5
    SUPPLEMENT_TIERS_VOL = [
        (0, 0.70, 1.0, 1.8), (1, 0.35, 1.4, 3.0),   # v5.13: 首次0.60→0.70
        (2, 0.25, 1.8, 4.5), (3, 0.15, 2.2, 6.0), (4, 0.10, 2.6, 7.5),
    ]
    SUPPLEMENT_TIERS = [
        (0, 0.70, -1.2, -2.5), (1, 0.35, -1.8, -4.0),  # v5.13: 首次0.60→0.70
        (2, 0.25, -2.2, -6.5), (3, 0.15, -2.8, -9.0), (4, 0.10, -3.2, -11.0),
    ]
    SUPPLEMENT_CAP_RATIO = 0.35           # v5.5
    SUPPLEMENT_MAX_COUNT_DEFAULT = 3
    SUPPLEMENT_MAX_COUNT_HARD_CAP = 5
    TAKE_PROFIT_TIERS = [(12.0, 100), (8.0, 70), (5.0, 50)]  # v5.5
    SLOW_PROFIT_TIERS = [(12.0, 70), (8.0, 50), (6.0, 30)]   # v5.5
    TOTAL_PROFIT_SELL_TIERS_VOL = [(7.0, 50), (5.0, 30)]      # v5.9: 提高扭亏门槛(5.0/3.5→7.0/5.0)
    TOTAL_PROFIT_SELL_TIERS = [(10.0, 50), (7.0, 30)]          # v5.9: 提高扭亏门槛(7.0/5.0→10.0/7.0)
    TREND_BUILD_TRIGGER_5D = -3.0
    TREND_BUILD_TRIGGER_10D = -5.0
    TRAIL_PROFIT_ACTIVATE = 5.0           # v5.5
    TRAIL_DD_BASE = 2.5                   # v5.5
    TRAIL_DD_MIN = 1.8                    # v5.5
    TRAIL_DD_MAX = 5.0                    # v5.5
    TRAIL_PROFIT_SELL_TIERS = [(12.0, 70), (8.0, 50), (5.0, 30)]  # v5.5
    VOL_LOW = 0.8
    VOL_NORMAL_HIGH = 1.8
    VOL_EXTREME = 3.0
    STOP_LOSS_L1_FACTOR = 0.7
    STOP_LOSS_L2_SELL_PCT_BASE = 35       # v5.5
    STOP_LOSS_L3_FACTOR = 1.5
    STOP_LOSS_L3_CONSEC_DOWN = 7          # v5.5
    DISASTER_CONSECUTIVE_DOWN = 3
    DISASTER_SELL_PCT_EXTREME = 50
    DISASTER_SELL_PCT_DAILY = 30
    SUPPLEMENT_MIN_GAP_TRADE_DAYS = 2  # v5.8 重构P3: 3→2
    SUPPLEMENT_REBUY_STEP_PCT = 1.0
    COOLDOWN_DAYS = 0                     # v5.5

# ============================================================
# v5.13: 行情模式参数（与 strategy.py 保持一致）
# ============================================================
REGIME_PARAMS = {
    # v5.16: 铁律驱动 — 只调买入端，锁定止盈/止损/仓位=基线
    "bull": {  # v5.17: =基线
        "first_build_ratio": 0.70,       # =基线
        "size_mul_cap": 1.40,            # 锁定基线
        "tp_suppress_threshold": 65,     # 锁定基线
        "supplement_max_count": 3,       # 锁定基线
        "l2_stop_loss_base": 35,         # 锁定基线
        "rebuy_discount": 0.015,         # =基线
    },
    "neutral": {
        "first_build_ratio": 0.70,
        "size_mul_cap": 1.40,
        "tp_suppress_threshold": 65,
        "supplement_max_count": 3,
        "l2_stop_loss_base": 35,
        "rebuy_discount": 0.015,
    },
    "bear": {
        "first_build_ratio": 0.55,       # 买入端收紧
        "size_mul_cap": 1.40,            # 锁定基线
        "tp_suppress_threshold": 65,     # 锁定基线
        "supplement_max_count": 3,       # 锁定基线
        "l2_stop_loss_base": 35,         # 锁定基线
        "rebuy_discount": 0.030,         # 买入端收紧
    },
}

# ============================================================
# 费率表（与 positions.py 一致）
# ============================================================
DEFAULT_FEE_SCHEDULE = [(7, 1.50), (30, 0.50), (None, 0.00)]


def get_sell_fee_rate(hold_days: int, fee_schedule=None) -> float:
    schedule = fee_schedule or DEFAULT_FEE_SCHEDULE
    for threshold, rate in schedule:
        if threshold is None or hold_days < threshold:
            return rate
    return 0.0


# ============================================================
# 数据获取：从天天基金拉取历史净值
# ============================================================

REQUEST_TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://fundf10.eastmoney.com/",
}


def fetch_fund_name(fund_code: str) -> str:
    """获取基金名称"""
    try:
        url = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
        req = Request(url, headers=HEADERS)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")
        import re
        match = re.search(r'var\s+fS_name\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return fund_code


def fetch_nav_history(fund_code: str, page_size: int = 49, max_records: int = 0) -> List[dict]:
    """
    从天天基金 F10DataApi 拉取历史净值（HTML接口，稳定可靠）。
    max_records: 最大记录数，0=不限（拉取全部历史）
    返回格式（按日期升序，最早在前）：
    [{"date": "2024-01-02", "nav": 1.2345, "change": -1.23}, ...]
    """
    import re as _re
    all_items = []
    page = 1

    while True:
        try:
            url = (
                f"http://fund.eastmoney.com/f10/F10DataApi.aspx?"
                f"type=lsjz&code={fund_code}&page={page}&per={page_size}"
            )
            req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                content = resp.read().decode("utf-8")

            # 解析总记录数
            records_match = _re.search(r'records:(\d+)', content)
            total_count = int(records_match.group(1)) if records_match else 0

            # 解析HTML表格行（td可能带class属性）
            items = []
            rows = _re.findall(r'<tr>(.*?)</tr>', content, _re.DOTALL)
            for row in rows:
                cells = _re.findall(r'<td[^>]*>(.*?)</td>', row, _re.DOTALL)
                if len(cells) >= 4:
                    date_str = cells[0].strip()
                    nav_str = cells[1].strip()
                    change_raw = _re.sub(r'<[^>]+>', '', cells[3]).strip()
                    change_str = change_raw.replace('%', '').replace('％', '')
                    if _re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        items.append({"FSRQ": date_str, "DWJZ": nav_str, "JZZZL": change_str})

            if not items:
                break
            all_items.extend(items)

            total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1
            if page >= total_pages or len(all_items) >= total_count:
                break
            page += 1
            time.sleep(0.3)
            # v5.4 优化: 解除2000条硬限制，支持全量历史拉取
            if max_records > 0 and len(all_items) >= max_records:
                break
        except Exception as e:
            if page == 1:
                print(f"  ⚠ 拉取失败: {e}")
            break

    # 转换为标准格式（升序）
    result = []
    for item in reversed(all_items):
        date_str = item.get("FSRQ", "")
        try:
            nav = float(item.get("DWJZ", ""))
        except (ValueError, TypeError):
            continue
        try:
            change_s = item.get("JZZZL", "")
            change = float(change_s) if change_s and change_s.strip() else None
        except (ValueError, TypeError):
            change = None
        result.append({"date": date_str, "nav": nav, "change": change})
    return result


def get_fund_max_history_range(nav_data: List[dict]) -> dict:
    """获取基金历史净值范围和极值统计"""
    if not nav_data:
        return {}
    navs = [d["nav"] for d in nav_data]
    changes = [d["change"] for d in nav_data if d.get("change") is not None]
    return {
        "start_date": nav_data[0]["date"],
        "end_date": nav_data[-1]["date"],
        "total_days": len(nav_data),
        "nav_min": min(navs),
        "nav_max": max(navs),
        "nav_start": navs[0],
        "nav_end": navs[-1],
        "buy_hold_return": round((navs[-1] / navs[0] - 1) * 100, 2),
        "max_daily_gain": round(max(changes), 2) if changes else None,
        "max_daily_loss": round(min(changes), 2) if changes else None,
        "avg_daily_change": round(sum(abs(c) for c in changes) / len(changes), 4) if changes else None,
    }


# ============================================================
# 回测引擎核心：趋势分析（复用 strategy.py 逻辑）
# ============================================================

def bt_analyze_trend(today_change: float, hist_changes: List[float],
                     nav_list: List[dict] = None) -> dict:
    """与 strategy._analyze_trend 等价的纯净值版本"""
    if STRATEGY_IMPORTED:
        # 直接用 strategy.py 的实现
        return _analyze_trend(today_change, hist_changes,
                              nav_history=nav_list,
                              nav_history_60=nav_list)
    # 内置简化版（结构与 strategy.py 一致）
    all_changes = [today_change] + hist_changes

    def _compound(changes):
        p = 1.0
        for c in changes:
            p *= (1 + c / 100)
        return round((p - 1) * 100, 2)

    short_3d = _compound(all_changes[:3]) if len(all_changes) >= 3 else _compound(all_changes)
    short_5d = _compound(all_changes[:5]) if len(all_changes) >= 5 else None
    mid_10d = _compound(all_changes[:10]) if len(all_changes) >= 10 else None
    long_20d = _compound(all_changes[:20]) if len(all_changes) >= 20 else None

    vol_data = all_changes[:20]
    volatility = volatility_robust = None
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

    volume_proxy = None
    if len(all_changes) >= 5 and volatility_robust and volatility_robust > 0:
        recent_abs = [abs(c) for c in all_changes[:5]]
        volume_proxy = round(sum(recent_abs) / len(recent_abs) / volatility_robust, 2)

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
    if nav_list and len(nav_list) >= 5:
        navs = [h["nav"] for h in nav_list if h.get("nav") is not None]
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
        "short_3d": short_3d,
        "short_5d": short_5d,
        "mid_10d": mid_10d,
        "long_20d": long_20d,
        "volatility": volatility,
        "volatility_robust": volatility_robust,
        "volume_proxy": volume_proxy,
        "consecutive_down": consecutive_down,
        "consecutive_up": consecutive_up,
        "max_drawdown": max_drawdown,
        "max_drawdown_60": max_drawdown,
        "trend_label": trend_label,
        "data_days": len(all_changes),
    }


# ============================================================
# 回测引擎核心：波动率自适应阈值
# ============================================================

def bt_vol_adaptive_thresholds(vol: float, sensitivity: float = 1.0) -> dict:
    """与 strategy._vol_adaptive_thresholds 等价"""
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
        "dip_threshold": dip, "tp_trigger": tp, "stop_loss": sl,
        "supplement_trigger": supp_trig, "supplement_loss_min": supp_loss,
        "consecutive_dip": consec_dip, "trend_weak": tw,
        "disaster_loss": dis_loss, "disaster_daily": dis_daily,
        "supplement_tiers": supp_tiers, "total_profit_tiers": tp_tiers,
        "_vol_based": True,
    }


def bt_calc_dynamic_thresholds(trend_ctx: dict) -> dict:
    """简化版动态阈值（回测不需要信号胜率自适应）"""
    mdd_20 = trend_ctx.get("max_drawdown") or 0.0
    mdd = mdd_20
    if mdd <= 5:
        mdd_term = 0.0
    elif mdd <= 10:
        mdd_term = (mdd - 5) * 0.06
    else:
        mdd_term = 0.30 + (mdd - 10) * 0.03
    risk_mul = max(0.85, min(1.5, 1.0 + mdd_term))

    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
    vol_state = "low_vol" if vol < VOL_LOW else "normal_vol" if vol < VOL_NORMAL_HIGH else "high_vol" if vol < VOL_EXTREME else "extreme_vol"

    va = bt_vol_adaptive_thresholds(vol)

    dip_threshold = round(va["dip_threshold"] * risk_mul, 2)
    tp_trigger = round(va["tp_trigger"], 2)
    stop_loss_adj = round(va["stop_loss"] * risk_mul, 2)

    supplement_tiers_adj = [(c, r, round(t * risk_mul, 2), round(l * risk_mul, 2))
                            for c, r, t, l in va["supplement_tiers"]]

    trail_dd = max(TRAIL_DD_MIN, min(TRAIL_DD_MAX, TRAIL_DD_BASE * risk_mul))

    if vol_state == "low_vol":
        dip_threshold = round(dip_threshold * 0.85, 2)
        tp_trigger = round(tp_trigger * 0.85, 2)

    dip_threshold = max(-6.0, dip_threshold)
    stop_loss_adj = max(-12.0, stop_loss_adj)
    rebuy_step = max(0.8, vol * 0.8) if vol else SUPPLEMENT_REBUY_STEP_PCT

    momentum = 0.0
    s5 = trend_ctx.get("short_5d")
    m10 = trend_ctx.get("mid_10d")
    l20 = trend_ctx.get("long_20d")
    if s5 is not None:
        momentum = round(max(-1.0, min(1.0, 0.5 * math.tanh(s5 / 4.0) +
                                        0.3 * math.tanh((m10 or 0) / 6.0) +
                                        0.2 * math.tanh((l20 or 0) / 10.0))), 3)

    return {
        "risk_multiplier": round(risk_mul, 2),
        "dip_threshold": dip_threshold,
        "tp_trigger": tp_trigger,
        "stop_loss_adj": stop_loss_adj,
        "supplement_tiers": supplement_tiers_adj,
        "trail_dd": round(trail_dd, 2),
        "vol_state": vol_state,
        "momentum_score": momentum,
        "rebuy_step": round(rebuy_step, 2),
        "_va": va,
        "consecutive_dip_trigger": round(va["consecutive_dip"], 2),
        "supplement_trigger": round(va["supplement_trigger"], 2),
        "supplement_loss_min": round(va["supplement_loss_min"], 2),
        "trend_weak_cumulative": round(va["trend_weak"], 2),
        "disaster_loss_threshold": round(va["disaster_loss"], 2),
        "disaster_daily_drop": round(va["disaster_daily"], 2),
        "total_profit_sell_tiers": va.get("total_profit_tiers", TOTAL_PROFIT_SELL_TIERS),
    }


# ============================================================
# 回测模拟器
# ============================================================

class BacktestSimulator:
    """低频网格策略回测模拟器"""

    def __init__(self, fund_code: str, nav_data: List[dict],
                 initial_capital: float = 10000.0,
                 max_position: float = 10000.0,
                 fee_schedule=None,
                 regime: str = "auto"):
        self.fund_code = fund_code
        self.nav_data = nav_data  # 升序
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.fee_schedule = fee_schedule or DEFAULT_FEE_SCHEDULE
        # v5.13: 行情模式（"auto"为逐日动态切换）
        self.regime_mode = regime  # "auto" / "bull" / "neutral" / "bear"
        self.regime = regime if regime != "auto" else "neutral"
        self.regime_params = REGIME_PARAMS.get(self.regime, REGIME_PARAMS["neutral"])
        # v5.13: 行情模式统计
        self.regime_day_counts = {"bull": 0, "neutral": 0, "bear": 0}

        # 模拟持仓状态
        self.cash = initial_capital
        self.batches = []  # [{id, buy_date, buy_idx, amount, nav, shares, status, is_supplement, peak_nav}]
        self.supplement_count = 0
        self.cooldown_until_idx = -1  # 冷却期截止的 nav_data 索引
        self.cooldown_sell_idx = -1
        self.last_sell_idx = -1
        self.last_nz_sell_idx = -1  # v5.9 P1: 扭亏止盈冷却追踪（上次扭亏止盈的day_idx）

        # v5.11 重构P0: 延迟回补追踪
        # 止盈后不立即买入，而是记录触发条件，后续净值回落到触发价时才执行
        # [{trigger_nav, amount, expire_idx, signal_name}]
        self.pending_rebuys = []

        # 交易记录
        self.trades = []  # [{day_idx, date, action, signal_name, amount/shares, nav, reason}]
        # 每日快照（用于资金曲线）
        self.daily_snapshots = []  # [{date, nav, cash, position_value, total_value, holding_batches}]

        # 统计
        self.total_invested = 0.0
        self.total_received = 0.0
        self.realized_pnl = 0.0
        self.buy_count = 0
        self.sell_count = 0
        self.max_drawdown_pct = 0.0
        self.peak_value = initial_capital

    def _holding_batches(self):
        return [b for b in self.batches if b["status"] == "holding"]

    def _total_cost(self):
        return sum(b["amount"] for b in self._holding_batches())

    def _total_shares(self):
        return sum(b["shares"] for b in self._holding_batches())

    def _total_profit_pct(self, current_nav):
        holding = self._holding_batches()
        total_cost = sum(b["amount"] for b in holding)
        if total_cost <= 0:
            return 0.0
        total_value = sum(b["shares"] * current_nav for b in holding)
        return round((total_value / total_cost - 1) * 100, 2)

    def _position_value(self, current_nav):
        return sum(b["shares"] * current_nav for b in self._holding_batches())

    def _calc_dynamic_supplement_max(self):
        holding = self._holding_batches()
        if holding:
            sorted_h = sorted(holding, key=lambda b: b["buy_date"])
            first_amount = sorted_h[0]["amount"]
        else:
            first_amount = self.max_position * 0.3
        if first_amount <= 0:
            return SUPPLEMENT_MAX_COUNT_DEFAULT
        dynamic_max = math.ceil(self.max_position / first_amount) - 1
        return max(1, min(SUPPLEMENT_MAX_COUNT_HARD_CAP, dynamic_max))

    def _calc_size_multiplier(self, risk_mul, trend_label, momentum=0):
        size_mul_risk = 1.0 / max(0.8, risk_mul)
        weak_labels = {"中期走弱", "连跌", "偏弱"}
        strong_labels = {"中期走强", "偏强", "连涨"}
        if trend_label in weak_labels:
            size_mul_trend = 0.8
        elif trend_label in strong_labels:
            size_mul_trend = 1.30  # v5.13: 原1.20→1.30
        else:
            size_mul_trend = 1.0
        momentum_adj = 1.0
        if momentum < -0.5:
            momentum_adj = 0.75
        elif momentum < -0.3:
            momentum_adj = 0.85
        raw = size_mul_risk * size_mul_trend * momentum_adj
        return round(max(0.40, min(1.40, raw)), 2)  # v5.13: 上限1.30→1.40

    def _buy(self, day_idx: int, amount: float, signal_name: str, reason: str,
             is_supplement: bool = False, is_rebuy: bool = False):
        """执行买入。is_rebuy=True标记为回补仓位，享受L2止损保护期"""
        nav = self.nav_data[day_idx]["nav"]
        date = self.nav_data[day_idx]["date"]
        amount = round(min(amount, self.cash), 2)
        if amount < 10:  # 最低买入金额
            return
        shares = round(amount / nav, 2)
        batch = {
            "id": f"b{day_idx}",
            "buy_date": date,
            "buy_idx": day_idx,
            "amount": round(amount, 2),
            "nav": nav,
            "shares": shares,
            "status": "holding",
            "is_supplement": is_supplement,
            "is_rebuy": is_rebuy,  # v5.11 P2: 标记回补仓位
            "peak_nav": nav,
        }
        self.batches.append(batch)
        self.cash -= amount
        self.total_invested += amount
        self.buy_count += 1
        if is_supplement:
            self.supplement_count += 1
        self.trades.append({
            "day_idx": day_idx, "date": date, "action": "buy",
            "signal_name": signal_name, "amount": amount, "shares": shares,
            "nav": nav, "reason": reason,
        })

    def _sell(self, day_idx: int, batch, sell_pct: float,
              signal_name: str, reason: str):
        """执行卖出"""
        nav = self.nav_data[day_idx]["nav"]
        date = self.nav_data[day_idx]["date"]
        sell_shares = round(batch["shares"] * sell_pct / 100, 2)
        if sell_shares <= 0:
            return

        hold_days = day_idx - batch["buy_idx"]
        fee_rate = get_sell_fee_rate(hold_days, self.fee_schedule)
        gross = round(sell_shares * nav, 2)
        fee = round(gross * fee_rate / 100, 2)
        net = round(gross - fee, 2)
        cost_ratio = sell_shares / batch["shares"] if batch["shares"] > 0 else 1.0
        cost = round(batch["amount"] * cost_ratio, 2)
        profit = round(net - cost, 2)

        if abs(sell_pct - 100) < 0.1 or abs(sell_shares - batch["shares"]) < 0.01:
            batch["status"] = "sold"
        else:
            batch["shares"] = round(batch["shares"] - sell_shares, 2)
            batch["amount"] = round(batch["amount"] * (1 - cost_ratio), 2)

        self.cash += net
        self.total_received += net
        self.realized_pnl += profit
        self.sell_count += 1
        self.cooldown_until_idx = day_idx + COOLDOWN_DAYS + 1  # 简化：用自然日索引近似
        self.cooldown_sell_idx = day_idx
        self.last_sell_idx = day_idx

        # 清仓重置补仓计数
        if not self._holding_batches():
            self.supplement_count = 0

        self.trades.append({
            "day_idx": day_idx, "date": date, "action": "sell",
            "signal_name": signal_name, "shares": sell_shares,
            "nav": nav, "fee_rate": fee_rate, "fee": fee,
            "net": net, "profit": profit, "sell_pct": sell_pct,
            "reason": reason,
        })

    def _get_history_window(self, day_idx: int, window: int = 20) -> List[dict]:
        """获取当前日期向前回看的净值数据（降序，最新在前，用于趋势分析）"""
        start = max(0, day_idx - window)
        data = self.nav_data[start:day_idx + 1]
        return list(reversed(data))

    def _get_hist_changes(self, day_idx: int, window: int = 20) -> List[float]:
        """获取历史涨跌幅（不含当天，降序）"""
        start = max(0, day_idx - window)
        changes = []
        for i in range(day_idx - 1, start - 1, -1):
            c = self.nav_data[i].get("change")
            if c is not None:
                changes.append(c)
        return changes

    def _is_in_cooldown(self, day_idx: int) -> bool:
        """冷却期判断（简化版：用索引间隔近似交易日）"""
        if self.cooldown_sell_idx < 0:
            return False
        gap = day_idx - self.cooldown_sell_idx
        return gap < COOLDOWN_DAYS + 1  # +1 因为是交易日间隔

    def _auto_detect_regime(self, trend_ctx: dict) -> str:
        """
        v5.17: 只检测熊市（牛市参数=基线，检测无意义）
        """
        long_20d = trend_ctx.get("long_20d")
        trend_label = trend_ctx.get("trend_label", "震荡")

        if (long_20d is not None and long_20d < -10
                and trend_label in ("连跌", "中期走弱")):
            return "bear"

        return "neutral"

    def _update_regime(self, trend_ctx: dict):
        """v5.13: 动态更新行情模式（仅regime_mode="auto"时切换）"""
        if self.regime_mode == "auto":
            detected = self._auto_detect_regime(trend_ctx)
            self.regime = detected
            self.regime_params = REGIME_PARAMS.get(detected, REGIME_PARAMS["neutral"])
        self.regime_day_counts[self.regime] = self.regime_day_counts.get(self.regime, 0) + 1

    def run(self, start_idx: int = 20, verbose: bool = False) -> dict:
        """
        运行回测主循环。
        start_idx: 从第几个交易日开始（前面的数据用于计算技术指标）
        """
        for day_idx in range(start_idx, len(self.nav_data)):
            day = self.nav_data[day_idx]
            current_nav = day["nav"]
            today_change = day.get("change") or 0.0
            date = day["date"]

            # 更新 peak_nav
            for b in self._holding_batches():
                if current_nav > b.get("peak_nav", b["nav"]):
                    b["peak_nav"] = current_nav

            # 趋势分析
            hist_changes = self._get_hist_changes(day_idx)
            nav_window = self._get_history_window(day_idx, 60)
            trend_ctx = bt_analyze_trend(today_change, hist_changes, nav_window)
            dyn = bt_calc_dynamic_thresholds(trend_ctx)

            # v5.13: 动态行情模式更新（auto模式逐日切换）
            self._update_regime(trend_ctx)

            vol_state = dyn["vol_state"]
            momentum = dyn.get("momentum_score", 0)
            in_cooldown = self._is_in_cooldown(day_idx)
            risk_mul = dyn["risk_multiplier"]
            size_mul = self._calc_size_multiplier(
                risk_mul, trend_ctx.get("trend_label", "震荡"), momentum)
            # v5.13: 行情模式覆盖size_mul上限
            size_mul = min(size_mul, self.regime_params["size_mul_cap"])

            # v5.13: 行情模式覆盖补仓档位首次建仓比例
            if self.regime != "neutral":
                _adj_tiers = dyn.get("supplement_tiers", [])
                if _adj_tiers:
                    _adj_tiers[0] = (_adj_tiers[0][0], self.regime_params["first_build_ratio"],
                                     _adj_tiers[0][2], _adj_tiers[0][3])
                    dyn["supplement_tiers"] = _adj_tiers

            # v5.11 重构P0: 执行延迟回补——检查pending_rebuys是否触发
            # 每日开始时检查: 净值≤trigger_nav → 执行回补（有安全边际）
            # 超过expire_idx → 放弃回补（窗口失效）
            _new_pending = []
            for pr in self.pending_rebuys:
                if day_idx > pr["expire_idx"]:
                    continue  # 窗口过期，放弃
                if current_nav <= pr["trigger_nav"]:
                    # 触发! 净值已回落到安全边际价格
                    _actual_amount = round(min(pr["amount"], self.cash), 2)
                    if _actual_amount >= 10:
                        self._buy(day_idx, _actual_amount,
                                  pr["signal_name"],
                                  f"净值{current_nav:.4f}≤触发价{pr['trigger_nav']:.4f}"
                                  f"(卖出价{pr['sell_nav']:.4f}), 延迟回补{_actual_amount:.0f}元",
                                  is_supplement=False, is_rebuy=True)
                    continue  # 已执行，不保留
                _new_pending.append(pr)  # 未触发且未过期，保留
            self.pending_rebuys = _new_pending

            # ========== 有持仓逻辑 ==========
            holding = self._holding_batches()
            if holding:
                total_profit_pct = self._total_profit_pct(current_nav)
                batches_sorted = sorted(holding, key=lambda b: b["buy_date"])

                executed_sell = False  # 当日是否已执行卖出

                for batch in batches_sorted:
                    if executed_sell:
                        break
                    hold_days = day_idx - batch["buy_idx"]
                    fee_rate = get_sell_fee_rate(hold_days, self.fee_schedule)
                    profit_pct = round((current_nav / batch["nav"] - 1) * 100, 2) if batch["nav"] > 0 else 0.0

                    # --- 三级止损 ---
                    if hold_days >= 7:
                        effective_stop = dyn["stop_loss_adj"] - fee_rate
                        # L3 极端止损（v5.8 重构P0: 分级减仓）——回补仓位不豁免L3
                        extreme_threshold = effective_stop * STOP_LOSS_L3_FACTOR
                        consec_down = trend_ctx.get("consecutive_down", 0)
                        if profit_pct <= extreme_threshold or (profit_pct <= effective_stop and consec_down >= STOP_LOSS_L3_CONSEC_DOWN):
                            # v5.9 重构P3: L3分级减仓窗口扩大——<25天+>-4%
                            l3_sell_pct = 100  # 默认清仓
                            if hold_days < 25 and today_change > -4.0:
                                l3_sell_pct = 70  # v5.9 重构P3: 持仓<25天+非极端暴跌→减仓70%
                            self._sell(day_idx, batch, l3_sell_pct, "极端止损(L3)",
                                       f"浮亏{profit_pct}%, 止损线{extreme_threshold:.1f}%, 减仓{l3_sell_pct}%")
                            executed_sell = True
                            continue

                        # v5.11 重构P2: 回补仓位L2止损保护期
                        # 回补买在止盈价附近（安全边际仅2-3%），正常波动就触发L2
                        # 保护期内(10天)跳过L2，只走L3/灾难保护（极端风险不豁免）
                        _is_rebuy_batch = batch.get("is_rebuy", False)
                        if _is_rebuy_batch and hold_days < 10:
                            pass  # v5.11 P2: 回补仓位<10天，跳过L2止损
                        else:
                            # v5.8 重构P0: L2止损线时间梯度（<15天→<10天）+震荡市放宽
                            l2_stop = effective_stop  # 基础止损线

                            # v5.7 重构: 震荡市放宽（无方向波动不应触发方向性止损）
                            trend_label_l2 = trend_ctx.get("trend_label", "震荡")
                            if trend_label_l2 == "震荡":
                                l2_stop = round(l2_stop * 1.25, 2)  # v5.7: 震荡市止损线放宽25%

                            if hold_days < 10:  # v5.8 重构P0: 从<15天→<10天
                                # v5.8 重构P0: 持仓<10天，L2需趋势确认(连跌≥3天)才触发
                                if profit_pct <= l2_stop and consec_down >= 3:
                                    l2_sell_pct = max(30, self.regime_params["l2_stop_loss_base"] - self.supplement_count * 10)
                                    self._sell(day_idx, batch, l2_sell_pct, "常规止损(L2)",
                                               f"浮亏{profit_pct}%≤止损线{l2_stop:.1f}%, 持仓{hold_days}天, 连跌{consec_down}天确认, 减仓{l2_sell_pct}%")
                                    executed_sell = True
                                    continue
                                elif profit_pct <= l2_stop:
                                    pass  # v5.7: 触及止损线但趋势未确认，不触发
                            else:
                                if hold_days <= 30:
                                    # v5.8 重构P0: 持仓10-30天（原15-30天），止损线放宽×1.3
                                    l2_stop = round(l2_stop * 1.3, 2)

                                # L2 常规止损
                                if profit_pct <= l2_stop:
                                    l2_sell_pct = max(30, self.regime_params["l2_stop_loss_base"] - self.supplement_count * 10)
                                    self._sell(day_idx, batch, l2_sell_pct, "常规止损(L2)",
                                               f"浮亏{profit_pct}%≤止损线{l2_stop:.1f}%, 减仓{l2_sell_pct}%")
                                    executed_sell = True
                                    continue

                    # --- 灾难保护（<7天）---
                    if hold_days < 7:
                        va = dyn.get("_va", {})
                        disaster_loss = va.get("disaster_loss", DEFAULT_DISASTER_LOSS)
                        disaster_daily = va.get("disaster_daily", DEFAULT_DISASTER_DAILY_DROP)
                        effective_disaster = min(disaster_loss, dyn["stop_loss_adj"] * 1.5)

                        if profit_pct <= effective_disaster:
                            sell_pct = max(30, DISASTER_SELL_PCT_EXTREME - self.supplement_count * 10)
                            self._sell(day_idx, batch, sell_pct, "灾难保护",
                                       f"仅{hold_days}天, 亏{profit_pct}%≤灾难线{effective_disaster}%")
                            executed_sell = True
                            continue

                        if (today_change <= disaster_daily
                                and trend_ctx.get("consecutive_down", 0) >= DISASTER_CONSECUTIVE_DOWN):
                            sell_pct = max(20, DISASTER_SELL_PCT_DAILY - self.supplement_count * 5)
                            self._sell(day_idx, batch, sell_pct, "灾难保护(暴跌)",
                                       f"仅{hold_days}天, 今日暴跌{today_change}%+连跌")
                            executed_sell = True
                            continue

                        # v5.5 短期深亏安全网（门槛从-6%→-8%）
                        if profit_pct <= -8.0:
                            self._sell(day_idx, batch, 30, "短期深亏安全网",
                                       f"仅{hold_days}天, 亏{profit_pct}%超6%")
                            executed_sell = True
                            continue
                        continue  # <7天不走止盈

                    # --- 统一止盈评分（简化版）---
                    # v5.5 优化: 持仓不足15个交易日不触发止盈（除非盈利>10%）
                    if hold_days < 15 and profit_pct < 10.0:
                        pass  # 持仓不足15日，跳过止盈
                    elif profit_pct > fee_rate * 2.0:
                        vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.2
                        # v5.5 优化: profit_norm大幅提高
                        profit_norm = max(8.0, vol * 8.0)
                        profit_score = math.tanh(profit_pct / profit_norm) * 40

                        # 回撤止盈
                        trail_score = 0
                        peak_profit = round((batch.get("peak_nav", batch["nav"]) / batch["nav"] - 1) * 100, 2)
                        if peak_profit > 3.0 and peak_profit > profit_pct:
                            dd = peak_profit - profit_pct
                            trail_dd_threshold = dyn.get("trail_dd", TRAIL_DD_BASE)
                            if dd >= trail_dd_threshold:
                                trail_score = min(30, dd / trail_dd_threshold * 15)

                        momentum_score = max(0, -momentum * 15)
                        fee_drag = -fee_rate * 5

                        # v5.10 重构P1: 撤销v5.9的hold_time_bonus
                        # v5.9 hold_time_bonus是全部恶化的罪魁祸首:
                        # 止盈卖出-67%, 强势止盈-100%, 利润-172k

                        # v5.10 重构P0: 扭亏奖励分（将扭亏止盈纳入评分系统）
                        # v5.11 重构P1: 降低nz_bonus——v5.10的min(20,tpp*3)过于慷慨
                        # 导致扭亏18362次(+50%)，降低为min(12,tpp*2)
                        nz_bonus = 0.0
                        if (total_profit_pct > 0 and len(batches_sorted) >= 2
                                and batch["id"] == batches_sorted[0]["id"]):
                            nz_bonus = min(12.0, total_profit_pct * 2.0)
                            oldest_hd_nz = day_idx - batches_sorted[0]["buy_idx"]
                            if oldest_hd_nz < 15:
                                nz_bonus *= 0.5

                        total_score = profit_score + trail_score + momentum_score + fee_drag + nz_bonus

                        sell_pct = 0
                        signal_name = None
                        # v5.8 重构P2: 分批止盈50→55，慢涨止盈45→52
                        if total_score >= 75:
                            sell_pct, signal_name = 100, "强势止盈"
                        elif total_score >= 60:
                            sell_pct, signal_name = 70, "止盈卖出"
                        elif total_score >= 55:  # v5.8 重构P2: 50→55
                            sell_pct, signal_name = 50, "分批止盈"
                        elif total_score >= 53:  # v5.13: 52→53
                            sell_pct, signal_name = 30, "慢涨止盈"

                        # v5.13: 止盈抑制门槛由行情模式控制
                        trend_label = trend_ctx.get("trend_label", "震荡")
                        _tp_suppress = self.regime_params["tp_suppress_threshold"]
                        if trend_label in ("连涨", "偏强", "中期走强") and total_score < _tp_suppress:
                            sell_pct = 0
                            signal_name = None

                        if sell_pct > 0:
                            # v5.10 重构P0: 扭亏止盈走评分通道，标记信号名称
                            effective_signal_name = signal_name
                            if nz_bonus > 0 and batch["id"] == batches_sorted[0]["id"]:
                                effective_signal_name = f"扭亏{signal_name}"

                            self._sell(day_idx, batch, sell_pct, effective_signal_name,
                                       f"浮盈{profit_pct}%, 评分{total_score:.0f}, 减仓{sell_pct}%"
                                       + (f", 扭亏加分{nz_bonus:.0f}" if nz_bonus > 0 else ""))

                            # v5.11 重构P0: 延迟回补——止盈后不立即买入，记录触发条件
                            # v5.10直接回补导致: 买在止盈价(无安全边际)→L2止损+69%→中位持仓3.5天
                            # v5.11: 记录pending_rebuy，后续净值跌到trigger_nav才执行
                            # v5.11 重构P3: 撤回弱趋势回补(下跌中买入注定被止损)
                            if (effective_signal_name in ("分批止盈", "慢涨止盈", "止盈卖出", "强势止盈",
                                                          "扭亏分批止盈", "扭亏慢涨止盈", "扭亏止盈卖出", "扭亏强势止盈")):
                                _rebuy_trend = trend_ctx.get("trend_label", "震荡")
                                _rebuy_ratio = 0.0
                                _rebuy_discount = 0.0
                                _regime_rebuy_discount = self.regime_params["rebuy_discount"]  # v5.13
                                if _regime_rebuy_discount <= 0:
                                    pass  # v5.13: 熊市模式不回补
                                elif _rebuy_trend in ("连涨", "偏强", "中期走强"):
                                    _rebuy_ratio = 0.70   # v5.13: 60%→70%
                                    _rebuy_discount = _regime_rebuy_discount  # v5.13
                                elif _rebuy_trend == "震荡" and today_change >= -0.5:
                                    _rebuy_ratio = 0.40   # v5.13: 35%→40%
                                    _rebuy_discount = _regime_rebuy_discount + 0.010  # v5.13
                                # v5.11 P3: 弱趋势不回补（撤回v5.10的20%）

                                if _rebuy_ratio > 0:
                                    _sell_amount_est = batch["shares"] * sell_pct / 100 * current_nav
                                    _rebuy_amount = round(_sell_amount_est * _rebuy_ratio * size_mul, 2)
                                    _rebuy_amount = round(min(_rebuy_amount, self.cash), 2)
                                    if _rebuy_amount >= 10:
                                        # v5.11 P0: 不立即买入，记录延迟回补
                                        _trigger_nav = round(current_nav * (1 - _rebuy_discount), 4)
                                        self.pending_rebuys.append({
                                            "trigger_nav": _trigger_nav,
                                            "amount": _rebuy_amount,
                                            "expire_idx": day_idx + 20,  # 20天窗口
                                            "signal_name": f"延迟回补({effective_signal_name})",
                                            "sell_nav": current_nav,
                                        })

                            executed_sell = True
                            continue

                    # --- 趋势转弱卖出 ---
                    vol = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
                    min_profit_buffer = max(1.5, fee_rate * 2.5 + max(0.3, vol * 0.5))
                    if profit_pct > min_profit_buffer and hold_days >= 7:
                        # v5.5: 需3天连跌确认（原2天）
                        if len(hist_changes) >= 3 and hist_changes[0] < 0 and hist_changes[1] < 0 and hist_changes[2] < 0:
                            cumulative_drop = hist_changes[0] + hist_changes[1] + hist_changes[2]
                            tw_thresh = dyn.get("trend_weak_cumulative", DEFAULT_TREND_WEAK_CUMULATIVE) * 1.5  # v5.5: 门槛提高50%
                            short_5d_val = trend_ctx.get("short_5d")
                            mid_10d_val = trend_ctx.get("mid_10d")
                            if cumulative_drop <= tw_thresh:
                                if (short_5d_val is not None and short_5d_val < -2) or \
                                   (mid_10d_val is not None and mid_10d_val < -3):  # v5.5: 更严格
                                    vp = trend_ctx.get("volume_proxy")
                                    if vp is None or vp >= 0.5:
                                        # v5.5 分级减仓（降低比例）
                                        if total_profit_pct < 1:
                                            tw_sell_pct = 70   # v5.5: 原100→70
                                        elif total_profit_pct < 3:
                                            tw_sell_pct = 50   # v5.5: 原70→50
                                        else:
                                            tw_sell_pct = 30   # v5.5: 原50→30
                                        if vp and vp > 1.5:
                                            tw_sell_pct = min(100, tw_sell_pct + 20)
                                        self._sell(day_idx, batch, tw_sell_pct, "趋势转弱",
                                                   f"浮盈{profit_pct}%, 总浮盈{total_profit_pct}%, 趋势转弱减仓{tw_sell_pct}%")
                                        executed_sell = True
                                        continue

                if executed_sell:
                    pass  # 已处理
                else:
                    # --- v5.10 重构P0: 扭亏止盈已纳入评分系统(nz_bonus)，不再独立触发 ---
                    # 旧逻辑(v5.4~v5.9): total_profit>0+≥2批 → 独立触发扭亏止盈
                    # 这是六版水床效应根源(v5.9: 12277次占TOP1)
                    # v5.10: 扭亏加分已在上面的sell_eval中处理，通过评分门槛才能触发
                    holding_now = self._holding_batches()

                    # --- 总仓位风控（v5.6 重构: 废除固定止损线死循环，改为"持仓时间豁免+趋势确认减仓+灾难保底"三层架构） ---
                    if not executed_sell and holding_now and total_profit_pct < 0:
                        oldest = sorted(holding_now, key=lambda b: b["buy_idx"])[0]
                        oldest_hd = day_idx - oldest["buy_idx"]

                        # v5.6 重构: 第一层——灾难保底（total_profit_pct <= -15%无条件减仓）
                        # 原因: 真正的灾难不能等，但-15%在正常波动中几乎不会触发，消除死循环
                        if total_profit_pct <= -15.0 and oldest_hd >= 7:
                            if total_profit_pct <= -25.0:
                                psl_pct = 70  # v5.6: 极端灾难
                            elif total_profit_pct <= -20.0:
                                psl_pct = 50  # v5.6: 严重灾难
                            else:
                                psl_pct = 35  # v5.6: 灾难保底
                            self._sell(day_idx, oldest, psl_pct, "灾难保底减仓",
                                       f"总浮亏{total_profit_pct}%≤-15%灾难线, 减仓{psl_pct}%")
                            executed_sell = True

                        # v5.6 重构: 第二层——持仓时间豁免（最老批次<20天完全不触发总仓位减仓）
                        # 原因: 新建仓需要时间度过正常波动，<20天内的浮亏是网格策略的正常成本
                        elif oldest_hd < 20:
                            pass  # 豁免期内不做总仓位减仓，只依赖单批L2/L3止损

                        # v5.6 重构: 第三层——趋势确认减仓（持仓≥20天后，需要趋势恶化双确认才减仓）
                        # 原因: 不再用固定止损线，改用趋势判断——只有确认趋势转坏才减仓
                        elif oldest_hd >= 20:
                            trend_label = trend_ctx.get("trend_label", "震荡")
                            consecutive_down_tc = trend_ctx.get("consecutive_down", 0)
                            mid_10d_tc = trend_ctx.get("mid_10d")

                            # v5.6 重构: 趋势恶化需要多重确认
                            trend_confirm_count = 0

                            # 条件A: 连跌确认（≥6天）
                            if consecutive_down_tc >= 6:  # v5.13: 5→6
                                trend_confirm_count += 1

                            # 条件B: 中期走弱确认（10日累跌≥-10%）
                            if mid_10d_tc is not None and mid_10d_tc <= -10:  # v5.13: -8→-10
                                trend_confirm_count += 1

                            # 条件C: 趋势标签确认
                            if trend_label in ("连跌", "中期走弱"):
                                trend_confirm_count += 1

                            trend_deteriorating = trend_confirm_count >= 2  # v5.6: 至少2个条件双确认

                            # v5.6 重构: 补仓已用尽仍深亏，降低确认门槛
                            dynamic_max_supp_tc = self._calc_dynamic_supplement_max()
                            dynamic_max_supp_tc = min(dynamic_max_supp_tc, self.regime_params["supplement_max_count"])  # v5.13
                            remaining_supp = max(0, dynamic_max_supp_tc - self.supplement_count)
                            if remaining_supp == 0 and total_profit_pct <= -8.0 and trend_confirm_count >= 1:
                                trend_deteriorating = True

                            if trend_deteriorating:
                                # v5.6 重构: 按亏损严重程度分级减仓
                                if total_profit_pct <= -12.0:
                                    psl_pct = 50
                                elif total_profit_pct <= -8.0:
                                    psl_pct = 35
                                elif total_profit_pct <= -6.0:  # v5.13: -5→-6
                                    psl_pct = 25
                                else:
                                    psl_pct = 0  # 浅亏不减仓
                                if psl_pct > 0:
                                    self._sell(day_idx, oldest, psl_pct, "趋势确认减仓",
                                               f"总浮亏{total_profit_pct}%, 持仓{oldest_hd}天, 趋势恶化确认, 减仓{psl_pct}%")
                                    executed_sell = True

                    # --- 递进补仓 ---
                    if not executed_sell and not in_cooldown and holding_now:
                        dynamic_max_supp = self._calc_dynamic_supplement_max()
                        dynamic_max_supp = min(dynamic_max_supp, self.regime_params["supplement_max_count"])  # v5.13
                        total_cost = self._total_cost()
                        # 补仓禁入
                        forbidden = False
                        if vol_state == "extreme_vol":
                            forbidden = True
                        mid_10d = trend_ctx.get("mid_10d")
                        consec_down = trend_ctx.get("consecutive_down", 0)
                        if mid_10d is not None and mid_10d <= -10 and consec_down >= 5:  # v5.5: -7/3→-10/5
                            forbidden = True

                        # v5.8 重构P3: 豁免期内趋势恶化冻结补仓（放宽条件: 3/5→4/7）
                        if not forbidden and holding_now:
                            oldest_supp = sorted(holding_now, key=lambda b: b["buy_idx"])[0]
                            oldest_supp_hd = day_idx - oldest_supp["buy_idx"]
                            if oldest_supp_hd < 20:  # 在20天豁免期内
                                short_5d_supp = trend_ctx.get("short_5d")
                                if consec_down >= 4 and short_5d_supp is not None and short_5d_supp <= -7:  # v5.8 重构P3: 3/5→4/7
                                    forbidden = True

                        if (not forbidden
                                and self.supplement_count < dynamic_max_supp
                                and total_cost < self.max_position):
                            # 补仓节奏阀（简化版：检查间隔）
                            last_supp_idx = -999
                            for b in self._holding_batches():
                                if b.get("is_supplement") and b["buy_idx"] > last_supp_idx:
                                    last_supp_idx = b["buy_idx"]
                            # 也检查所有买入（深亏时）
                            if total_profit_pct < -3.0:
                                for b in self._holding_batches():
                                    if b["buy_idx"] > last_supp_idx:
                                        last_supp_idx = b["buy_idx"]

                            gap = day_idx - last_supp_idx if last_supp_idx >= 0 else 999
                            # v5.3 动态间隔
                            vol_r = trend_ctx.get("volatility_robust") or 1.0
                            short_3d = trend_ctx.get("short_3d") or 0
                            dynamic_gap = SUPPLEMENT_MIN_GAP_TRADE_DAYS
                            if abs(short_3d) > vol_r * 2 and short_3d < 0:
                                dynamic_gap = max(2, dynamic_gap - 1)
                            elif consec_down >= 5:
                                dynamic_gap = min(5, dynamic_gap + 2)

                            if gap >= dynamic_gap:
                                adj_tiers = dyn.get("supplement_tiers", SUPPLEMENT_TIERS)
                                for tier_count, tier_ratio, tier_trigger, tier_loss_min in adj_tiers:
                                    if self.supplement_count == tier_count:
                                        if (total_profit_pct <= tier_loss_min
                                                and today_change <= tier_trigger):
                                            risk_budget = self.max_position - total_cost
                                            supplement_amount = round(risk_budget * tier_ratio * size_mul, 2)
                                            cap = self.max_position * SUPPLEMENT_CAP_RATIO
                                            supplement_amount = round(min(supplement_amount, cap, risk_budget, self.cash), 2)
                                            if supplement_amount >= 10:
                                                self._buy(day_idx, supplement_amount,
                                                          f"补仓(第{self.supplement_count+1}次)",
                                                          f"总浮亏{total_profit_pct}%, 今跌{today_change}%, 补仓{supplement_amount}元",
                                                          is_supplement=True)
                                        break

            # ========== 空仓逻辑 ==========
            else:
                if vol_state == "extreme_vol":
                    pass  # 极端波动禁买
                elif not in_cooldown:
                    dip_threshold = dyn["dip_threshold"]

                    # 大跌抄底
                    if today_change <= dip_threshold:
                        buy_amount = round(self.max_position * 0.80 * size_mul, 2)  # v5.13: 0.70→0.80
                        vp = trend_ctx.get("volume_proxy")
                        if vp and vp < 0.5:
                            buy_amount = round(buy_amount * 0.6, 2)
                        self._buy(day_idx, buy_amount, "大跌抄底",
                                  f"跌{today_change}%≤阈值{dip_threshold}%")

                    # 趋势建仓
                    elif trend_ctx.get("mid_10d") is not None:
                        mid_10d = trend_ctx["mid_10d"]
                        short_5d = trend_ctx.get("short_5d")
                        consec_down = trend_ctx.get("consecutive_down", 0)

                        if mid_10d <= TREND_BUILD_TRIGGER_10D and today_change >= -0.5:
                            buy_amount = round(self.max_position * 0.55 * size_mul, 2)  # v5.5: 0.4→0.55
                            self._buy(day_idx, buy_amount, "低位建仓",
                                      f"10日累跌{mid_10d}%, 今日企稳")
                        elif short_5d is not None and short_5d <= TREND_BUILD_TRIGGER_5D and today_change > 0:
                            buy_amount = round(self.max_position * 0.45 * size_mul, 2)  # v5.5: 0.3→0.45
                            self._buy(day_idx, buy_amount, "反弹建仓",
                                      f"5日累跌{short_5d}%, 今日反弹")
                        elif (consec_down >= 3 and today_change < 0
                              and len(hist_changes) >= 1
                              and abs(today_change) < abs(hist_changes[0]) * 0.6):
                            buy_amount = round(self.max_position * 0.35 * size_mul, 2)  # v5.5: 0.25→0.35
                            self._buy(day_idx, buy_amount, "跌势放缓建仓",
                                      f"连跌{consec_down}天, 跌幅收窄")

                    # v5.13新增: 温和回调建仓（在连跌低吸之前）
                    if not self._holding_batches() and not in_cooldown:
                        vol_for_mild = trend_ctx.get("volatility_robust") or trend_ctx.get("volatility") or 1.0
                        short_3d_mild = trend_ctx.get("short_3d")
                        if (short_3d_mild is not None
                                and short_3d_mild < 0
                                and abs(short_3d_mild) > vol_for_mild
                                and vol_state != "extreme_vol"):
                            buy_amount = round(self.max_position * 0.45 * size_mul, 2)
                            self._buy(day_idx, buy_amount, "温和回调建仓",
                                      f"3日累跌{short_3d_mild}%>波动率{vol_for_mild:.1f}%")

                    # 连跌低吸
                    if not self._holding_batches() and not in_cooldown:
                        consec_dip_thresh = dyn.get("consecutive_dip_trigger", DEFAULT_CONSECUTIVE_DIP_TRIGGER)
                        if (today_change <= consec_dip_thresh
                                and len(hist_changes) >= 1 and hist_changes[0] < 0):
                            buy_amount = round(self.max_position * 0.45 * size_mul, 2)  # v5.5: 0.3→0.45
                            self._buy(day_idx, buy_amount, "连跌低吸",
                                      f"今跌{today_change}%≤{consec_dip_thresh}%, 昨跌{hist_changes[0]}%")

                    # 冷却期后建仓
                    if (not self._holding_batches()
                            and not in_cooldown
                            and self.cooldown_sell_idx >= 0
                            and today_change <= 0.3):  # v5.13: 0→0.3
                        short_5d_cd = trend_ctx.get("short_5d")
                        consec_down_cd = trend_ctx.get("consecutive_down", 0)
                        trend_ok = True
                        if short_5d_cd is not None and short_5d_cd <= -5 and consec_down_cd >= 3:
                            trend_ok = False
                        if trend_ok:
                            buy_amount = round(self.max_position * 0.50 * size_mul, 2)  # v5.5: 0.3→0.50
                            self._buy(day_idx, buy_amount, "冷却期后建仓",
                                      f"冷却结束, 今{today_change}%")

            # 记录每日快照
            pos_val = self._position_value(current_nav)
            total_val = self.cash + pos_val
            if total_val > self.peak_value:
                self.peak_value = total_val
            dd = (self.peak_value - total_val) / self.peak_value * 100 if self.peak_value > 0 else 0
            if dd > self.max_drawdown_pct:
                self.max_drawdown_pct = dd

            self.daily_snapshots.append({
                "date": date,
                "nav": current_nav,
                "cash": round(self.cash, 2),
                "position_value": round(pos_val, 2),
                "total_value": round(total_val, 2),
                "holding_count": len(self._holding_batches()),
            })

        # 返回结果摘要
        return self._build_summary()

    def _build_summary(self) -> dict:
        if not self.daily_snapshots:
            return {"error": "无回测数据"}

        final = self.daily_snapshots[-1]
        start_snap = self.daily_snapshots[0]
        final_value = final["total_value"]
        total_return = round((final_value / self.initial_capital - 1) * 100, 2)

        # 买入持有基准收益
        first_nav = self.nav_data[20]["nav"] if len(self.nav_data) > 20 else self.nav_data[0]["nav"]
        last_nav = self.nav_data[-1]["nav"]
        buy_hold_return = round((last_nav / first_nav - 1) * 100, 2)

        # 年化收益
        days = len(self.daily_snapshots)
        annual_return = round((((final_value / self.initial_capital) ** (252 / max(1, days))) - 1) * 100, 2)
        bh_annual = round((((last_nav / first_nav) ** (252 / max(1, days))) - 1) * 100, 2)

        # 胜率统计
        buy_trades = [t for t in self.trades if t["action"] == "buy"]
        sell_trades = [t for t in self.trades if t["action"] == "sell"]
        win_sells = [t for t in sell_trades if t.get("profit", 0) > 0]
        lose_sells = [t for t in sell_trades if t.get("profit", 0) < 0]
        win_rate = round(len(win_sells) / len(sell_trades) * 100, 1) if sell_trades else 0.0
        avg_win = round(sum(t["profit"] for t in win_sells) / len(win_sells), 2) if win_sells else 0
        avg_lose = round(sum(t["profit"] for t in lose_sells) / len(lose_sells), 2) if lose_sells else 0

        # 信号分布
        signal_dist = {}
        for t in self.trades:
            name = t.get("signal_name", "unknown")
            signal_dist[name] = signal_dist.get(name, 0) + 1

        # === v5.4 新增：资深基金经理级诊断数据 ===

        # --- 1. 资金利用率：平均仓位占比 ---
        total_pos_pct = 0
        holding_days_count = 0  # 有持仓的天数
        for snap in self.daily_snapshots:
            pos_ratio = snap["position_value"] / snap["total_value"] * 100 if snap["total_value"] > 0 else 0
            total_pos_pct += pos_ratio
            if snap["holding_count"] > 0:
                holding_days_count += 1
        avg_position_pct = round(total_pos_pct / max(1, days), 1)
        holding_day_ratio = round(holding_days_count / max(1, days) * 100, 1)
        empty_day_ratio = round(100 - holding_day_ratio, 1)

        # --- 2. 持仓时间分布 ---
        hold_durations = []
        for t in sell_trades:
            hd = t.get("day_idx", 0) - next(
                (b["buy_idx"] for b in self.batches if b["id"] == f"b{t.get('day_idx', 0)}"
                 ), t.get("day_idx", 0))
            # 从trades中推断hold_days
            reason = t.get("reason", "")
            hold_durations.append(max(1, hd))
        # 更精确的方式：从已sold的batch计算
        sold_batches = [b for b in self.batches if b["status"] == "sold"]
        batch_hold_days = []
        for b in sold_batches:
            # 找对应的sell trade
            for t in sell_trades:
                if t.get("day_idx", -1) >= b["buy_idx"]:
                    batch_hold_days.append(t["day_idx"] - b["buy_idx"])
                    break
        if not batch_hold_days:
            batch_hold_days = [0]
        avg_hold_days = round(sum(batch_hold_days) / max(1, len(batch_hold_days)), 1)
        median_hold_days = sorted(batch_hold_days)[len(batch_hold_days) // 2] if batch_hold_days else 0
        max_hold_days = max(batch_hold_days) if batch_hold_days else 0

        # --- 3. 每笔交易盈亏分布 ---
        sell_profits = [t.get("profit", 0) for t in sell_trades]
        if sell_profits:
            sorted_profits = sorted(sell_profits)
            n_p = len(sorted_profits)
            median_profit = sorted_profits[n_p // 2]
            p10_profit = sorted_profits[max(0, int(n_p * 0.1))]
            p90_profit = sorted_profits[min(n_p - 1, int(n_p * 0.9))]
            max_single_win = max(sell_profits)
            max_single_loss = min(sell_profits)
        else:
            median_profit = p10_profit = p90_profit = max_single_win = max_single_loss = 0

        # --- 4. 连续亏损/盈利统计 ---
        max_consec_loss = 0
        max_consec_win = 0
        cur_loss = cur_win = 0
        for t in sell_trades:
            if t.get("profit", 0) < 0:
                cur_loss += 1
                cur_win = 0
                max_consec_loss = max(max_consec_loss, cur_loss)
            elif t.get("profit", 0) > 0:
                cur_win += 1
                cur_loss = 0
                max_consec_win = max(max_consec_win, cur_win)
            else:
                cur_loss = cur_win = 0

        # --- 5. 回撤分析 ---
        max_dd = 0
        max_dd_start = max_dd_end = ""
        dd_duration = 0
        max_dd_duration = 0
        peak_val = self.initial_capital
        peak_date = self.daily_snapshots[0]["date"] if self.daily_snapshots else ""
        dd_start_date = peak_date
        for snap in self.daily_snapshots:
            tv = snap["total_value"]
            if tv >= peak_val:
                peak_val = tv
                peak_date = snap["date"]
                dd_start_date = snap["date"]
                dd_duration = 0
            else:
                dd = (peak_val - tv) / peak_val * 100
                dd_duration += 1
                if dd > max_dd:
                    max_dd = dd
                    max_dd_start = dd_start_date
                    max_dd_end = snap["date"]
                if dd_duration > max_dd_duration:
                    max_dd_duration = dd_duration

        # --- 6. 夏普比率 & 卡尔马比率 ---
        daily_returns = []
        for i in range(1, len(self.daily_snapshots)):
            prev = self.daily_snapshots[i - 1]["total_value"]
            cur = self.daily_snapshots[i]["total_value"]
            if prev > 0:
                daily_returns.append((cur / prev - 1) * 100)
        if daily_returns and len(daily_returns) > 10:
            dr_mean = sum(daily_returns) / len(daily_returns)
            dr_std = (sum((r - dr_mean) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
            sharpe = round(dr_mean / dr_std * (252 ** 0.5), 2) if dr_std > 0 else 0
        else:
            sharpe = 0
        calmar = round(annual_return / max(0.01, self.max_drawdown_pct), 2) if self.max_drawdown_pct > 0 else 0

        # --- 7. 按信号类型统计盈亏 ---
        signal_pnl = {}  # signal_name -> {count, total_profit, wins, losses}
        for t in sell_trades:
            sn = t.get("signal_name", "unknown")
            if sn not in signal_pnl:
                signal_pnl[sn] = {"count": 0, "total_profit": 0, "wins": 0, "losses": 0,
                                  "total_win_amt": 0, "total_loss_amt": 0}
            sp = signal_pnl[sn]
            sp["count"] += 1
            p = t.get("profit", 0)
            sp["total_profit"] += p
            if p > 0:
                sp["wins"] += 1
                sp["total_win_amt"] += p
            elif p < 0:
                sp["losses"] += 1
                sp["total_loss_amt"] += p

        # --- 8. 月度收益率（用于牛熊阶段分析）---
        monthly_returns = {}
        month_start_val = self.daily_snapshots[0]["total_value"]
        current_month = self.daily_snapshots[0]["date"][:7]
        for snap in self.daily_snapshots:
            m = snap["date"][:7]
            if m != current_month:
                monthly_returns[current_month] = round((snap["total_value"] / max(1, month_start_val) - 1) * 100, 2)
                month_start_val = snap["total_value"]
                current_month = m
        # 最后一个月
        if self.daily_snapshots:
            monthly_returns[current_month] = round(
                (self.daily_snapshots[-1]["total_value"] / max(1, month_start_val) - 1) * 100, 2)

        # 月度买持基准收益
        monthly_bh = {}
        nav_by_date = {d["date"]: d["nav"] for d in self.nav_data}
        month_start_nav = first_nav
        current_month_bh = self.daily_snapshots[0]["date"][:7]
        for snap in self.daily_snapshots:
            m = snap["date"][:7]
            nav_now = nav_by_date.get(snap["date"], snap["nav"])
            if m != current_month_bh:
                monthly_bh[current_month_bh] = round((nav_now / max(0.001, month_start_nav) - 1) * 100, 2)
                month_start_nav = nav_now
                current_month_bh = m
        if self.daily_snapshots:
            last_nav_val = nav_by_date.get(self.daily_snapshots[-1]["date"], last_nav)
            monthly_bh[current_month_bh] = round((last_nav_val / max(0.001, month_start_nav) - 1) * 100, 2)

        # 正收益月/负收益月统计
        pos_months = sum(1 for v in monthly_returns.values() if v > 0)
        neg_months = sum(1 for v in monthly_returns.values() if v < 0)
        total_months = len(monthly_returns)
        monthly_win_rate = round(pos_months / max(1, total_months) * 100, 1)

        # 季度聚合
        quarterly_returns = {}
        for ym, ret in monthly_returns.items():
            q = ym[:4] + "Q" + str((int(ym[5:7]) - 1) // 3 + 1)
            quarterly_returns[q] = quarterly_returns.get(q, 0) + ret

        # --- 9. 牛熊阶段划分（基于买持基准的累计走势）---
        # 按累计涨跌幅标记阶段
        cum_bh = 0
        phase_start = self.daily_snapshots[0]["date"]
        phases = []  # [(start, end, bh_return, strat_return, label)]
        phase_peak_cum = 0
        phase_trough_cum = 0
        phase_strat_start_val = self.daily_snapshots[0]["total_value"]
        phase_bh_start_nav = first_nav
        current_phase = "neutral"

        # 简化：按半年切分统计
        half_year_stats = {}
        hy_start_val = self.daily_snapshots[0]["total_value"]
        hy_start_nav = first_nav
        current_hy = self.daily_snapshots[0]["date"][:4] + ("H1" if int(self.daily_snapshots[0]["date"][5:7]) <= 6 else "H2")
        for snap in self.daily_snapshots:
            m = int(snap["date"][5:7])
            hy = snap["date"][:4] + ("H1" if m <= 6 else "H2")
            nav_now = nav_by_date.get(snap["date"], snap["nav"])
            if hy != current_hy:
                bh_ret = round((nav_now / max(0.001, hy_start_nav) - 1) * 100, 2)
                st_ret = round((snap["total_value"] / max(1, hy_start_val) - 1) * 100, 2)
                label = "牛" if bh_ret > 10 else "熊" if bh_ret < -10 else "震荡"
                half_year_stats[current_hy] = {"bh": bh_ret, "strat": st_ret, "label": label}
                hy_start_val = snap["total_value"]
                hy_start_nav = nav_now
                current_hy = hy
        # 最后一个半年
        if self.daily_snapshots:
            last_snap = self.daily_snapshots[-1]
            last_nav_v = nav_by_date.get(last_snap["date"], last_snap["nav"])
            bh_ret = round((last_nav_v / max(0.001, hy_start_nav) - 1) * 100, 2)
            st_ret = round((last_snap["total_value"] / max(1, hy_start_val) - 1) * 100, 2)
            label = "牛" if bh_ret > 10 else "熊" if bh_ret < -10 else "震荡"
            half_year_stats[current_hy] = {"bh": bh_ret, "strat": st_ret, "label": label}

        # 牛市/熊市/震荡分别的累计收益
        bull_strat = sum(v["strat"] for v in half_year_stats.values() if v["label"] == "牛")
        bear_strat = sum(v["strat"] for v in half_year_stats.values() if v["label"] == "熊")
        neutral_strat = sum(v["strat"] for v in half_year_stats.values() if v["label"] == "震荡")
        bull_bh = sum(v["bh"] for v in half_year_stats.values() if v["label"] == "牛")
        bear_bh = sum(v["bh"] for v in half_year_stats.values() if v["label"] == "熊")
        neutral_bh = sum(v["bh"] for v in half_year_stats.values() if v["label"] == "震荡")
        bull_count = sum(1 for v in half_year_stats.values() if v["label"] == "牛")
        bear_count = sum(1 for v in half_year_stats.values() if v["label"] == "熊")
        neutral_count = sum(1 for v in half_year_stats.values() if v["label"] == "震荡")

        # --- 10. 波动率特征（判断基金"脾气"）---
        all_changes = [d.get("change") for d in self.nav_data if d.get("change") is not None]
        if all_changes:
            avg_daily_vol = round(sum(abs(c) for c in all_changes) / len(all_changes), 3)
            max_daily_gain = round(max(all_changes), 2)
            max_daily_loss = round(min(all_changes), 2)
            # 均值回归性：涨跌交替的频率（相邻两天符号不同的比例）
            sign_changes = sum(1 for i in range(1, len(all_changes))
                              if all_changes[i] * all_changes[i-1] < 0)
            mean_revert_ratio = round(sign_changes / max(1, len(all_changes) - 1) * 100, 1)
        else:
            avg_daily_vol = max_daily_gain = max_daily_loss = 0
            mean_revert_ratio = 0

        # --- 11. 策略适配度评分（0~100，越高越适合网格策略）---
        # 买持回撤（从最高点的回撤）
        bh_navs = [d["nav"] for d in self.nav_data if d.get("nav")]
        bh_peak = bh_navs[0] if bh_navs else 1
        bh_max_dd = 0
        for nv in bh_navs:
            if nv > bh_peak:
                bh_peak = nv
            dd = (bh_peak - nv) / bh_peak * 100
            if dd > bh_max_dd:
                bh_max_dd = dd

        # 分项打分
        # A. 收益效率分 (0~30): 年化收益/回撤 越高越好
        efficiency = annual_return / max(0.1, self.max_drawdown_pct)
        score_efficiency = round(min(30, max(0, efficiency * 30)), 1)

        # B. 回撤控制分 (0~25): 策略回撤比买持回撤小多少
        dd_control = 1 - self.max_drawdown_pct / max(0.1, bh_max_dd)
        score_dd_control = round(min(25, max(0, (dd_control + 0.2) * 25)), 1)

        # C. 资金效率分 (0~20): 仓位利用率
        score_capital = round(min(20, avg_position_pct / 100 * 20), 1)

        # D. 胜率质量分 (0~15): 月度胜率
        score_winrate = round(min(15, monthly_win_rate / 100 * 15), 1)

        # E. 波动适配分 (0~10): 日均波动0.8~2.0最适合网格
        if 0.8 <= avg_daily_vol <= 2.0:
            score_vol_fit = 10
        elif 0.5 <= avg_daily_vol <= 3.0:
            score_vol_fit = 5
        else:
            score_vol_fit = 0

        fitness_score = round(score_efficiency + score_dd_control + score_capital
                              + score_winrate + score_vol_fit, 1)

        # 适配等级
        if fitness_score >= 70:
            fitness_grade = "A-极佳"
        elif fitness_score >= 55:
            fitness_grade = "B-良好"
        elif fitness_score >= 40:
            fitness_grade = "C-一般"
        elif fitness_score >= 25:
            fitness_grade = "D-偏差"
        else:
            fitness_grade = "E-不适合"

        # 趋势捕获率：策略收益 / 买持收益（牛市看抓住多少涨幅，熊市看躲掉多少跌幅）
        if buy_hold_return > 5:
            capture_rate = round(total_return / max(0.01, buy_hold_return) * 100, 1)
        elif buy_hold_return < -5:
            # 熊市：策略少亏就是赢
            capture_rate = round((1 - total_return / min(-0.01, buy_hold_return)) * 100, 1)
        else:
            capture_rate = 100.0 if total_return >= 0 else 0.0

        # 市场环境适配标签
        bull_excess = bull_strat - bull_bh if bull_count > 0 else 0
        bear_excess = bear_strat - bear_bh if bear_count > 0 else 0
        env_tags = []
        if bear_count > 0 and bear_excess > 0:
            env_tags.append("熊市优")
        if bull_count > 0 and bull_excess > -20:
            env_tags.append("牛市可")
        if neutral_count > 0 and neutral_strat > 0:
            env_tags.append("震荡优")
        if len(env_tags) >= 2:
            env_tags = ["全天候"] + env_tags
        env_label = "/".join(env_tags) if env_tags else "待观察"

        return {
            "fund_code": self.fund_code,
            "backtest_period": f"{start_snap['date']} ~ {final['date']}",
            "trading_days": days,
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return_pct": total_return,
            "annual_return_pct": annual_return,
            "buy_hold_return_pct": buy_hold_return,
            "buy_hold_annual_pct": bh_annual,
            "excess_return_pct": round(total_return - buy_hold_return, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "total_trades": len(self.trades),
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "win_rate_pct": win_rate,
            "avg_win_amount": avg_win,
            "avg_lose_amount": avg_lose,
            "realized_pnl": round(self.realized_pnl, 2),
            "remaining_cash": round(self.cash, 2),
            "remaining_position": round(final["position_value"], 2),
            "signal_distribution": signal_dist,
            # v5.4 新增
            "avg_position_pct": avg_position_pct,
            "holding_day_ratio": holding_day_ratio,
            "empty_day_ratio": empty_day_ratio,
            "avg_hold_days": avg_hold_days,
            "median_hold_days": median_hold_days,
            "max_hold_days": max_hold_days,
            "median_profit": round(median_profit, 2),
            "p10_profit": round(p10_profit, 2),
            "p90_profit": round(p90_profit, 2),
            "max_single_win": round(max_single_win, 2),
            "max_single_loss": round(max_single_loss, 2),
            "max_consec_loss": max_consec_loss,
            "max_consec_win": max_consec_win,
            "max_dd_start": max_dd_start,
            "max_dd_end": max_dd_end,
            "max_dd_duration_days": max_dd_duration,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "monthly_win_rate": monthly_win_rate,
            "pos_months": pos_months,
            "neg_months": neg_months,
            "signal_pnl": signal_pnl,
            "monthly_returns": monthly_returns,
            "monthly_bh": monthly_bh,
            "quarterly_returns": quarterly_returns,
            "half_year_stats": half_year_stats,
            "bull_strat_return": round(bull_strat, 2),
            "bear_strat_return": round(bear_strat, 2),
            "neutral_strat_return": round(neutral_strat, 2),
            "bull_bh_return": round(bull_bh, 2),
            "bear_bh_return": round(bear_bh, 2),
            "neutral_bh_return": round(neutral_bh, 2),
            "bull_half_years": bull_count,
            "bear_half_years": bear_count,
            "neutral_half_years": neutral_count,
            # v5.4 策略适配度
            "fitness_score": fitness_score,
            "fitness_grade": fitness_grade,
            "score_efficiency": score_efficiency,
            "score_dd_control": score_dd_control,
            "score_capital": score_capital,
            "score_winrate": score_winrate,
            "score_vol_fit": score_vol_fit,
            "capture_rate": capture_rate,
            "env_label": env_label,
            "avg_daily_vol": avg_daily_vol,
            "max_daily_gain": max_daily_gain,
            "max_daily_loss": max_daily_loss,
            "mean_revert_ratio": mean_revert_ratio,
            "bh_max_drawdown_pct": round(bh_max_dd, 2),
            # v5.13: 行情模式统计
            "regime_mode": self.regime_mode,
            "regime_day_counts": self.regime_day_counts,
        }


# ============================================================
# 输出格式化（精简版：只保留策略迭代关键信息）
# ============================================================

def print_comparison(results: List[dict], fund_names: Dict[str, str]):
    """核心对比表"""
    # 按超额收益排序
    results_sorted = sorted(results, key=lambda s: s["excess_return_pct"], reverse=True)

    print(f"\n{'─'*100}")
    print(f" 基金                  | 策略收益 | 买持收益 | 超额收益 | 最大回撤 | 胜率  | 买/卖    | 核心信号")
    print(f"{'─'*100}")
    for s in results_sorted:
        name = fund_names.get(s["fund_code"], "")[:8]
        code = s["fund_code"]
        # 提取前2个最频繁信号
        sig = s.get("signal_distribution", {})
        top_sigs = sorted(sig.items(), key=lambda x: -x[1])[:2]
        sig_str = ", ".join(f"{n}({c})" for n, c in top_sigs)
        ex_mark = "✅" if s['excess_return_pct'] >= 0 else "  "
        print(f" {ex_mark}{name}({code}) | {s['total_return_pct']:>+7.2f}% | {s['buy_hold_return_pct']:>+7.2f}% | "
              f"{s['excess_return_pct']:>+7.2f}% | {s['max_drawdown_pct']:>6.2f}% | {s['win_rate_pct']:>4.1f}% | "
              f"{s['buy_count']:>2}/{s['sell_count']:<4} | {sig_str}")
    print(f"{'─'*100}")

    # 汇总行
    n = len(results)
    avg = lambda key: sum(s[key] for s in results) / n
    win_count = sum(1 for s in results if s["excess_return_pct"] > 0)
    print(f"  平均({n}只)            | {avg('total_return_pct'):>+7.2f}% | {avg('buy_hold_return_pct'):>+7.2f}% | "
          f"{avg('excess_return_pct'):>+7.2f}% | {avg('max_drawdown_pct'):>6.2f}% | {avg('win_rate_pct'):>4.1f}% | "
          f"跑赢基准: {win_count}/{n}")


def print_diagnosis(results: List[dict]):
    """策略问题诊断（仅输出可操作的迭代建议）"""
    n = len(results)
    avg_excess = sum(s["excess_return_pct"] for s in results) / n
    avg_dd = sum(s["max_drawdown_pct"] for s in results) / n
    avg_win = sum(s["win_rate_pct"] for s in results) / n
    avg_buy = sum(s["buy_count"] for s in results) / n
    avg_sell = sum(s["sell_count"] for s in results) / n

    # 汇总信号
    all_sigs = {}
    for s in results:
        for name, count in s.get("signal_distribution", {}).items():
            all_sigs[name] = all_sigs.get(name, 0) + count
    total_trades = sum(all_sigs.values())

    # 分类统计
    tp_sigs = sum(v for k, v in all_sigs.items() if "止盈" in k)
    sl_sigs = sum(v for k, v in all_sigs.items() if "止损" in k or "灾难" in k or "安全网" in k)
    buy_sigs = sum(v for k, v in all_sigs.items() if "建仓" in k or "抄底" in k or "低吸" in k or "补仓" in k)

    print(f"\n{'='*60}")
    print(f" 📋 策略诊断 (共{n}只基金, {total_trades}笔交易)")
    print(f"{'='*60}")
    print(f" 平均超额: {avg_excess:+.2f}% | 平均回撤: {avg_dd:.2f}% | 平均胜率: {avg_win:.1f}%")
    print(f" 平均每基金: 买{avg_buy:.1f}次 / 卖{avg_sell:.1f}次")
    print(f" 信号构成: 止盈{tp_sigs}次({tp_sigs*100//max(1,total_trades)}%) | "
          f"止损{sl_sigs}次({sl_sigs*100//max(1,total_trades)}%) | "
          f"买入{buy_sigs}次({buy_sigs*100//max(1,total_trades)}%)")

    print(f"\n 🔍 关键问题:")
    issues = []
    if avg_excess < -20:
        sell_ratio = avg_sell / max(1, avg_buy)
        issues.append(f"  1. 卖出过于频繁 (买卖比1:{sell_ratio:.1f})，大量小额止盈消耗了持仓时间")
        if tp_sigs > sl_sigs * 3:
            issues.append(f"     → 止盈{tp_sigs}次远多于止损{sl_sigs}次，策略在上涨中反复卖出再买回")
            issues.append(f"     → 建议: 提高止盈评分阈值(当前30分即触发)，或增加持仓最低天数限制")
    if avg_buy < 5:
        issues.append(f"  2. 买入机会少 (平均{avg_buy:.1f}次/基金)，多数时间空仓踏空")
        issues.append(f"     → 建议: 放宽大跌抄底阈值、增加温和回调建仓条件")
    if avg_dd < 5 and avg_excess < -30:
        issues.append(f"  3. 回撤极低({avg_dd:.2f}%)但收益也低，策略过于保守")
        issues.append(f"     → 建议: 适当放大单次建仓比例、延长持仓时间")
    if avg_win < 50:
        issues.append(f"  4. 卖出胜率偏低({avg_win:.1f}%)，止损频率高")
        issues.append(f"     → 建议: 收紧买入条件(避免抄底抄在半山腰)，或放宽止损线")

    if not issues:
        issues.append(f"  暂无明显异常，可尝试调整止盈/持仓参数进一步优化")

    for line in issues:
        print(line)

    # 信号TOP5
    print(f"\n 📊 信号频次 TOP5:")
    for name, count in sorted(all_sigs.items(), key=lambda x: -x[1])[:5]:
        pct = count * 100 // max(1, total_trades)
        bar = "█" * (pct // 2)
        print(f"  {name:<20} {count:>4}次 ({pct:>2}%) {bar}")


def print_trades(trades: List[dict], limit: int = 50):
    """打印交易明细"""
    print(f"\n{'─'*80}")
    print(f"  交易明细 (共{len(trades)}笔, 显示前{min(limit, len(trades))}笔)")
    print(f"{'─'*80}")
    print(f"  {'日期':<12} {'操作':<5} {'信号':<18} {'金额/份额':<12} {'净值':<8} {'盈亏':>8}  {'原因'}")
    print(f"  {'─'*78}")
    for t in trades[:limit]:
        if t["action"] == "buy":
            detail = f"{t['amount']:.0f}元"
            profit_str = ""
        else:
            detail = f"{t['shares']:.2f}份"
            profit_str = f"{t.get('profit', 0):+.2f}" if t.get('profit') is not None else ""
        reason = t.get("reason", "")[:30]
        print(f"  {t['date']:<12} {'买入' if t['action']=='buy' else '卖出':<5} "
              f"{t['signal_name']:<18} {detail:<12} {t['nav']:<8.4f} {profit_str:>8}  {reason}")


# ============================================================
# 基金列表加载
# ============================================================

# v5.4 优化: 10只覆盖牛熊的代表性基金
# 5只已验证长周期(2020~2026) + 5只2019年前成立的老基金
DEFAULT_FUNDS = [
    "007119",  # 睿远成长价值混合A    | 价值风格
    "001938",  # 中欧时代先锋股票A    | 成长风格
    "005827",  # 易方达蓝筹精选混合    | 蓝筹风格
    "110011",  # 易方达优质精选(QDII) | 全球配置
    "163406",  # 兴全合润混合A       | 均衡风格
    "161725",  # 招商中证白酒LOF     | 消费赛道,2015成立,经历白酒腰斩
    "000961",  # 天弘沪深300联接A    | 宽基指数,2014成立
    "519674",  # 银河创新成长混合      | 科技成长,2010成立
    "003745",  # 广发多因子混合       | 量化策略,2016成立
    "008701",  # 华夏黄金ETF联接A    | 黄金避险,与股市负相关
]


def load_funds_from_cache() -> List[str]:
    """从 data/cache/ 目录扫描所有 holdings_*.json 提取基金代码"""
    cache_dir = Path(__file__).parent / "data" / "cache"
    if not cache_dir.exists():
        return []
    codes = []
    for f in sorted(cache_dir.glob("holdings_*.json")):
        # holdings_017193.json -> 017193
        code = f.stem.replace("holdings_", "")
        if code and code not in codes:
            codes.append(code)
    return codes


def load_funds_from_state() -> List[str]:
    """从 data/state.json 读取所有自选基金代码"""
    state_file = Path(__file__).parent / "data" / "state.json"
    if not state_file.exists():
        return []
    try:
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        codes = []
        for sector in state.get("sectors", []):
            for fund in sector.get("funds", []):
                code = fund.get("code", "")
                if code and code not in codes:
                    codes.append(code)
        return codes
    except Exception:
        return []


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="低频网格策略历史回测")
    parser.add_argument("--funds", nargs="+", default=None,
                        help="手动指定基金代码列表")
    parser.add_argument("--days", type=int, default=1500,
                        help="回测天数 (默认1500，约6年覆盖完整牛熊)")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="每只基金初始资金 (默认10000)")
    parser.add_argument("--max-position", type=float, default=None,
                        help="每只基金最大仓位 (默认=初始资金)")
    parser.add_argument("--detail", action="store_true",
                        help="打印每笔交易明细")
    parser.add_argument("--all-history", action="store_true",
                        help="使用全部可用历史数据")
    parser.add_argument("--csv", type=str, default="backtest_result.csv",
                        help="CSV输出路径 (默认 backtest_result.csv)")
    parser.add_argument("--no-csv", action="store_true",
                        help="不输出CSV文件")
    parser.add_argument("--default", action="store_true",
                        help="强制使用内置默认基金列表（10只覆盖牛熊）")  # v5.4 优化
    parser.add_argument("--regime", type=str, default="auto",
                        choices=["bull", "neutral", "bear", "auto"],
                        help="行情模式: bull/neutral/bear/auto(默认,逐日动态切换)")
    args = parser.parse_args()

    # 基金列表: --default > --funds > cache目录 > state.json > 默认
    if args.default:  # v5.4 优化: --default 直接用内置10只，跳过state.json
        fund_codes = DEFAULT_FUNDS
        source = f"内置默认({len(DEFAULT_FUNDS)}只)"
    elif args.funds:
        fund_codes = args.funds
        source = "手动指定"
    else:
        fund_codes = load_funds_from_cache()
        if fund_codes:
            source = f"cache目录({len(fund_codes)}只)"
        else:
            fund_codes = load_funds_from_state()
            if fund_codes:
                source = f"state.json({len(fund_codes)}只)"
            else:
                fund_codes = DEFAULT_FUNDS
                source = "默认列表"

    max_position = args.max_position or args.capital
    regime = args.regime  # v5.13

    regime_label = {"bull": "🐂牛市", "neutral": "⚖️震荡", "bear": "🐻熊市", "auto": "🔄自动"}.get(regime, regime)
    print(f" 回测 v1.0 | {source} | 资金{args.capital:.0f}元 | "
          f"{'全历史' if args.all_history else f'近{args.days}天'} | {len(fund_codes)}只 | 模式:{regime_label}")
    print(f"{'─'*60}")

    all_results = []
    fund_names = {}
    skipped = []

    for i, code in enumerate(fund_codes):
        fund_name = fetch_fund_name(code)
        fund_names[code] = fund_name

        sys.stdout.write(f"\r [{i+1}/{len(fund_codes)}] {fund_name}({code})...")
        sys.stdout.flush()

        nav_data = fetch_nav_history(code)
        if not nav_data or len(nav_data) < 30:
            skipped.append(f"{fund_name}({code})")
            continue

        total_available = len(nav_data)  # v5.4: 记录API返回的全量天数

        if not args.all_history:
            nav_data = nav_data[-args.days:] if len(nav_data) > args.days else nav_data

        # v5.4: 显示实际使用的数据范围
        sys.stdout.write(f"\r [{i+1}/{len(fund_codes)}] {fund_name}({code}) "
                         f"API共{total_available}天, 使用{len(nav_data)}天 "
                         f"({nav_data[0]['date']}~{nav_data[-1]['date']})...")
        sys.stdout.flush()

        sim = BacktestSimulator(
            fund_code=code,
            nav_data=nav_data,
            initial_capital=args.capital,
            max_position=max_position,
            regime=regime,  # v5.13
        )
        result = sim.run(start_idx=min(20, len(nav_data) - 1))

        if "error" in result:
            skipped.append(f"{fund_name}({code})")
            continue

        all_results.append(result)

        if args.detail:
            sys.stdout.write(f"\r [{i+1}/{len(fund_codes)}] {fund_name}({code}) ✓\n")
            print_trades(sim.trades)

        if i < len(fund_codes) - 1:
            time.sleep(0.3)

    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()

    if skipped:
        print(f" ⚠ 跳过{len(skipped)}只(数据不足): {', '.join(skipped[:5])}{'...' if len(skipped)>5 else ''}")

    if len(all_results) >= 1:
        print_comparison(all_results, fund_names)

    if all_results:
        print_diagnosis(all_results)

    # === CSV 输出 ===
    if all_results and not args.no_csv:
        csv_path = Path(args.csv)
        export_csv(all_results, fund_names, csv_path)

    # === v5.13: 适配评分缓存写入（供前端展示）===
    if all_results:
        _write_fitness_cache(all_results, fund_names)


def _write_fitness_cache(results: List[dict], fund_names: Dict[str, str]):
    """v5.13: 将适配评分写入 data/fitness_cache.json 供前端读取"""
    import json as _json
    cache = {}
    for s in results:
        code = s.get("fund_code", "")
        if not code:
            continue
        cache[code] = {
            "score": s.get("fitness_score"),
            "grade": s.get("fitness_grade", ""),
            "env_label": s.get("env_label", ""),
            "name": fund_names.get(code, ""),
            "annual_return": s.get("annual_return_pct"),
            "sharpe": s.get("sharpe_ratio"),
            "max_drawdown": s.get("max_drawdown_pct"),
        }
    cache_path = Path(__file__).parent / "data" / "fitness_cache.json"
    cache_path.parent.mkdir(exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            _json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f" 📊 适配评分缓存已保存: {cache_path.resolve()} ({len(cache)}只基金)")
    except Exception as e:
        print(f" ⚠️ 适配评分缓存写入失败: {e}")


def export_csv(results: List[dict], fund_names: Dict[str, str], csv_path: Path):
    """v5.4: 输出完整策略诊断CSV，包含资金效率、牛熊分段、信号盈亏等核心指标"""
    import csv

    # 汇总信号名称
    all_sig_names = set()
    all_sig_pnl_names = set()
    for s in results:
        all_sig_names.update(s.get("signal_distribution", {}).keys())
        all_sig_pnl_names.update(s.get("signal_pnl", {}).keys())
    sig_names_sorted = sorted(all_sig_names)
    sig_pnl_sorted = sorted(all_sig_pnl_names)

    headers = [
        # --- 策略适配度（排序用，最重要）---
        "适配评分", "适配等级", "环境适配", "趋势捕获率%",
        "效率分", "回撤控分", "资金分", "胜率分", "波动分",
        # --- 基础信息 ---
        "基金代码", "基金名称", "回测区间", "交易天数",
        # --- 收益指标 ---
        "策略收益%", "年化收益%", "买入持有%", "买持年化%", "超额收益%",
        # --- 风险指标 ---
        "最大回撤%", "买持最大回撤%", "回撤起始", "回撤结束", "回撤持续天数",
        "夏普比率", "卡尔马比率",
        # --- 波动率特征 ---
        "日均波动%", "最大单日涨%", "最大单日跌%", "均值回归率%",
        # --- 交易统计 ---
        "买入次数", "卖出次数", "胜率%", "月度胜率%",
        "平均盈利", "平均亏损", "盈亏比",
        "中位盈亏", "P10盈亏", "P90盈亏",
        "最大单笔盈利", "最大单笔亏损",
        "最大连续亏损次数", "最大连续盈利次数",
        # --- 资金效率（核心诊断）---
        "平均仓位%", "持仓天数占比%", "空仓天数占比%",
        "平均持仓天数", "中位持仓天数", "最长持仓天数",
        # --- 盈亏结构 ---
        "已实现盈亏", "剩余现金", "持仓市值", "期末总值",
        # --- 牛熊分段收益（核心诊断）---
        "牛市半年数", "牛市策略收益%", "牛市买持收益%",
        "熊市半年数", "熊市策略收益%", "熊市买持收益%",
        "震荡半年数", "震荡策略收益%", "震荡买持收益%",
        # --- v5.13: 行情模式统计 ---
        "牛市天数", "震荡天数", "熊市天数",
    ]
    # 信号次数
    headers += [f"信号:{n}" for n in sig_names_sorted]
    # 信号盈亏分析
    for sn in sig_pnl_sorted:
        headers += [f"{sn}:次数", f"{sn}:胜率%", f"{sn}:总盈亏", f"{sn}:平均盈亏"]

    # 按适配评分降序排序（最适合的基金排最前面）
    results_sorted = sorted(results, key=lambda s: s.get("fitness_score", 0), reverse=True)

    rows = []
    for s in results_sorted:
        code = s["fund_code"]
        name = fund_names.get(code, "")
        sig = s.get("signal_distribution", {})
        sp = s.get("signal_pnl", {})
        pnl_ratio = round(abs(s["avg_win_amount"] / s["avg_lose_amount"]), 2) if s["avg_lose_amount"] != 0 else 999

        row = [
            # 适配度（放最前面）
            s.get("fitness_score", ""), s.get("fitness_grade", ""),
            s.get("env_label", ""), s.get("capture_rate", ""),
            s.get("score_efficiency", ""), s.get("score_dd_control", ""),
            s.get("score_capital", ""), s.get("score_winrate", ""),
            s.get("score_vol_fit", ""),
            # 基础信息
            code, name, s["backtest_period"], s["trading_days"],
            s["total_return_pct"], s["annual_return_pct"],
            s["buy_hold_return_pct"], s.get("buy_hold_annual_pct", ""),
            s["excess_return_pct"],
            s["max_drawdown_pct"], s.get("bh_max_drawdown_pct", ""),
            s.get("max_dd_start", ""), s.get("max_dd_end", ""),
            s.get("max_dd_duration_days", ""),
            s.get("sharpe_ratio", ""), s.get("calmar_ratio", ""),
            # 波动率特征
            s.get("avg_daily_vol", ""), s.get("max_daily_gain", ""),
            s.get("max_daily_loss", ""), s.get("mean_revert_ratio", ""),
            # 交易统计
            s["buy_count"], s["sell_count"], s["win_rate_pct"],
            s.get("monthly_win_rate", ""),
            s["avg_win_amount"], s["avg_lose_amount"], pnl_ratio,
            s.get("median_profit", ""), s.get("p10_profit", ""), s.get("p90_profit", ""),
            s.get("max_single_win", ""), s.get("max_single_loss", ""),
            s.get("max_consec_loss", ""), s.get("max_consec_win", ""),
            s.get("avg_position_pct", ""), s.get("holding_day_ratio", ""),
            s.get("empty_day_ratio", ""),
            s.get("avg_hold_days", ""), s.get("median_hold_days", ""),
            s.get("max_hold_days", ""),
            s["realized_pnl"], s["remaining_cash"],
            s["remaining_position"], s["final_value"],
            s.get("bull_half_years", ""), s.get("bull_strat_return", ""),
            s.get("bull_bh_return", ""),
            s.get("bear_half_years", ""), s.get("bear_strat_return", ""),
            s.get("bear_bh_return", ""),
            s.get("neutral_half_years", ""), s.get("neutral_strat_return", ""),
            s.get("neutral_bh_return", ""),
            # v5.13: 行情模式天数
            s.get("regime_day_counts", {}).get("bull", 0),
            s.get("regime_day_counts", {}).get("neutral", 0),
            s.get("regime_day_counts", {}).get("bear", 0),
        ]
        row += [sig.get(n, 0) for n in sig_names_sorted]
        for sn in sig_pnl_sorted:
            d = sp.get(sn, {})
            cnt = d.get("count", 0)
            wr = round(d["wins"] / cnt * 100, 1) if cnt > 0 else 0
            tp = round(d.get("total_profit", 0), 2)
            ap = round(tp / cnt, 2) if cnt > 0 else 0
            row += [cnt, wr, tp, ap]

        rows.append(row)

    # 添加汇总行
    n = len(results)
    def avg(key):
        vals = [s.get(key, 0) for s in results if s.get(key) is not None and s.get(key) != ""]
        return round(sum(vals) / max(1, len(vals)), 2) if vals else ""

    pnl_ratio_avg = round(abs(avg("avg_win_amount") / avg("avg_lose_amount")), 2) if avg("avg_lose_amount") else 999

    avg_row = [
        avg("fitness_score"), "", "", avg("capture_rate"),
        avg("score_efficiency"), avg("score_dd_control"),
        avg("score_capital"), avg("score_winrate"), avg("score_vol_fit"),
        "AVG", f"平均({n}只)", "", "",
        avg("total_return_pct"), avg("annual_return_pct"),
        avg("buy_hold_return_pct"), avg("buy_hold_annual_pct"),
        avg("excess_return_pct"),
        avg("max_drawdown_pct"), avg("bh_max_drawdown_pct"),
        "", "", avg("max_dd_duration_days"),
        avg("sharpe_ratio"), avg("calmar_ratio"),
        avg("avg_daily_vol"), avg("max_daily_gain"),
        avg("max_daily_loss"), avg("mean_revert_ratio"),
        round(sum(s["buy_count"] for s in results) / n, 1),
        round(sum(s["sell_count"] for s in results) / n, 1),
        avg("win_rate_pct"), avg("monthly_win_rate"),
        avg("avg_win_amount"), avg("avg_lose_amount"), pnl_ratio_avg,
        avg("median_profit"), avg("p10_profit"), avg("p90_profit"),
        avg("max_single_win"), avg("max_single_loss"),
        avg("max_consec_loss"), avg("max_consec_win"),
        avg("avg_position_pct"), avg("holding_day_ratio"), avg("empty_day_ratio"),
        avg("avg_hold_days"), avg("median_hold_days"), avg("max_hold_days"),
        avg("realized_pnl"), "", "", avg("final_value"),
        avg("bull_half_years"), avg("bull_strat_return"), avg("bull_bh_return"),
        avg("bear_half_years"), avg("bear_strat_return"), avg("bear_bh_return"),
        avg("neutral_half_years"), avg("neutral_strat_return"), avg("neutral_bh_return"),
    ]
    avg_row += [round(sum(s.get("signal_distribution", {}).get(sn, 0) for s in results) / n, 1)
                for sn in sig_names_sorted]
    for sn in sig_pnl_sorted:
        all_cnt = sum(s.get("signal_pnl", {}).get(sn, {}).get("count", 0) for s in results)
        all_wins = sum(s.get("signal_pnl", {}).get(sn, {}).get("wins", 0) for s in results)
        all_tp = sum(s.get("signal_pnl", {}).get(sn, {}).get("total_profit", 0) for s in results)
        avg_wr = round(all_wins / max(1, all_cnt) * 100, 1)
        avg_ap = round(all_tp / max(1, all_cnt), 2)
        avg_row += [round(all_cnt / n, 1), avg_wr, round(all_tp / n, 2), avg_ap]
    rows.append(avg_row)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"\n 📄 CSV已保存: {csv_path.resolve()}")
    print(f"    {len(results)}只基金 + 汇总行, {len(headers)}列")
    print(f"    含: 收益风险指标 | 资金效率 | 牛熊分段 | {len(sig_names_sorted)}种信号频次 | {len(sig_pnl_sorted)}种信号盈亏分析")


if __name__ == "__main__":
    main()