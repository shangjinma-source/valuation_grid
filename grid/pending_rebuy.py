"""
grid/pending_rebuy.py - 延迟回补挂单生命周期管理

止盈卖出后，不立即买入，而是记录"挂单"：
  - 记录触发净值（trigger_nav = 卖出价 × (1 - rebuy_discount)）
  - 记录回补金额（amount）
  - 记录窗口期（默认 28 自然日，约 20 交易日，与回测对齐）

每次生成信号时，检查是否有挂单触发：
  - 当前净值 ≤ trigger_nav 且未过期 → 触发，生成 buy 信号
  - 已过期但未触发 → 标记为 expired，保留历史
  - 已触发 → 标记为 triggered，保留历史，关联到具体 batch

设计目标：和 backtest.py 中 self.pending_rebuys[] 的行为完全对齐。
"""

import threading
from datetime import datetime, timedelta
from typing import Optional, List

from positions import load_positions, save_positions, parse_fund_key

# ============================================================
# 常量
# ============================================================

# 挂单有效窗口（自然日）。回测里是 day_idx + 20（20 交易日），
# 实盘改用自然日近似：20 交易日 ≈ 28 自然日（含周末缓冲）
PENDING_REBUY_WINDOW_DAYS = 28

# 同一基金的最大并发挂单数（实盘软上限，防止 bug 导致挂单堆积）
# 回测无此限制；实际策略中由于 cooldown 机制，正常情况下也不会堆积过多
PENDING_REBUY_MAX_CONCURRENT = 3

_pending_lock = threading.Lock()


# ============================================================
# 内部工具
# ============================================================

def _next_pending_id(existing: list, today_str: str) -> str:
    """生成挂单 ID: rb + 日期(去-) + 序号字母"""
    date_part = today_str.replace("-", "")
    same_day = [p["id"] for p in existing if p.get("id", "").startswith(f"rb{date_part}")]
    idx = len(same_day)
    letter = chr(ord("a") + min(idx, 25))
    return f"rb{date_part}{letter}"


def _is_active(pr: dict, today_str: str) -> bool:
    """是否还在有效期内（未过期、未触发、未取消）"""
    if pr.get("status") != "pending":
        return False
    expire_date = pr.get("expire_date", "")
    if not expire_date:
        return True
    return today_str <= expire_date


# ============================================================
# 核心 API（供 engine.py 调用）
# ============================================================

def add_pending_rebuy(fund_code: str, trigger_nav: float, amount: float,
                      ratio: float, trend_at_creation: str,
                      source_signal: str, sell_nav: float) -> Optional[str]:
    """
    创建一条延迟回补挂单。
    返回挂单 ID（成功）或 None（被软上限拒绝）。

    与 backtest.py 中 self.pending_rebuys.append({...}) 完全对齐：
    - trigger_nav: 触发净值
    - amount: 回补金额
    - 窗口: today + PENDING_REBUY_WINDOW_DAYS 自然日
    """
    with _pending_lock:
        data = load_positions()
        fund = data.get("funds", {}).get(fund_code)
        if not fund:
            print(f"[PendingRebuy] {fund_code} 不存在，跳过创建")
            return None

        pending = fund.setdefault("pending_rebuys", [])
        today = datetime.now().date()
        today_str = today.strftime("%Y-%m-%d")

        # 软上限检查：超过最大并发数则拒绝
        active = [p for p in pending if _is_active(p, today_str)]
        if len(active) >= PENDING_REBUY_MAX_CONCURRENT:
            print(f"[PendingRebuy] {fund_code} 已有 {len(active)} 个活跃挂单，"
                  f"达到上限 {PENDING_REBUY_MAX_CONCURRENT}，拒绝新建")
            return None

        expire_date = (today + timedelta(days=PENDING_REBUY_WINDOW_DAYS)).strftime("%Y-%m-%d")
        pr_id = _next_pending_id(pending, today_str)

        # 标签：把 source_signal 包成回测用的格式 "延迟回补(xxx)"
        # 如果 source_signal 已经是该格式，直接用
        if source_signal.startswith("延迟回补"):
            signal_label = source_signal
        else:
            signal_label = f"延迟回补({source_signal})"

        pr = {
            "id": pr_id,
            "created_date": today_str,
            "expire_date": expire_date,
            "trigger_nav": round(trigger_nav, 4),
            "amount": round(amount, 2),
            "ratio": round(ratio, 4),
            "trend_at_creation": trend_at_creation,
            "source_signal": source_signal,
            "signal_label": signal_label,
            "sell_nav": round(sell_nav, 4),
            "status": "pending",
        }
        pending.append(pr)
        save_positions(data)
        print(f"[PendingRebuy] 创建 {fund_code} {pr_id}: "
              f"trigger={pr['trigger_nav']} amount={pr['amount']} "
              f"expire={expire_date}")
        return pr_id


def check_pending_rebuy_trigger(fund_code: str, current_nav: float) -> Optional[dict]:
    """
    检查 fund_code 是否有挂单触发。返回触发的挂单字典（取最早创建的活跃挂单中第一个触发的）；
    若过期则原地标记 expired；若无触发则返回 None。

    与 backtest.py 主循环开头的检查逻辑完全对齐：
    - 过期 → continue（标记 expired）
    - current_nav <= trigger_nav → 触发
    - 否则保留

    注意：本函数仅"检查并标记过期"，不执行"消费"。
    实际触发后由 engine.py 生成 buy 信号，用户确认买入时由 add_batch 调用
    consume_pending_rebuy 标记为 triggered。
    """
    if current_nav is None or current_nav <= 0:
        return None

    with _pending_lock:
        data = load_positions()
        fund = data.get("funds", {}).get(fund_code)
        if not fund:
            return None
        pending = fund.get("pending_rebuys", [])
        if not pending:
            return None

        today_str = datetime.now().strftime("%Y-%m-%d")
        triggered = None
        changed = False

        for pr in pending:
            if pr.get("status") != "pending":
                continue
            # 过期检查
            if today_str > pr.get("expire_date", "9999-99-99"):
                pr["status"] = "expired"
                pr["expired_at"] = today_str
                changed = True
                print(f"[PendingRebuy] {fund_code} {pr['id']} 已过期")
                continue
            # 触发检查
            if current_nav <= pr["trigger_nav"]:
                triggered = pr  # 找到第一个触发的就停（按创建顺序）
                break

        if changed:
            save_positions(data)

        # 返回触发的副本（避免外部修改 pending 列表中的对象）
        return dict(triggered) if triggered else None


def consume_pending_rebuy(fund_code: str, pending_id: str, batch_id: str) -> bool:
    """
    将挂单标记为 triggered（已被一次买入消费）。
    通常由 positions.add_batch 在创建 batch 后自动调用。
    """
    with _pending_lock:
        data = load_positions()
        fund = data.get("funds", {}).get(fund_code)
        if not fund:
            return False
        pending = fund.get("pending_rebuys", [])
        for pr in pending:
            if pr.get("id") == pending_id and pr.get("status") == "pending":
                pr["status"] = "triggered"
                pr["triggered_at"] = datetime.now().strftime("%Y-%m-%d")
                pr["triggered_batch_id"] = batch_id
                save_positions(data)
                print(f"[PendingRebuy] {fund_code} {pending_id} → triggered "
                      f"(batch={batch_id})")
                return True
        return False


def cleanup_expired_pending_rebuys(fund_code: str = None) -> int:
    """
    清理已过期但仍标记为 pending 的挂单（兜底，正常情况下 check 时会自动标记）。
    fund_code=None 表示全量扫描。返回标记的条数。
    """
    with _pending_lock:
        data = load_positions()
        funds = data.get("funds", {})
        today_str = datetime.now().strftime("%Y-%m-%d")
        count = 0
        codes = [fund_code] if fund_code else list(funds.keys())
        for code in codes:
            fund = funds.get(code)
            if not fund:
                continue
            for pr in fund.get("pending_rebuys", []):
                if pr.get("status") == "pending" and today_str > pr.get("expire_date", ""):
                    pr["status"] = "expired"
                    pr["expired_at"] = today_str
                    count += 1
        if count > 0:
            save_positions(data)
            print(f"[PendingRebuy] 清理过期挂单 {count} 条")
        return count


def get_pending_rebuys(fund_code: str, include_history: bool = False) -> List[dict]:
    """
    查询某基金的挂单列表（供前端展示）。
    include_history=False（默认）只返回 pending 状态；True 返回全部。
    """
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return []
    all_pending = fund.get("pending_rebuys", [])
    if include_history:
        return list(all_pending)
    today_str = datetime.now().strftime("%Y-%m-%d")
    return [p for p in all_pending if _is_active(p, today_str)]


def get_all_pending_rebuys_summary() -> dict:
    """
    全量挂单概览（供监控/调试）。
    返回 {fund_code: {pending: N, triggered: N, expired: N}}
    """
    data = load_positions()
    funds = data.get("funds", {})
    summary = {}
    for code, fund in funds.items():
        prs = fund.get("pending_rebuys", [])
        if not prs:
            continue
        summary[code] = {
            "pending": sum(1 for p in prs if p.get("status") == "pending"),
            "triggered": sum(1 for p in prs if p.get("status") == "triggered"),
            "expired": sum(1 for p in prs if p.get("status") == "expired"),
            "total": len(prs),
        }
    return summary