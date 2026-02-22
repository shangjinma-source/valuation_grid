"""
positions.py - 持仓记录管理（data/positions.json）
"""
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
POS_FILE = DATA_DIR / "positions.json"
_pos_lock = threading.Lock()

# ============================================================
# 复合键支持（同一基金多人持仓）
# 格式: "fund_code__owner"，如 "017193__老公"
# 不含 "__" 的为默认实例（向后兼容）
# ============================================================

def parse_fund_key(fund_key: str) -> tuple:
    """解析复合键，返回 (real_fund_code, owner)"""
    if "__" in fund_key:
        parts = fund_key.split("__", 1)
        return parts[0], parts[1]
    return fund_key, ""

def make_fund_key(fund_code: str, owner: str = "") -> str:
    """构造复合键"""
    if owner:
        return f"{fund_code}__{owner}"
    return fund_code

# ============================================================
# 费率表（持有天数 < threshold 时收取该费率%）
# threshold=None 表示兜底（≥所有前面阈值）
# 硬编码为兜底，实际优先使用 positions.json 中每只基金的 fee_schedule
# ============================================================

FEE_SCHEDULES_FALLBACK = {
    "015968": [(7, 1.50), (30, 0.50), (None, 0.00)],
    "025500": [(7, 1.50), (30, 0.50), (None, 0.00)],
    "017193": [(7, 1.50), (None, 0.00)],
}
DEFAULT_FEE_SCHEDULE = [(7, 1.50), (30, 0.50), (None, 0.00)]


def get_sell_fee_rate(fund_key: str, hold_days: int) -> float:
    """根据基金键名和持有天数查费率表，返回卖出费率%
    优先从 positions.json 的 fee_schedule 读取，否则用硬编码兜底"""
    real_code, _ = parse_fund_key(fund_key)
    # 优先从持仓配置读取（用完整 fund_key）
    data = load_positions()
    fund = data.get("funds", {}).get(fund_key)
    if fund and fund.get("fee_schedule"):
        schedule = [(s["days"], s["rate"]) for s in fund["fee_schedule"]]
    else:
        schedule = FEE_SCHEDULES_FALLBACK.get(real_code, DEFAULT_FEE_SCHEDULE)

    for threshold, rate in schedule:
        if threshold is None or hold_days < threshold:
            return rate
    return 0.0


def update_fee_schedule(fund_code: str, fee_schedule: list) -> bool:
    """更新基金的费率表。fee_schedule: [{days: int|null, rate: float}, ...]"""
    data = load_positions()
    funds = data.setdefault("funds", {})
    if fund_code not in funds:
        funds[fund_code] = {
            "fund_name": "",
            "max_position": 5000,
            "batches": [],
            "supplement_count": 0,
            "cooldown_until": None,
        }
    funds[fund_code]["fee_schedule"] = fee_schedule
    save_positions(data)
    print(f"[Position] 更新费率 {fund_code}: {fee_schedule}")
    return True


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _empty_positions() -> dict:
    return {
        "funds": {},
        "cash_reserve_ratio": 0.30,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ============================================================
# 持仓读写（文件锁保护）
# ============================================================

def load_positions() -> dict:
    """加载持仓，文件不存在或损坏时返回空结构"""
    _ensure_data_dir()
    if not POS_FILE.exists():
        return _empty_positions()
    try:
        with open(POS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 基本校验
        if not isinstance(data, dict) or "funds" not in data:
            print("[Position] positions.json 结构异常，返回空数据")
            return _empty_positions()
        return data
    except Exception as e:
        print(f"[Position] 读取 positions.json 失败: {e}")
        return _empty_positions()


def save_positions(data: dict) -> bool:
    """原子写入（tmp+rename）"""
    _ensure_data_dir()
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _pos_lock:
        try:
            tmp_file = POS_FILE.with_suffix(".tmp")
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            tmp_file.replace(POS_FILE)
            return True
        except Exception as e:
            print(f"[Position] 保存 positions.json 失败: {e}")
            return False


# ============================================================
# 批次操作
# ============================================================

def _next_batch_id(batches: list, date_str: str) -> str:
    """生成批次ID: b + 日期(去掉-) + 序号字母"""
    date_part = date_str.replace("-", "")
    existing = [b["id"] for b in batches if b["id"].startswith(f"b{date_part}")]
    idx = len(existing)  # 0->a, 1->b, ...
    letter = chr(ord("a") + min(idx, 25))
    return f"b{date_part}{letter}"


def add_batch(fund_code: str, amount: float, nav: float, note: str = "",
              buy_date: str = None, is_supplement: bool = False) -> dict:
    """
    新增一笔买入批次。
    - buy_date 格式 YYYY-MM-DD，默认今天
    - is_supplement: 是否为补仓买入（自动递增 supplement_count）
    - 自动创建 fund entry（如果不存在）
    - 返回新增的 batch dict
    """
    data = load_positions()
    funds = data.setdefault("funds", {})

    if fund_code not in funds:
        funds[fund_code] = {
            "fund_name": "",
            "max_position": 5000,
            "batches": [],
            "supplement_count": 0,
            "cooldown_until": None,
        }

    fund = funds[fund_code]

    # 买入日期：支持手动指定（录入历史持仓），默认今天
    if buy_date and _is_valid_date(buy_date):
        date_str = buy_date
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")

    shares = round(amount / nav, 2) if nav > 0 else 0.0

    # 自动识别补仓：显式标记 或 note 以"补仓"开头
    _is_supp = is_supplement or (note and note.startswith("补仓"))

    batch = {
        "id": _next_batch_id(fund["batches"], date_str),
        "buy_date": date_str,
        "amount": round(amount, 2),
        "nav": round(nav, 4),
        "shares": shares,
        "status": "holding",
        "note": note,
        "is_supplement": _is_supp,
    }
    fund["batches"].append(batch)

    # 补仓计数递增
    if _is_supp:
        fund["supplement_count"] = fund.get("supplement_count", 0) + 1
        print(f"[Position] 补仓计数 {fund_code}: {fund['supplement_count']}")

    save_positions(data)
    print(f"[Position] 新增批次 {fund_code} {batch['id']}: {amount}元 @ {nav} ({date_str})"
          f"{' [补仓]' if _is_supp else ''}")
    return batch


def _is_valid_date(s: str) -> bool:
    """校验日期格式 YYYY-MM-DD"""
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except:
        return False


def sell_batch(fund_code: str, batch_id: str, sell_shares: float,
               sell_nav: float = None, sell_date: str = None) -> dict:
    """
    卖出某批次（按份额）。
    - sell_shares: 卖出份额（必填，和支付宝一致）
    - sell_nav: 卖出确认净值（选填，确认后补录）
    - sell_date: 卖出日期（选填，默认今天）
    返回 { hold_days, sell_fee_rate, sell_shares, fee, net, profit, profit_pct }

    无论全部卖出还是部分卖出，都会生成一条 sell_records 记录，
    对应支付宝的一笔卖出交易，方便对账。
    """
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        raise ValueError(f"基金 {fund_code} 不存在")

    batch = None
    for b in fund["batches"]:
        if b["id"] == batch_id:
            batch = b
            break
    if not batch:
        raise ValueError(f"批次 {batch_id} 不存在")
    if batch["status"] != "holding":
        raise ValueError(f"批次 {batch_id} 状态为 {batch['status']}，无法卖出")

    if sell_shares <= 0:
        raise ValueError("卖出份额必须大于0")
    if sell_shares > batch["shares"] + 0.01:  # 容差
        raise ValueError(f"卖出份额 {sell_shares} 超过持有份额 {batch['shares']}")

    # 卖出日期
    if sell_date and _is_valid_date(sell_date):
        sd = datetime.strptime(sell_date, "%Y-%m-%d").date()
    else:
        sd = datetime.now().date()

    buy_date = datetime.strptime(batch["buy_date"], "%Y-%m-%d").date()
    hold_days = (sd - buy_date).days

    sell_fee_rate = get_sell_fee_rate(fund_code, hold_days)

    # 计算收益（sell_nav 可选）
    sell_cost_ratio = sell_shares / batch["shares"] if batch["shares"] > 0 else 1.0
    cost = round(batch["amount"] * sell_cost_ratio, 2)

    if sell_nav and sell_nav > 0:
        gross = round(sell_shares * sell_nav, 2)
        fee = round(gross * sell_fee_rate / 100, 2)
        net = round(gross - fee, 2)
        profit = round(net - cost, 2)
        profit_pct = round(profit / cost * 100, 1) if cost > 0 else 0.0
    else:
        # 净值未确认，仅扣减份额，收益待确认
        gross = None
        fee = None
        net = None
        profit = None
        profit_pct = None

    # === 生成卖出记录（无论全部/部分卖出都生成） ===
    sell_records = fund.setdefault("sell_records", [])
    sell_record_id = f"s{sd.strftime('%Y%m%d')}{chr(ord('a') + len([r for r in sell_records if r.get('sell_date') == sd.strftime('%Y-%m-%d')]))}"
    sell_record = {
        "id": sell_record_id,
        "batch_id": batch_id,
        "sell_date": sd.strftime("%Y-%m-%d"),
        "sell_shares": round(sell_shares, 2),
        "buy_nav": round(batch["nav"], 4),
        "sell_nav": round(sell_nav, 4) if sell_nav and sell_nav > 0 else None,
        "cost": cost,
        "gross": gross,
        "fee": fee,
        "net": net,
        "profit": profit,
        "profit_pct": profit_pct,
        "hold_days": hold_days,
        "sell_fee_rate": sell_fee_rate,
        "note": batch.get("note", ""),
    }
    sell_records.append(sell_record)

    # === 更新批次 ===
    is_full_sell = abs(sell_shares - batch["shares"]) < 0.01
    if is_full_sell:
        batch["status"] = "sold"
        batch["sell_date"] = sd.strftime("%Y-%m-%d")
        if sell_nav:
            batch["sell_nav"] = round(sell_nav, 4)
        batch["sell_shares"] = round(sell_shares, 2)
    else:
        # 部分卖出：保存原始金额（首次部分卖出时记录），扣减份额和对应成本
        if "original_amount" not in batch:
            batch["original_amount"] = batch["amount"]
            batch["original_shares"] = batch["shares"]
        batch["shares"] = round(batch["shares"] - sell_shares, 2)
        batch["amount"] = round(batch["amount"] * (1 - sell_cost_ratio), 2)

    # 设置冷却期（自然日兜底 + 交易日精确）
    cooldown_date = sd + timedelta(days=COOLDOWN_DAYS_NATURAL)
    fund["cooldown_until"] = cooldown_date.strftime("%Y-%m-%d")
    # v4.1: 交易日冷却（由 strategy 在信号生成时用 nav_history 日期精确推进）
    fund["cooldown_sell_date"] = sd.strftime("%Y-%m-%d")
    fund["cooldown_trade_days"] = COOLDOWN_TRADE_DAYS

    # 如果全部批次已清仓，重置补仓计数
    holding_batches = [b for b in fund["batches"] if b["status"] == "holding"]
    if not holding_batches:
        fund["supplement_count"] = 0
        print(f"[Position] {fund_code} 全部清仓，重置补仓计数")

    save_positions(data)
    print(f"[Position] 卖出 {fund_code} {batch_id}: {sell_shares}份 @ {sell_nav or '待确认'}")

    return {
        "hold_days": hold_days,
        "sell_fee_rate": sell_fee_rate,
        "sell_shares": round(sell_shares, 2),
        "fee": fee,
        "net": net,
        "profit": profit,
        "profit_pct": profit_pct,
        "nav_pending": sell_nav is None or sell_nav <= 0,
        "sell_record_id": sell_record_id,
    }


# 冷却自然日（简化：用自然日近似交易日）
COOLDOWN_DAYS_NATURAL = 4  # 2个交易日 ≈ 4个自然日（含周末缓冲）
COOLDOWN_TRADE_DAYS = 2    # v4.1: 精确交易日冷却天数


def update_sell_nav(fund_code: str, sell_record_id: str, sell_nav: float) -> dict:
    """补录卖出确认净值，重新计算收益"""
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        raise ValueError(f"基金 {fund_code} 不存在")

    sell_records = fund.get("sell_records", [])
    record = None
    for r in sell_records:
        if r["id"] == sell_record_id:
            record = r
            break
    if not record:
        raise ValueError(f"卖出记录 {sell_record_id} 不存在")

    record["sell_nav"] = round(sell_nav, 4)
    gross = round(record["sell_shares"] * sell_nav, 2)
    fee = round(gross * record["sell_fee_rate"] / 100, 2)
    net = round(gross - fee, 2)
    profit = round(net - record["cost"], 2)
    profit_pct = round(profit / record["cost"] * 100, 1) if record["cost"] > 0 else 0.0

    record["gross"] = gross
    record["fee"] = fee
    record["net"] = net
    record["profit"] = profit
    record["profit_pct"] = profit_pct

    # 同步更新对应 batch 的 sell_nav（如果是全卖）
    for b in fund["batches"]:
        if b["id"] == record["batch_id"] and b.get("status") == "sold":
            b["sell_nav"] = round(sell_nav, 4)

    save_positions(data)
    print(f"[Position] 补录净值 {fund_code} {sell_record_id}: {sell_nav}")
    return record


def delete_sell_record(fund_code: str, sell_record_id: str) -> bool:
    """删除卖出记录，并将份额/金额还原回对应的买入批次"""
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return False

    sell_records = fund.get("sell_records", [])
    record = None
    for r in sell_records:
        if r["id"] == sell_record_id:
            record = r
            break
    if not record:
        return False

    # 找到对应的买入批次并还原
    batch_id = record["batch_id"]
    for b in fund["batches"]:
        if b["id"] == batch_id:
            if b.get("status") == "sold":
                # 全部卖出 → 恢复为 holding
                b["status"] = "holding"
                # 恢复份额：用 original_shares 或 sell_shares
                if "original_shares" in b:
                    b["shares"] = b["original_shares"]
                    b["amount"] = b["original_amount"]
                    del b["original_shares"]
                    del b["original_amount"]
                else:
                    b["shares"] = round(record["sell_shares"], 2)
                    b["amount"] = round(record["cost"], 2)
                # 清理卖出字段
                b.pop("sell_date", None)
                b.pop("sell_nav", None)
                b.pop("sell_shares", None)
            else:
                # 部分卖出 → 加回份额和对应成本
                b["shares"] = round(b["shares"] + record["sell_shares"], 2)
                b["amount"] = round(b["amount"] + record["cost"], 2)
                # 如果恢复后份额等于原始份额，清理 original_ 字段
                if "original_shares" in b and abs(b["shares"] - b["original_shares"]) < 0.01:
                    b["shares"] = b["original_shares"]
                    b["amount"] = b["original_amount"]
                    del b["original_shares"]
                    del b["original_amount"]
            break

    # 删除卖出记录
    fund["sell_records"] = [r for r in sell_records if r["id"] != sell_record_id]

    save_positions(data)
    print(f"[Position] 删除卖出记录并还原批次 {fund_code} {sell_record_id}")
    return True


def delete_batch(fund_code: str, batch_id: str) -> bool:
    """删除买入批次，同时删除该批次关联的所有卖出记录"""
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        return False

    original_len = len(fund["batches"])
    fund["batches"] = [b for b in fund["batches"] if b["id"] != batch_id]
    if len(fund["batches"]) == original_len:
        return False

    # 级联删除关联的卖出记录
    sell_records = fund.get("sell_records", [])
    removed_sells = [r["id"] for r in sell_records if r["batch_id"] == batch_id]
    if removed_sells:
        fund["sell_records"] = [r for r in sell_records if r["batch_id"] != batch_id]
        print(f"[Position] 级联删除卖出记录: {removed_sells}")

    save_positions(data)
    print(f"[Position] 删除批次 {fund_code} {batch_id}")
    return True


def remove_fund(fund_code: str) -> bool:
    """从持仓管理中完全移除一只基金（删除所有记录）"""
    data = load_positions()
    funds = data.get("funds", {})
    if fund_code not in funds:
        return False
    del funds[fund_code]
    save_positions(data)
    print(f"[Position] 移除基金 {fund_code}")
    return True


def rename_fund_key(old_key: str, new_key: str) -> bool:
    """
    重命名持仓键（用于给已有持仓追加/修改标签）。
    例如 "017193" → "017193__老公"，或 "017193__老公" → "017193__老婆"
    数据整体迁移，不影响批次和卖出记录。
    """
    if old_key == new_key:
        return True
    data = load_positions()
    funds = data.get("funds", {})
    if old_key not in funds:
        return False
    if new_key in funds:
        return False  # 目标键已存在，不能覆盖
    funds[new_key] = funds.pop(old_key)
    # 同步更新 groups 中的 fund_codes 引用
    for grp in data.get("groups", []):
        grp["fund_codes"] = [new_key if c == old_key else c for c in grp.get("fund_codes", [])]
    save_positions(data)
    print(f"[Position] 重命名 {old_key} → {new_key}")
    return True


def update_fund_config(fund_code: str, max_position: int = None, fund_name: str = None) -> bool:
    """更新单只基金的配置"""
    data = load_positions()
    funds = data.setdefault("funds", {})

    if fund_code not in funds:
        funds[fund_code] = {
            "fund_name": "",
            "max_position": 5000,
            "batches": [],
            "supplement_count": 0,
            "cooldown_until": None,
        }

    fund = funds[fund_code]
    if max_position is not None:
        fund["max_position"] = max_position
    if fund_name is not None:
        fund["fund_name"] = fund_name

    save_positions(data)
    print(f"[Position] 更新配置 {fund_code}: max={fund.get('max_position')}, name={fund.get('fund_name')}")
    return True


def sell_fifo(fund_code: str, total_sell_shares: float,
              sell_nav: float = None, sell_date: str = None) -> dict:
    """
    按 FIFO（先进先出）顺序卖出指定总份额。
    模拟支付宝实际行为：用户只输入总份额，系统从最早批次开始依次扣减。

    返回 {
        total_sell_shares, batch_details: [{batch_id, sell_shares, hold_days, fee_rate, ...}],
        total_fee, total_net, total_profit
    }
    """
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)
    if not fund:
        raise ValueError(f"基金 {fund_code} 不存在")

    holding = [b for b in fund.get("batches", []) if b["status"] == "holding"]
    holding_sorted = sorted(holding, key=lambda b: b["buy_date"])

    total_available = sum(b["shares"] for b in holding_sorted)
    if total_sell_shares > total_available + 0.01:
        raise ValueError(f"卖出份额 {total_sell_shares} 超过持有总份额 {total_available:.2f}")

    if sell_date and _is_valid_date(sell_date):
        sd = datetime.strptime(sell_date, "%Y-%m-%d").date()
    else:
        sd = datetime.now().date()

    remaining = total_sell_shares
    batch_details = []

    for batch in holding_sorted:
        if remaining <= 0.005:
            break
        shares_to_sell = min(remaining, batch["shares"])
        # 容差处理：差距极小时当作全部卖出
        if abs(shares_to_sell - batch["shares"]) < 0.01:
            shares_to_sell = batch["shares"]

        buy_date = datetime.strptime(batch["buy_date"], "%Y-%m-%d").date()
        hold_days = (sd - buy_date).days
        fee_rate = get_sell_fee_rate(fund_code, hold_days)

        sell_cost_ratio = shares_to_sell / batch["shares"] if batch["shares"] > 0 else 1.0
        cost = round(batch["amount"] * sell_cost_ratio, 2)

        if sell_nav and sell_nav > 0:
            gross = round(shares_to_sell * sell_nav, 2)
            fee = round(gross * fee_rate / 100, 2)
            net = round(gross - fee, 2)
            profit = round(net - cost, 2)
            profit_pct = round(profit / cost * 100, 1) if cost > 0 else 0.0
        else:
            gross = fee = net = profit = profit_pct = None

        batch_details.append({
            "batch_id": batch["id"],
            "buy_date": batch["buy_date"],
            "sell_shares": round(shares_to_sell, 2),
            "hold_days": hold_days,
            "sell_fee_rate": fee_rate,
            "cost": cost,
            "gross": gross,
            "fee": fee,
            "net": net,
            "profit": profit,
            "profit_pct": profit_pct,
            "is_full_sell": abs(shares_to_sell - batch["shares"]) < 0.01,
        })

        # === 生成卖出记录 ===
        sell_records = fund.setdefault("sell_records", [])
        sell_record_id = f"s{sd.strftime('%Y%m%d')}{chr(ord('a') + len([r for r in sell_records if r.get('sell_date') == sd.strftime('%Y-%m-%d')]))}"
        sell_record = {
            "id": sell_record_id,
            "batch_id": batch["id"],
            "sell_date": sd.strftime("%Y-%m-%d"),
            "sell_shares": round(shares_to_sell, 2),
            "buy_nav": round(batch["nav"], 4),
            "sell_nav": round(sell_nav, 4) if sell_nav and sell_nav > 0 else None,
            "cost": cost,
            "gross": gross,
            "fee": fee,
            "net": net,
            "profit": profit,
            "profit_pct": profit_pct,
            "hold_days": hold_days,
            "sell_fee_rate": fee_rate,
            "note": batch.get("note", ""),
        }
        sell_records.append(sell_record)
        batch_details[-1]["sell_record_id"] = sell_record_id

        # === 更新批次 ===
        is_full = abs(shares_to_sell - batch["shares"]) < 0.01
        if is_full:
            batch["status"] = "sold"
            batch["sell_date"] = sd.strftime("%Y-%m-%d")
            if sell_nav:
                batch["sell_nav"] = round(sell_nav, 4)
            batch["sell_shares"] = round(shares_to_sell, 2)
        else:
            if "original_amount" not in batch:
                batch["original_amount"] = batch["amount"]
                batch["original_shares"] = batch["shares"]
            batch["shares"] = round(batch["shares"] - shares_to_sell, 2)
            batch["amount"] = round(batch["amount"] * (1 - sell_cost_ratio), 2)

        remaining -= shares_to_sell

    # 设置冷却期（自然日兜底 + 交易日精确）
    cooldown_date = sd + timedelta(days=COOLDOWN_DAYS_NATURAL)
    fund["cooldown_until"] = cooldown_date.strftime("%Y-%m-%d")
    # v4.1: 交易日冷却
    fund["cooldown_sell_date"] = sd.strftime("%Y-%m-%d")
    fund["cooldown_trade_days"] = COOLDOWN_TRADE_DAYS

    # 全部清仓则重置补仓计数
    holding_after = [b for b in fund["batches"] if b["status"] == "holding"]
    if not holding_after:
        fund["supplement_count"] = 0
        print(f"[Position] {fund_code} 全部清仓，重置补仓计数")

    save_positions(data)

    total_fee = sum(d["fee"] for d in batch_details if d["fee"] is not None)
    total_net = sum(d["net"] for d in batch_details if d["net"] is not None)
    total_profit = sum(d["profit"] for d in batch_details if d["profit"] is not None)

    print(f"[Position] FIFO卖出 {fund_code}: 总份额{total_sell_shares}, 涉及{len(batch_details)}个批次")

    return {
        "total_sell_shares": round(total_sell_shares, 2),
        "batch_count": len(batch_details),
        "batch_details": batch_details,
        "total_fee": round(total_fee, 2) if any(d["fee"] is not None for d in batch_details) else None,
        "total_net": round(total_net, 2) if any(d["net"] is not None for d in batch_details) else None,
        "total_profit": round(total_profit, 2) if any(d["profit"] is not None for d in batch_details) else None,
        "nav_pending": sell_nav is None or sell_nav <= 0,
    }


def get_fund_position(fund_code: str) -> dict:
    """
    返回某基金的持仓汇总（供策略模块调用）。
    仅统计 status=holding 的批次。
    """
    data = load_positions()
    fund = data.get("funds", {}).get(fund_code)

    if not fund:
        return {
            "fund_code": fund_code,
            "has_position": False,
            "total_amount": 0,
            "total_shares": 0,
            "batches": [],
            "oldest_hold_days": 0,
            "newest_hold_days": 0,
            "max_position": 5000,
            "supplement_count": 0,
            "in_cooldown": False,
        }

    today = datetime.now().date()
    holding_batches = [b for b in fund.get("batches", []) if b["status"] == "holding"]

    total_amount = round(sum(b["amount"] for b in holding_batches), 2)
    total_shares = round(sum(b["shares"] for b in holding_batches), 2)

    hold_days_list = []
    for b in holding_batches:
        bd = datetime.strptime(b["buy_date"], "%Y-%m-%d").date()
        hold_days_list.append((today - bd).days)

    # 冷却期判断
    in_cooldown = False
    cd = fund.get("cooldown_until")
    if cd:
        try:
            cd_date = datetime.strptime(cd, "%Y-%m-%d").date()
            in_cooldown = today <= cd_date
        except:
            pass

    # === 已实现盈亏（从 sell_records 汇总）===
    sell_records = fund.get("sell_records", [])
    realized_pnl = round(sum(r.get("profit", 0) or 0 for r in sell_records), 2)
    total_invested = round(
        sum(b.get("original_amount", b["amount"]) for b in fund.get("batches", [])),
        2
    )
    total_received = round(sum(r.get("net", 0) or 0 for r in sell_records), 2)

    return {
        "fund_code": fund_code,
        "has_position": len(holding_batches) > 0,
        "total_amount": total_amount,       # 向后兼容：当前持仓成本
        "total_cost": total_amount,          # 语义明确：当前持仓成本
        "total_shares": total_shares,
        "batches": holding_batches,
        "oldest_hold_days": max(hold_days_list) if hold_days_list else 0,
        "newest_hold_days": min(hold_days_list) if hold_days_list else 0,
        "max_position": fund.get("max_position", 5000),
        "supplement_count": fund.get("supplement_count", 0),
        "in_cooldown": in_cooldown,
        "cooldown_until": cd,
        "cooldown_sell_date": fund.get("cooldown_sell_date"),
        "cooldown_trade_days": fund.get("cooldown_trade_days", 2),
        "fee_schedule": fund.get("fee_schedule"),
        # v5.2 新增：盈亏追踪字段
        "realized_pnl": realized_pnl,           # 已实现盈亏（从卖出记录汇总）
        "total_invested": total_invested,        # 历史总投入（含已卖出批次）
        "total_received": total_received,        # 已回款总额
        "sell_records_count": len(sell_records),
    }


def get_all_positions() -> dict:
    """返回全部持仓数据"""
    return load_positions()


# ============================================================
# 分组管理（groups 存储在 positions.json 顶层）
# ============================================================

def get_groups() -> list:
    """获取所有分组"""
    data = load_positions()
    return data.get("groups", [])


def save_groups(groups: list) -> bool:
    """保存分组列表"""
    data = load_positions()
    data["groups"] = groups
    return save_positions(data)


def add_group(name: str) -> dict:
    """新增分组，返回新分组对象"""
    data = load_positions()
    groups = data.setdefault("groups", [])
    gid = f"g{datetime.now().strftime('%Y%m%d%H%M%S')}"
    group = {"id": gid, "name": name, "fund_codes": []}
    groups.append(group)
    save_positions(data)
    return group


def update_group(group_id: str, name: str = None, fund_codes: list = None) -> bool:
    """更新分组名称或基金列表"""
    data = load_positions()
    groups = data.get("groups", [])
    for g in groups:
        if g["id"] == group_id:
            if name is not None:
                g["name"] = name
            if fund_codes is not None:
                g["fund_codes"] = fund_codes
            save_positions(data)
            return True
    return False


def delete_group(group_id: str) -> bool:
    """删除分组"""
    data = load_positions()
    groups = data.get("groups", [])
    new_groups = [g for g in groups if g["id"] != group_id]
    if len(new_groups) == len(groups):
        return False
    data["groups"] = new_groups
    save_positions(data)
    return True