"""
patch_pending_rebuys.py - 一次性补录 017560__自选 的两笔历史延迟回补挂单

背景：改造前实盘的延迟回补只有前端提示，没持久化。本脚本把当时
2026-05-06 和 2026-05-08 两次止盈遗留的挂单按"方案B"补回 positions.json：
两条独立记录、相同触发净值 2.8608、金额各自记真值。

用法：
  python patch_pending_rebuys.py            # dry-run（默认，只打印不写盘）
  python patch_pending_rebuys.py --apply    # 实际写盘
"""
import json
import sys
import shutil
from pathlib import Path
from datetime import datetime

# ============================================================
# 待补的两条挂单（方案 B：两笔同 trigger，各自独立）
# ============================================================
TARGET_FUND_KEY = "017560__自选"

PATCHES = [
    {
        "id": "rb20260506a",
        "created_date": "2026-05-06",
        "expire_date": "2026-06-03",   # 5-6 + 28 自然日
        "trigger_nav": 2.8608,
        "amount": 11216.54,
        "ratio": 0.70,                  # 强趋势/连涨档常见值；不影响触发
        "trend_at_creation": "震荡",
        "source_signal": "改造前历史回补",
        "signal_label": "延迟回补(改造前历史回补)",
        "sell_nav": 2.9278,             # 真实收盘净值
        "status": "pending",
    },
    {
        "id": "rb20260508a",
        "created_date": "2026-05-08",
        "expire_date": "2026-06-05",   # 5-8 + 28 自然日
        "trigger_nav": 2.8608,
        "amount": 1348.98,
        "ratio": 0.40,                  # 震荡档常见值；不影响触发
        "trend_at_creation": "震荡",
        "source_signal": "改造前历史回补",
        "signal_label": "延迟回补(改造前历史回补)",
        "sell_nav": 2.9044,             # 真实收盘净值
        "status": "pending",
    },
]

# ============================================================
# 主流程
# ============================================================
def main(apply: bool):
    pos_path = Path(__file__).parent / "data" / "positions.json"
    if not pos_path.exists():
        print(f"[ERROR] 找不到 {pos_path}")
        print(f"        请把脚本放到 valuation_grid 根目录下（与 data/ 平级）运行")
        sys.exit(1)

    # 读取
    with open(pos_path, encoding="utf-8") as f:
        data = json.load(f)

    funds = data.get("funds", {})
    if TARGET_FUND_KEY not in funds:
        print(f"[ERROR] {TARGET_FUND_KEY} 不在 positions.json 的 funds 里")
        print(f"        现有 keys: {list(funds.keys())}")
        sys.exit(1)

    fund = funds[TARGET_FUND_KEY]
    pending = fund.setdefault("pending_rebuys", [])

    # 重复检查（按 id 去重，已存在就跳过）
    existing_ids = {p.get("id") for p in pending}
    to_add = [p for p in PATCHES if p["id"] not in existing_ids]
    skipped = [p for p in PATCHES if p["id"] in existing_ids]

    print("=" * 60)
    print(f"目标基金: {TARGET_FUND_KEY}")
    print(f"现有 pending_rebuys 条数: {len(pending)}")
    print(f"待新增: {len(to_add)} 条")
    print(f"已存在跳过: {len(skipped)} 条")
    print("=" * 60)

    for p in to_add:
        print(f"\n[NEW] id={p['id']}  created={p['created_date']}  expire={p['expire_date']}")
        print(f"      sell_nav={p['sell_nav']}  trigger_nav={p['trigger_nav']}  amount={p['amount']}")
        print(f"      signal_label={p['signal_label']}  status={p['status']}")
    for p in skipped:
        print(f"\n[SKIP] id={p['id']} 已存在，不重复添加")

    if not to_add:
        print("\n没有需要新增的记录，退出。")
        return

    if not apply:
        print("\n" + "=" * 60)
        print("DRY-RUN 模式：以上变更未写入文件。")
        print("确认无误后用 `python patch_pending_rebuys.py --apply` 实际写盘。")
        print("=" * 60)
        return

    # 备份原文件
    backup_path = pos_path.with_suffix(
        f".json.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    shutil.copy2(pos_path, backup_path)
    print(f"\n[BACKUP] 已备份原文件 → {backup_path.name}")

    # 写入
    pending.extend(to_add)
    with open(pos_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[WRITE] 已写入 {len(to_add)} 条到 {pos_path.name}")
    print(f"\n现在可以 curl http://localhost:8000/v1/pending-rebuys/summary 验证")


if __name__ == "__main__":
    apply = "--apply" in sys.argv
    main(apply)