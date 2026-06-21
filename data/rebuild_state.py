r"""
rebuild_state.py
从 C:\Users\Administrator\Desktop\基金热门板块.csv 重建 data/state.json
"""
import csv
import sys
import time
import json
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from valuation.core import save_state
from valuation.providers import get_fund_name


def parse_csv(path: str) -> list:
    """解析CSV，返回 [{name, funds: [{code}]}}]"""
    sectors = []
    with open(path, "r", encoding="gbk") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        print("[ERROR] CSV为空")
        return []

    # 第一行：板块名称
    sector_names = [c.strip() for c in rows[0] if c.strip()]
    print(f"[CSV] 板块数: {len(sector_names)}")

    # 初始化 sectors
    sectors = [{"name": name, "funds": []} for name in sector_names]

    # 后续行：提取每列基金代码
    for row_idx, row in enumerate(rows[1:], start=2):
        for col_idx in range(min(len(row), len(sector_names))):
            code = row[col_idx].strip()
            if code:
                # 过滤非数字字符（清理可能的空白/异常）
                code_clean = code.strip()
                if code_clean.isdigit():
                    sectors[col_idx]["funds"].append({"code": code_clean})

    return sectors


def main():
    csv_path = r"C:\Users\Administrator\Desktop\基金热门板块.csv"

    print("=" * 60)
    print("步骤1: 解析CSV")
    print("=" * 60)
    sectors = parse_csv(csv_path)

    total_funds = sum(len(s["funds"]) for s in sectors)
    print(f"板块数: {len(sectors)}, 基金总数: {total_funds}")
    for s in sectors:
        print(f"  {s['name']}: {len(s['funds'])}只")

    print()
    print("=" * 60)
    print("步骤2: 批量获取基金名称")
    print("=" * 60)

    failed_codes = []
    total = total_funds
    processed = 0

    for si, sector in enumerate(sectors):
        for fi, fund in enumerate(sector["funds"]):
            code = fund["code"]
            processed += 1
            print(f"[{processed}/{total}] {code} ... ", end="", flush=True)

            try:
                name = get_fund_name(code)
                if name:
                    fund["alias"] = name
                    print(f"[OK] {name}")
                else:
                    fund["alias"] = ""
                    failed_codes.append(code)
                    print("[FAIL] 获取失败")
            except Exception as e:
                fund["alias"] = ""
                failed_codes.append(code)
                print(f"[FAIL] 异常: {e}")

            # 每隔 0.5s 请求一次，避免限流
            if processed < total:
                time.sleep(0.5)

    print()
    print("=" * 60)
    print("步骤3: 写入 state.json")
    print("=" * 60)

    from datetime import datetime
    state = {
        "version": 1,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sectors": sectors,
    }

    ok = save_state(state)
    if ok:
        print("✓ 写入成功: data/state.json")
    else:
        print("✗ 写入失败")
        return

    print()
    print("=" * 60)
    print("步骤4: 摘要")
    print("=" * 60)

    print(f"板块数: {len(sectors)}")
    print(f"基金总数: {total_funds}")
    for s in sectors:
        print(f"  {s['name']}: {len(s['funds'])}只")
    print(f"名称获取失败: {len(failed_codes)}只")
    if failed_codes:
        for fc in failed_codes:
            print(f"  {fc}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
