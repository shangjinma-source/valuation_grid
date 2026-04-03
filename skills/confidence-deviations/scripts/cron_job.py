"""
cron_job.py - 每日收盘后任务(entry point for Windows Task Scheduler)
功能：
1. 加载 state.json 中的全部基金
2. 执行 calculate_valuation_batch() 计算所有基金估值
3. 记录估值 vs 净值偏差到 confidence_deviations.json
4. 执行策略信号计算（如果有需要）

调用方式：
python cron_job.py

"""
import json
from pathlib import Path
import sys

# 添加项目根目录到 Python path
PROJECT_ROOT = Path(r"E:\Git\valuation_grid")
sys.path.insert(0, str(PROJECT_ROOT))

from valuation.core import calculate_valuation_batch, load_state


def run_daily_batch():
    """执行每日收盘后批量估值计算"""
    # 1. 加载基金列表
    state = load_state()
    
    # 收集所有基金代码
    all_codes = []
    for sector in state.get("sectors", []):
        for fund in sector.get("funds", []):
            code = fund.get("code", "")
            if code and code not in all_codes:
                all_codes.append(code)
    
    if not all_codes:
        print("No funds found in state.json")
        return
    
    print(f"Found {len(all_codes)} funds, computing valuations...")
    
    # 2. 批量估值（自动记录偏差到 confidence_deviations.json）
    results = calculate_valuation_batch(all_codes)
    
    # 3. 统计结果
    success_count = 0
    error_count = 0
    for r in results:
        if "error" in r:
            error_count += 1
        else:
            success_count += 1
    
    print(f"Batch completed: {success_count} success, {error_count} errors")
    
    # 4. 检查 confidence_deviations.json 是否生成
    data_dir = PROJECT_ROOT / "data"
    dev_file = data_dir / "confidence_deviations.json"
    
    if dev_file.exists():
        try:
            with open(dev_file, "r", encoding="utf-8") as f:
                devs = json.load(f)
            print(f"confidence_deviations.json updated: {len(devs)} funds with history")
        except Exception as e:
            print(f"Warning: failed to load confidence_deviations.json: {e}")
    else:
        print("Warning: confidence_deviations.json not found after batch run")
    
    return results


if __name__ == "__main__":
    print("=== valuation_grid Daily Batch (22:00) ===")
    run_daily_batch()
    print("=== Done ===")
