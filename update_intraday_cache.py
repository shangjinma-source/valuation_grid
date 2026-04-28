"""
update_intraday_cache.py - 更新盘中估值缓存并保存
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from valuation.core import calculate_valuation_by_state, flush_deviations

def main():
    print("开始更新盘中估值缓存...")
    
    # 计算所有基金的估值（这会更新 intraday_cache.json）
    result = calculate_valuation_by_state()
    
    # 刷新偏差缓冲区
    flush_deviations()
    
    # 输出结果摘要
    count = len(result.get("items", []))
    print(f"已更新 {count} 只基金的估值")
    
    # 显示部分结果
    items = result.get("items", [])[:5]
    for item in items:
        code = item.get("code", "N/A")
        est = item.get("est_change", "N/A")
        print(f"  {code}: {est}%")
    
    if count > 5:
        print(f"  ... 还有 {count - 5} 只基金")
    
    print("缓存更新完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())
