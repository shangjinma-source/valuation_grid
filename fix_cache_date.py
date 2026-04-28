"""
fix_cache_date.py - 修正 intraday_cache.json 中的日期为今天
"""
import json
from datetime import datetime
from pathlib import Path

# 获取今天的日期
today = datetime.now().strftime("%Y-%m-%d")
cache_file = Path(__file__).parent / "data" / "intraday_cache.json"

# 读取缓存文件
with open(cache_file, "r", encoding="utf-8") as f:
    cache = json.load(f)

# 更新所有条目的日期为今天
updated = False
for code, data in cache.items():
    if data["date"] != today:
        print(f"Updating {code}: {data['date']} -> {today}")
        data["date"] = today
        updated = True

# 如果有更新，保存文件
if updated:
    tmp_file = cache_file.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    tmp_file.replace(cache_file)
    print(f"\nCache file updated to {today}")
else:
    print(f"\nCache file already has today's date ({today})")

# 验证
with open(cache_file, "r", encoding="utf-8") as f:
    cache = json.load(f)

sample_funds = list(cache.keys())[:5]
print("Sample cache entries after fix:")
for code in sample_funds:
    entry = cache[code]
    print(f"  {code}: {entry}")
