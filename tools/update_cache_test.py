import datetime
from valuation.core import calculate_valuation_batch, load_state, _intraday_estimation_cache, _INTRADAY_CACHE_FILE

print(f'Today: {datetime.datetime.now().strftime("%Y-%m-%d")}')
state = load_state()
fund_codes = [f['code'] for s in state.get('sectors', []) for f in s.get('funds', [])]
print(f'Found {len(fund_codes)} funds')

results = calculate_valuation_batch(fund_codes)
print(f'Calculated {len(results)} valuation records')

print('First 3 results:')
for r in results[:3]:
    print(f'{r["fund_code"]}: est={r.get("estimation_change")}, source={r.get("_source")}, notes={r.get("notes", [])[:2]}')

print(f'Cache loaded: {len(_intraday_estimation_cache)}')
print(f'Cache file exists: {_INTRADAY_CACHE_FILE.exists()}')
print(f'Cache file modified: {_INTRADAY_CACHE_FILE.stat().st_mtime}')

# 检查缓存中的日期
sample_funds = list(_intraday_estimation_cache.keys())[:5]
print('Sample cache entries:')
for code in sample_funds:
    entry = _intraday_estimation_cache[code]
    print(f'  {code}: {entry}')
