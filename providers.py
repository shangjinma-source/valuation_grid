"""
providers.py - 持仓抓取 + 行情拉取（含缓存/超时/批量）
支持：普通基金、ETF联接基金穿透、用户截图导入
"""
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from typing import Dict, List, Optional, Tuple

# === 配置常量 ===
CACHE_DIR = Path(__file__).parent / "cache"
HOLDINGS_CACHE_DAYS = 30
QUOTES_TTL_SECONDS = 8
REQUEST_TIMEOUT = 15

# === 内存缓存 ===
_quotes_cache: Dict[str, dict] = {}
_week_change_cache: Dict[str, dict] = {}  # 基金本周涨幅缓存

# === ETF联接基金：动态检测 + 用户手动指定 ===

# 用户手动指定的ETF映射（运行时可通过API添加）
_user_etf_map: Dict[str, str] = {}

def set_etf_link_target(link_code: str, etf_code: str):
    """用户手动设置联接基金的目标ETF"""
    _user_etf_map[link_code] = etf_code
    print(f"[Holdings] 用户设置ETF映射: {link_code} -> {etf_code}")

def get_etf_link_target(link_code: str) -> Optional[str]:
    """获取用户设置的目标ETF"""
    return _user_etf_map.get(link_code)

def clear_etf_link_target(link_code: str):
    """清除用户设置的ETF映射"""
    if link_code in _user_etf_map:
        del _user_etf_map[link_code]

def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 基金名称获取
# ============================================================

def get_fund_name(fund_code: str) -> Optional[str]:
    """从天天基金获取基金名称"""
    try:
        url = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"http://fund.eastmoney.com/{fund_code}.html",
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        match = re.search(r'var\s+fS_name\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    except:
        pass
    return None


def _is_etf_link_fund(fund_name: Optional[str]) -> bool:
    """根据基金名称判断是否为ETF联接基金"""
    if not fund_name:
        return False
    return "ETF联接" in fund_name or "ETF发起式联接" in fund_name


# ============================================================
# 持仓 Provider
# ============================================================

def _get_holdings_cache_path(fund_code: str) -> Path:
    return CACHE_DIR / f"holdings_{fund_code}.json"

def _is_holdings_cache_valid(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        return datetime.now() - mtime < timedelta(days=HOLDINGS_CACHE_DAYS)
    except:
        return False


def _fetch_holdings_combined(fund_code: str, depth: int = 0) -> Optional[dict]:
    """
    组合数据源获取持仓，支持ETF联接基金穿透（最多2层）
    """
    if depth > 2:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"http://fund.eastmoney.com/{fund_code}.html",
    }

    # === 第一步：从 pingzhongdata 获取股票总仓位和基金名称 ===
    stock_position_ratio = 95.0
    fund_name = None
    try:
        url1 = f"http://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
        req1 = Request(url1, headers=headers)
        with urlopen(req1, timeout=REQUEST_TIMEOUT) as resp:
            content1 = resp.read().decode("utf-8")

        name_match = re.search(r'var\s+fS_name\s*=\s*"([^"]+)"', content1)
        if name_match:
            fund_name = name_match.group(1)

        asset_match = re.search(r'var\s+Data_assetAllocation\s*=\s*(\{.*?\});', content1, re.DOTALL)
        if asset_match:
            stock_data = re.search(r'"name":"股票占净比"[^]]*"data":\[([^\]]+)\]', asset_match.group(1))
            if stock_data:
                values = [float(x) for x in stock_data.group(1).split(',') if x.strip()]
                if values:
                    stock_position_ratio = values[-1]
    except Exception as e:
        print(f"[Holdings] 获取仓位比例失败: {e}")

    # === 检测是否需要使用用户手动设置的ETF映射 ===
    user_etf_target = get_etf_link_target(fund_code)
    if user_etf_target and depth == 0:
        print(f"[Holdings] 使用用户指定的目标ETF: {fund_code} -> {user_etf_target}")
        etf_holdings = _fetch_holdings_combined(user_etf_target, depth + 1)
        if etf_holdings and etf_holdings.get("positions"):
            etf_holdings["original_fund_code"] = fund_code
            etf_holdings["fund_code"] = fund_code
            etf_holdings["fund_name"] = fund_name
            etf_holdings["etf_target"] = user_etf_target
            etf_holdings["is_etf_link"] = True
            return etf_holdings

    # === 第二步：从 FundArchivesDatas 获取详细持仓 ===
    try:
        url2 = f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?type=jjcc&code={fund_code}&topline=100&year=&month=&rt={int(time.time()*1000)}"
        req2 = Request(url2, headers=headers)
        with urlopen(req2, timeout=REQUEST_TIMEOUT) as resp:
            content2 = resp.read().decode("utf-8")

        # 解析报告日期
        report_date = None
        quarter_match = re.search(r'(\d{4})年(\d)季度', content2)
        if quarter_match:
            year = quarter_match.group(1)
            quarter = int(quarter_match.group(2))
            quarter_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            report_date = f"{year}-{quarter_end.get(quarter, '12-31')}"

        first_table = re.search(r'<table[^>]*class="w782 comm tzxq"[^>]*>(.*?)</table>', content2, re.DOTALL)
        if not first_table:
            first_table = re.search(r'<table[^>]*>(.*?)</table>', content2, re.DOTALL)

        if not first_table:
            return None

        table_content = first_table.group(1)
        holdings = []
        parsed_weight = 0.0
        seen_codes = set()

        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL)
        for row in rows:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
            if len(cells) < 8:
                continue

            code_match = re.search(r'>(\d{6})<', cells[1])
            if not code_match:
                code_match = re.search(r'(\d{6})', cells[1])
            if not code_match:
                continue
            stock_code = code_match.group(1)

            if stock_code in seen_codes:
                continue
            seen_codes.add(stock_code)

            name_match = re.search(r'>([^<]+)<', cells[2])
            stock_name = name_match.group(1).strip() if name_match else stock_code

            weight_text = re.sub(r'<[^>]+>', '', cells[6]).strip()
            weight_text = weight_text.replace('%', '').replace('％', '')
            try:
                weight = float(weight_text)
            except:
                continue

            if weight <= 0:
                continue

            holdings.append({
                "stock_code": stock_code,
                "stock_name": stock_name,
                "weight": weight
            })
            parsed_weight += weight

        if not holdings:
            return None

        holdings.sort(key=lambda x: x["weight"], reverse=True)
        missing_weight = max(0, stock_position_ratio - parsed_weight)

        return {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "holdings_asof_date": report_date,
            "stock_total_weight": round(stock_position_ratio, 2),
            "parsed_weight": round(parsed_weight, 2),
            "missing_weight": round(missing_weight, 2),
            "positions": holdings,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        print(f"[Holdings] 抓取失败 {fund_code}: {e}")
        return None


def get_holdings(fund_code: str, force_refresh: bool = False) -> dict:
    """获取基金持仓快照"""
    _ensure_cache_dir()
    cache_path = _get_holdings_cache_path(fund_code)

    if not force_refresh and _is_holdings_cache_valid(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("positions"):
                    # ETF联接穿透后的数据也是有效的
                    if data.get("is_etf_link") or data.get("parsed_weight", 0) > 30:
                        return data
        except:
            pass

    data = _fetch_holdings_combined(fund_code)

    if data and data.get("positions"):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Holdings] 缓存写入失败: {e}")
        return data

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_stale"] = True
                return data
        except:
            pass

    return {
        "fund_code": fund_code,
        "error": "无法获取持仓数据",
        "holdings_asof_date": None,
        "stock_total_weight": 0,
        "positions": []
    }


# ============================================================
# 行情 Provider
# ============================================================

def _convert_to_sina_symbol(stock_code: str) -> str:
    if stock_code.startswith(("6", "5", "9")):
        return f"sh{stock_code}"
    return f"sz{stock_code}"


def _fetch_quotes_batch_sina(tickers: List[str]) -> Dict[str, dict]:
    if not tickers:
        return {}

    all_results = {}
    batch_size = 50

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        symbols = [_convert_to_sina_symbol(t) for t in batch]
        url = f"https://hq.sinajs.cn/list={','.join(symbols)}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        }

        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                content = resp.read().decode("gbk")

            now = datetime.now()
            for line in content.strip().split("\n"):
                match = re.match(r'var hq_str_(s[hz]\d{6})="([^"]*)"', line)
                if not match:
                    continue
                symbol, data_str = match.groups()
                if not data_str:
                    continue
                data = data_str.split(",")
                if len(data) < 4:
                    continue

                stock_code = symbol[2:]
                try:
                    current = float(data[3]) if data[3] else 0
                    yesterday = float(data[2]) if data[2] else 0
                    pct = round((current - yesterday) / yesterday * 100, 2) if yesterday > 0 and current > 0 else 0.0
                    all_results[stock_code] = {
                        "stock_code": stock_code,
                        "name": data[0],
                        "pct_change": pct,
                        "asof_time": now.strftime("%Y-%m-%d %H:%M:%S")
                    }
                except:
                    continue
        except Exception as e:
            print(f"[Quotes] 请求失败: {e}")

    return all_results


def get_quotes(tickers: List[str]) -> dict:
    if not tickers:
        return {"quotes": {}, "missing": []}

    tickers = list(set(tickers))
    now = time.time()
    results = {}
    need_fetch = []

    for t in tickers:
        if t in _quotes_cache and now - _quotes_cache[t]["ts"] < QUOTES_TTL_SECONDS:
            results[t] = _quotes_cache[t]["data"]
        else:
            need_fetch.append(t)

    if need_fetch:
        fetched = _fetch_quotes_batch_sina(need_fetch)
        for t in need_fetch:
            if t in fetched:
                results[t] = fetched[t]
                _quotes_cache[t] = {"data": fetched[t], "ts": now}

    return {"quotes": results, "missing": [t for t in tickers if t not in results]}


if __name__ == "__main__":
    import sys
    fund_code = sys.argv[1] if len(sys.argv) > 1 else "016708"

    print(f"=== 测试: {fund_code} ===")
    h = get_holdings(fund_code, force_refresh=True)

    print(f"基金名称: {h.get('fund_name')}")
    print(f"持仓日期: {h.get('holdings_asof_date')}")
    print(f"股票总仓位: {h.get('stock_total_weight')}%")
    print(f"已解析权重: {h.get('parsed_weight')}%")
    print(f"持仓数量: {len(h.get('positions', []))}")

    if h.get("is_etf_link"):
        print(f"ETF联接穿透: {h.get('etf_target')}")

    if h.get("positions"):
        print("\n前10大:")
        for i, p in enumerate(h["positions"][:10], 1):
            print(f"  {i}. {p['stock_code']} {p['stock_name']}: {p['weight']:.2f}%")


# ============================================================
# 基金近5日涨幅获取
# ============================================================

def _fetch_fund_performance_batch(fund_codes: List[str]) -> Dict[str, Optional[float]]:
    """
    使用东方财富基金业绩API批量获取近1周涨幅
    接口返回字段: 近1周涨幅
    """
    result = {}
    if not fund_codes:
        return result

    try:
        # 东方财富基金业绩排行API，可以按基金代码查询
        # fields: f12=代码, f14=名称, f3=近1周涨幅
        codes_str = ",".join(fund_codes)
        url = (
            f"https://push2.eastmoney.com/api/qt/clist/get?"
            f"fid=f3&po=1&pz={len(fund_codes)}&np=1&fltt=2&invt=2"
            f"&fields=f12,f14,f3"
            f"&fs=b:{codes_str}"
        )

        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://fund.eastmoney.com/",
        })
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        data = json.loads(content)
        if data.get("data") and data["data"].get("diff"):
            for item in data["data"]["diff"]:
                code = item.get("f12", "")
                week_change = item.get("f3")
                if code and week_change is not None and week_change != "-":
                    try:
                        result[code] = float(week_change)
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        print(f"[5DayChange] 批量API失败: {e}")

    return result


def _fetch_fund_week_change_single(fund_code: str) -> Optional[float]:
    """
    单个基金近1周涨幅 - 使用天天基金净值API
    获取最近6个交易日净值，计算5日涨幅
    """
    try:
        url = (
            f"https://api.fund.eastmoney.com/f10/lsjz?"
            f"fundCode={fund_code}&pageIndex=1&pageSize=6"
        )
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://fundf10.eastmoney.com/jjjz_{fund_code}.html",
        })
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        data = json.loads(content)
        items = data.get("Data", {}).get("LSJZList", [])

        if len(items) >= 6:
            # items[0] = 最新, items[5] = 5个交易日前
            newest = float(items[0]["DWJZ"])
            oldest = float(items[5]["DWJZ"])
            if oldest > 0:
                change = (newest - oldest) / oldest * 100
                return round(change, 2)
    except Exception as e:
        print(f"[5DayChange] 净值API失败 {fund_code}: {e}")

    return None


def get_fund_5day_change(fund_code: str) -> Optional[float]:
    """获取基金近5日涨幅（带缓存）"""
    cache_key = fund_code
    if cache_key in _week_change_cache:
        cached = _week_change_cache[cache_key]
        if time.time() - cached["time"] < 3600:
            return cached["value"]

    val = _fetch_fund_week_change_single(fund_code)
    if val is not None:
        _week_change_cache[cache_key] = {"value": val, "time": time.time()}
    return val


def get_fund_5day_changes_batch(fund_codes: List[str]) -> Dict[str, Optional[float]]:
    """批量获取基金近5日涨幅"""
    result = {}
    for code in fund_codes:
        result[code] = get_fund_5day_change(code)
    return result


# ============================================================
# 持仓缓存定期自动刷新
# ============================================================

# 缓存剩余天数 <= 此值时视为"即将过期"，提前刷新
_REFRESH_AHEAD_DAYS = 3
# 每只基金刷新间隔（秒），避免请求过密
_REFRESH_INTERVAL_SEC = 2

def _get_tracked_fund_codes() -> List[str]:
    """从 state.json 读取用户跟踪的全部基金代码"""
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


def _holdings_cache_remaining_days(fund_code: str) -> Optional[float]:
    """返回缓存剩余有效天数，文件不存在返回 None"""
    cache_path = _get_holdings_cache_path(fund_code)
    if not cache_path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        elapsed = datetime.now() - mtime
        return HOLDINGS_CACHE_DAYS - elapsed.total_seconds() / 86400
    except Exception:
        return None


def refresh_stale_holdings() -> dict:
    """
    扫描用户跟踪的全部基金，刷新已过期或即将过期的持仓缓存。
    返回 {"refreshed": [...], "failed": [...], "skipped": int}
    """
    _ensure_cache_dir()
    fund_codes = _get_tracked_fund_codes()

    refreshed = []
    failed = []
    skipped = 0

    for code in fund_codes:
        remaining = _holdings_cache_remaining_days(code)

        # 缓存仍然充裕，跳过
        if remaining is not None and remaining > _REFRESH_AHEAD_DAYS:
            skipped += 1
            continue

        label = "expired" if remaining is None or remaining <= 0 else f"{remaining:.1f}d left"
        print(f"[AutoRefresh] {code} ({label}) -> refreshing...")

        try:
            data = _fetch_holdings_combined(code)
            if data and data.get("positions"):
                cache_path = _get_holdings_cache_path(code)
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                refreshed.append(code)
                print(f"[AutoRefresh] {code} OK, {len(data['positions'])} positions")
            else:
                failed.append(code)
                print(f"[AutoRefresh] {code} FAILED: no positions returned")
        except Exception as e:
            failed.append(code)
            print(f"[AutoRefresh] {code} ERROR: {e}")

        # 请求间隔
        if code != fund_codes[-1]:
            time.sleep(_REFRESH_INTERVAL_SEC)

    summary = {"refreshed": refreshed, "failed": failed, "skipped": skipped}
    print(f"[AutoRefresh] Done: {len(refreshed)} refreshed, {len(failed)} failed, {skipped} skipped")
    return summary