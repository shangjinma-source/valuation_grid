"""
providers.py - 持仓抓取 + 行情拉取（含缓存/超时/批量）
支持：普通基金、ETF联接基金穿透
数据源：天天基金季报持仓（自动定期刷新）
"""
import json
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request, install_opener, build_opener, ProxyHandler
from urllib.error import URLError, HTTPError
from typing import Dict, List, Optional, Tuple

# 强制所有 urlopen 请求不走系统代理，避免梯子干扰
install_opener(build_opener(ProxyHandler({})))

# === 配置常量 ===
CACHE_DIR = Path(__file__).parent.parent / "cache"
QUOTES_TTL_SECONDS = 8
REQUEST_TIMEOUT = 15
HOLDINGS_CACHE_DAYS = 30  # 持仓缓存天数（唯一控制持仓缓存有效期的常量）

# === 内存缓存 ===
_quotes_cache: Dict[str, dict] = {}
_week_change_cache: Dict[str, dict] = {}  # 基金本周涨幅缓存

# === ETF联接基金：动态检测 + 用户手动指定（持久化到文件） ===

_ETF_LINKS_FILE = CACHE_DIR / "etf_links.json"

def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _load_etf_map() -> Dict[str, str]:
    """从文件加载ETF映射"""
    if _ETF_LINKS_FILE.exists():
        try:
            with open(_ETF_LINKS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[ETFMap] 加载映射文件失败: {e}")
    return {}

def _save_etf_map():
    """将ETF映射持久化到文件"""
    _ensure_cache_dir()
    try:
        with open(_ETF_LINKS_FILE, "w", encoding="utf-8") as f:
            json.dump(_user_etf_map, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ETFMap] 保存映射文件失败: {e}")

# 启动时从文件加载
_user_etf_map: Dict[str, str] = _load_etf_map()

def set_etf_link_target(link_code: str, etf_code: str):
    """用户手动设置联接基金的目标ETF"""
    _user_etf_map[link_code] = etf_code
    _save_etf_map()
    print(f"[Holdings] 用户设置ETF映射: {link_code} -> {etf_code}")

def get_etf_link_target(link_code: str) -> Optional[str]:
    """获取用户设置的目标ETF（内存优先，回退到缓存文件中的etf_target）"""
    target = _user_etf_map.get(link_code)
    if target:
        return target
    # 回退：从已有持仓缓存文件中读取etf_target
    cache_path = _get_holdings_cache_path(link_code)
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                etf_target = data.get("etf_target")
                if etf_target:
                    _user_etf_map[link_code] = etf_target
                    _save_etf_map()
                    print(f"[ETFMap] 从缓存恢复映射: {link_code} -> {etf_target}")
                    return etf_target
        except Exception:
            pass
    return None

def clear_etf_link_target(link_code: str):
    """清除用户设置的ETF映射"""
    if link_code in _user_etf_map:
        del _user_etf_map[link_code]
        _save_etf_map()




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
    """根据基金名称判断是否为ETF联接基金
    覆盖所有命名变体: ETF联接、ETF发起联接、ETF发起式联接 等
    """
    if not fund_name:
        return False
    return "联接" in fund_name



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


def _fetch_eastmoney_holdings(fund_code: str, headers: dict) -> Tuple[Optional[List[dict]], Optional[str], float]:
    """
    从天天基金 FundArchivesDatas 获取持仓列表。
    策略：先用 topline=1000 尝试获取年报/半年报全量持仓，
         解析最新一期报告中所有表格，合并去重。
         季报只有前10大，年报/半年报可拿到全部持仓。
    返回 (holdings_list, report_date, parsed_weight)
    """
    try:
        url = (
            f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?"
            f"type=jjcc&code={fund_code}&topline=1000&year=&month=&rt={int(time.time()*1000)}"
        )
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        # === 解析报告日期 ===
        report_date = None
        # 优先匹配季度格式："2025年4季度"
        quarter_match = re.search(r'(\d{4})年(\d)季度', content)
        if quarter_match:
            year = quarter_match.group(1)
            quarter = int(quarter_match.group(2))
            quarter_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            report_date = f"{year}-{quarter_end.get(quarter, '12-31')}"
            # 防御：报告日期不能超过今天（新基金页面可能含未来季度标签）
            try:
                if datetime.strptime(report_date, "%Y-%m-%d") > datetime.now():
                    print(f"[Holdings] {fund_code} 报告日期 {report_date} 超过今天，视为无效")
                    report_date = None
            except Exception:
                pass

        # === 定位最新一期报告的内容边界 ===
        # 页面按时间倒序排列多期报告，用季度标题分割
        first_period_end = None
        quarter_markers = list(re.finditer(r'\d{4}年\d季度', content))
        if len(quarter_markers) >= 2:
            first_period_end = quarter_markers[1].start()

        first_period_content = content[:first_period_end] if first_period_end else content

        # === 从最新一期内容中提取所有表格 ===
        all_tables = re.findall(
            r'<table[^>]*class="w782 comm tzxq"[^>]*>(.*?)</table>',
            first_period_content, re.DOTALL
        )
        if not all_tables:
            all_tables = re.findall(r'<table[^>]*>(.*?)</table>', first_period_content, re.DOTALL)

        if not all_tables:
            return None, report_date, 0.0

        # === 解析所有表格中的持仓行，合并去重 ===
        holdings = []
        parsed_weight = 0.0
        seen_codes = set()

        for table_content in all_tables:
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_content, re.DOTALL)
            for row in rows:
                cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
                if len(cells) < 8:
                    continue

                # 支持5~6位股票代码（兼容港股5位代码如09896）
                code_match = re.search(r'>(\d{5,6})<', cells[1])
                if not code_match:
                    code_match = re.search(r'(\d{5,6})', cells[1])
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

        if holdings:
            holdings.sort(key=lambda x: x["weight"], reverse=True)

        n = len(holdings)
        print(f"[Holdings] {fund_code} 解析到 {n} 只持仓, 权重合计 {parsed_weight:.2f}%"
              f" (report_date={report_date})")

        return holdings, report_date, parsed_weight

    except Exception as e:
        print(f"[Holdings] 天天基金持仓抓取失败 {fund_code}: {e}")
        return None, None, 0.0


def _fetch_holdings_combined(fund_code: str, depth: int = 0) -> Optional[dict]:
    """
    组合数据源获取持仓，优先级：
    1. ETF联接基金穿透（用户映射 + 自动检测母ETF）
    2. 天天基金季报持仓
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

    is_link = _is_etf_link_fund(fund_name)

    if fund_name:
        print(f"[Holdings] {fund_code} fund_name={fund_name}, is_etf_link={is_link}")
    else:
        print(f"[Holdings] {fund_code} fund_name=None (获取名称失败)")

    # === 第二步：用户手动设置的ETF映射 ===
    user_etf_target = get_etf_link_target(fund_code)
    if user_etf_target and depth == 0:
        print(f"[Holdings] 使用用户指定的目标ETF: {fund_code} -> {user_etf_target}")
        etf_holdings = _fetch_holdings_combined(user_etf_target, depth + 1)
        if etf_holdings and etf_holdings.get("positions"):
            etf_pw = etf_holdings.get("parsed_weight", 0)
            # 穿透结果质量校验：如果ETF持仓权重过低（<30%），
            # 说明目标ETF可能是新基金、尚无完整季报，回退到联接基金自身持仓
            if etf_pw < 30:
                print(f"[Holdings] ETF穿透 {user_etf_target} 持仓权重仅 {etf_pw:.1f}%，"
                      f"质量不足，回退到联接基金 {fund_code} 自身持仓")
            else:
                etf_holdings["original_fund_code"] = fund_code
                etf_holdings["fund_code"] = fund_code
                etf_holdings["fund_name"] = fund_name
                etf_holdings["etf_target"] = user_etf_target
                etf_holdings["is_etf_link"] = True
                return etf_holdings

    # === 未配映射的联接基金，直接走天天基金 ===
    if is_link and depth == 0:
        print(f"[Holdings] {fund_code} 是ETF联接基金但未配置etf_links映射, 使用天天基金持仓")

    # === 第三步：天天基金持仓 ===
    em_holdings, report_date, em_parsed_weight = _fetch_eastmoney_holdings(fund_code, headers)

    if em_holdings:
        missing_weight = max(0, stock_position_ratio - em_parsed_weight)
        return {
            "fund_code": fund_code,
            "fund_name": fund_name,
            "holdings_asof_date": report_date,
            "stock_total_weight": round(stock_position_ratio, 2),
            "parsed_weight": round(em_parsed_weight, 2),
            "missing_weight": round(missing_weight, 2),
            "positions": em_holdings,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "eastmoney",
        }

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
    if len(stock_code) == 5:
        # 港股代码（5位），新浪格式为 hk + 5位代码
        return f"hk{stock_code}"
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
                raw = resp.read()
                try:
                    content = raw.decode("gbk")
                except (UnicodeDecodeError, LookupError):
                    content = raw.decode("utf-8", errors="replace")

            now = datetime.now()
            for line in content.strip().split("\n"):
                # A股: var hq_str_sh600000="..." 或 var hq_str_sz000001="..."
                # 港股: var hq_str_hk09896="..."
                match = re.match(r'var hq_str_((?:s[hz]\d{6})|(?:hk\d{5}))="([^"]*)"', line)
                if not match:
                    continue
                symbol, data_str = match.groups()
                if not data_str:
                    continue
                data = data_str.split(",")

                is_hk = symbol.startswith("hk")
                stock_code = symbol[2:]

                try:
                    if is_hk:
                        # 港股新浪格式: 名称,开盘价,昨收,最高,最低,现价,...
                        if len(data) < 6:
                            continue
                        current = float(data[6]) if data[6] else 0
                        yesterday = float(data[3]) if data[3] else 0
                        name = data[1]
                    else:
                        # A股格式: 名称,今开,昨收,现价,...
                        if len(data) < 4:
                            continue
                        current = float(data[3]) if data[3] else 0
                        yesterday = float(data[2]) if data[2] else 0
                        name = data[0]

                    pct = round((current - yesterday) / yesterday * 100, 2) if yesterday > 0 and current > 0 else 0.0
                    all_results[stock_code] = {
                        "stock_code": stock_code,
                        "name": name,
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


def get_etf_realtime_change(etf_code: str) -> Optional[dict]:
    """获取ETF实时涨跌幅（用于商品类ETF联接基金的盘中估值）
    直接复用新浪行情接口，返回 {"pct_change": float, "name": str, "asof_time": str} 或 None
    """
    result = _fetch_quotes_batch_sina([etf_code])
    if etf_code in result:
        return result[etf_code]
    return None


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
    print(f"数据源: {h.get('data_source', 'unknown')}")

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
    """使用东方财富基金业绩API批量获取近1周涨幅"""
    result = {}
    if not fund_codes:
        return result

    try:
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
    """单个基金近1周涨幅 - 使用天天基金净值API"""
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

_REFRESH_AHEAD_DAYS = 3
_REFRESH_INTERVAL_SEC = 2

def _get_tracked_fund_codes() -> List[str]:
    """从 state.json 读取用户跟踪的全部基金代码"""
    state_file = Path(__file__).parent.parent / "data" / "state.json"
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
    静默跳过已知无法获取持仓的基金类型（QDII原油、黄金ETF等），
    仅在汇总中报告，不逐只刷屏。
    """
    _ensure_cache_dir()
    fund_codes = _get_tracked_fund_codes()

    refreshed = []
    failed = []
    failed_detail = {}  # code -> reason
    skipped = 0
    skipped_no_holdings = {}  # code -> reason

    for code in fund_codes:
        # 优先检查是否为已知无持仓类型（必须在 remaining 检查之前）
        cache_path = _get_holdings_cache_path(code)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("_no_holdings"):
                    reason = cached.get("_reason", "")
                    fn = cached.get("fund_name", "")
                    # 旧缓存可能无 _reason 或 fund_name，需重新诊断
                    if not reason or reason == "无持仓数据" or reason == "未知":
                        if not fn:
                            fn = get_fund_name(code) or ""
                        reason = _diagnose_holdings_failure(code, fn)
                        # 回写修正后的缓存
                        cached["_reason"] = reason
                        cached["fund_name"] = fn
                        try:
                            with open(cache_path, "w", encoding="utf-8") as fw:
                                json.dump(cached, fw, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
                    skipped_no_holdings[code] = reason
                    cache_path.touch()
                    continue
            except Exception:
                pass

        remaining = _holdings_cache_remaining_days(code)

        if remaining is not None and remaining > _REFRESH_AHEAD_DAYS:
            skipped += 1
            continue

        label = "expired" if remaining is None or remaining <= 0 else f"{remaining:.1f}d left"
        print(f"[AutoRefresh] {code} ({label}) -> refreshing...")

        try:
            data = _fetch_holdings_combined(code)
            if data and data.get("positions"):
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                refreshed.append(code)
                src = data.get("data_source", "?")
                n = len(data["positions"])
                pw = data.get("parsed_weight", 0)
                print(f"[AutoRefresh] {code} OK, {n} positions, {pw:.1f}% (src={src})")
            else:
                failed.append(code)
                # 获取基金名用于诊断
                fn = data.get("fund_name", "") if data else ""
                if not fn:
                    fn = get_fund_name(code) or ""
                reason = _diagnose_holdings_failure(code, fn)
                failed_detail[code] = f"{fn} | {reason}"
                # 写入标记缓存，下次静默跳过
                no_hold_cache = {
                    "fund_code": code,
                    "fund_name": fn,
                    "_no_holdings": True,
                    "_reason": reason,
                    "positions": [],
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                try:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(no_hold_cache, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        except Exception as e:
            failed.append(code)
            failed_detail[code] = f"ERROR: {e}"
            print(f"[AutoRefresh] {code} ERROR: {e}")

        if code != fund_codes[-1]:
            time.sleep(_REFRESH_INTERVAL_SEC)

    # === 汇总日志 ===
    if failed:
        by_reason = {}
        for code, detail in failed_detail.items():
            reason = detail.split(" | ")[-1] if " | " in detail else detail
            by_reason.setdefault(reason, []).append(code)
        print(f"\n[AutoRefresh] === {len(failed)} 只基金无持仓数据（按原因分类）===")
        for reason, codes in by_reason.items():
            print(f"  [{reason}] ({len(codes)}只): {', '.join(codes[:5])}{'...' if len(codes) > 5 else ''}")
        print(f"[AutoRefresh] 这些基金的盘中估值将依赖 fundgz 接口或纯净值模式")

    if skipped_no_holdings:
        by_reason = {}
        for code, reason in skipped_no_holdings.items():
            by_reason.setdefault(reason, []).append(code)
        print(f"\n[AutoRefresh] === 非标准估值基金 ({len(skipped_no_holdings)}只，走 fundgz/净值模式) ===")
        for reason, codes in by_reason.items():
            print(f"  [{reason}] ({len(codes)}只): {', '.join(codes)}")

    # 打印未配映射的ETF联接基金
    unmapped = []
    for code in fund_codes:
        if get_etf_link_target(code):
            continue
        cache_path_check = _get_holdings_cache_path(code)
        if cache_path_check.exists():
            try:
                with open(cache_path_check, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fn = data.get("fund_name", "")
                if "联接" in fn and not data.get("is_etf_link") and not data.get("_no_holdings"):
                    unmapped.append(f"{code} {fn}")
            except Exception:
                pass

    if unmapped:
        print(f"\n[AutoRefresh] === {len(unmapped)}只ETF联接基金未配置映射 ===")
        for item in unmapped:
            print(f"  {item}")

    summary = {
        "refreshed": refreshed, "failed": failed,
        "skipped": skipped, "skipped_no_holdings": len(skipped_no_holdings),
        "unmapped_etf_links": unmapped,
    }
    print(f"\n[AutoRefresh] Done: {len(refreshed)} refreshed, {len(failed)} failed, "
          f"{skipped} cached, {len(skipped_no_holdings)} no-holdings-skipped")
    return summary


def _diagnose_holdings_failure(fund_code: str, fund_name: str) -> str:
    """诊断持仓获取失败的原因，返回分类标签"""
    fn = fund_name or ""
    # QDII基金投资海外，天天基金无A股持仓
    if "QDII" in fn or "标普" in fn or "道琼斯" in fn or "纳斯达克" in fn:
        return "QDII海外投资"
    # 原油类基金（FOF结构，持仓为其他基金份额）
    if "原油" in fn or "石油" in fn or "油气" in fn:
        return "原油/油气FOF"
    # 黄金ETF联接（持仓为黄金合约，非A股）
    if "黄金" in fn or "上海金" in fn or "金ETF" in fn:
        return "黄金ETF/合约"
    # 抗通胀FOF
    if "抗通胀" in fn:
        return "抗通胀FOF"
    # 其他联接基金
    if "联接" in fn:
        return "ETF联接无持仓"
    # 通用
    return "无持仓数据"


# ============================================================
# 天天基金盘中估值接口（fundgz）
# 用于无法获取持仓的基金（QDII/指数/FOF等）的备用估值源
# ============================================================

_fundgz_cache: Dict[str, dict] = {}
_FUNDGZ_TTL = 60  # 缓存60秒

def get_fundgz_estimation(fund_code: str) -> Optional[dict]:
    """
    从天天基金 fundgz 接口获取盘中估值。
    返回 {"gsz": 估算净值, "gszzl": 估算涨跌幅%, "dwjz": 上日净值,
           "gztime": 估值时间, "name": 基金名称} 或 None
    注意：非交易时段返回的是上一次估值快照，QDII基金24小时更新。
    """
    # 缓存检查
    if fund_code in _fundgz_cache:
        cached = _fundgz_cache[fund_code]
        if time.time() - cached["ts"] < _FUNDGZ_TTL:
            return cached["data"]

    try:
        url = f"http://fundgz.1234567.com.cn/js/{fund_code}.js?rt={int(time.time()*1000)}"
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://fund.eastmoney.com/",
        })
        with urlopen(req, timeout=8) as resp:
            content = resp.read().decode("utf-8")

        match = re.search(r'jsonpgz\((.*)\)', content)
        if match:
            data = json.loads(match.group(1))
            gszzl = data.get("gszzl", "")
            if gszzl and gszzl.strip():
                _fundgz_cache[fund_code] = {"data": data, "ts": time.time()}
                return data
    except Exception:
        pass

    # 标记无数据，避免短时间内重复请求
    _fundgz_cache[fund_code] = {"data": None, "ts": time.time()}
    return None


# ============================================================
# 基金历史净值（逐日真实涨跌幅）
# ============================================================

_nav_history_cache: Dict[str, dict] = {}  # key=fund_code, value={"data": list, "ts": float}
_NAV_HISTORY_TTL = 3600  # 1小时缓存
_NAV_HISTORY_FETCH_SIZE = 30  # 固定请求30条，缓存完整数据


def get_fund_nav_history(fund_code: str, days: int = 15) -> list:
    """
    获取基金最近N个交易日的真实净值数据（收盘结算后的精确值）。

    返回格式（按日期降序，最新的在前）：
    [
        {"date": "2026-02-19", "nav": 1.2345, "change": -1.23},
        {"date": "2026-02-18", "nav": 1.2500, "change": 0.85},
        ...
    ]

    其中 change 是当日真实涨跌幅%（来自JZZZL字段）。
    缓存策略：内存缓存，TTL = 1小时（历史数据不需要频繁刷新）。
    """
    # 检查缓存
    cache_key = fund_code
    if cache_key in _nav_history_cache:
        cached = _nav_history_cache[cache_key]
        if time.time() - cached["ts"] < _NAV_HISTORY_TTL:
            # 缓存数据够用，直接截取
            if len(cached["data"]) >= days:
                return cached["data"][:days]
            # 缓存数据不够，重新拉取

    # 请求时用 max(days, _NAV_HISTORY_FETCH_SIZE) 确保拿够
    fetch_size = max(days, _NAV_HISTORY_FETCH_SIZE)

    # 请求天天基金历史净值API
    try:
        url = (
            f"https://api.fund.eastmoney.com/f10/lsjz?"
            f"fundCode={fund_code}&pageIndex=1&pageSize={fetch_size}"
        )
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://fundf10.eastmoney.com/jjjz_{fund_code}.html",
        })
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        data = json.loads(content)
        items = data.get("Data", {}).get("LSJZList", [])

        result = []
        for item in items:
            date_str = item.get("FSRQ", "")
            nav_str = item.get("DWJZ", "")
            change_str = item.get("JZZZL", "")

            try:
                nav = float(nav_str) if nav_str else None
            except (ValueError, TypeError):
                nav = None

            try:
                change = float(change_str) if change_str and change_str.strip() else None
            except (ValueError, TypeError):
                change = None

            result.append({
                "date": date_str,
                "nav": nav,
                "change": change,
            })

        # 写入缓存
        _nav_history_cache[cache_key] = {"data": result, "ts": time.time()}
        # 静默成功，仅失败时输出日志
        return result[:days]

    except Exception as e:
        print(f"[NavHistory] {fund_code} 获取失败: {e}")
        return []