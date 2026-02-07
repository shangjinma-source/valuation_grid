"""
providers.py - 持仓抓取 + 行情拉取（含缓存/超时/批量）
支持：普通基金、ETF联接基金穿透、用户截图导入
增强：中证指数成分股权重下载（ETF/指数基金全覆盖）、年报全持仓解析
"""
import json
import time
import re
import io
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from typing import Dict, List, Optional, Tuple

# === 配置常量 ===
CACHE_DIR = Path(__file__).parent / "cache"
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

# ============================================================
# 指数代码映射：ETF/指数基金 -> 跟踪指数代码（持久化）
# ============================================================

_INDEX_MAP_FILE = CACHE_DIR / "index_map.json"

def _load_index_map() -> Dict[str, str]:
    """从文件加载 基金代码 -> 跟踪指数代码 映射"""
    if _INDEX_MAP_FILE.exists():
        try:
            with open(_INDEX_MAP_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {}

def _save_index_map():
    _ensure_cache_dir()
    try:
        with open(_INDEX_MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(_index_map, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

_index_map: Dict[str, str] = _load_index_map()


def _detect_tracking_index(fund_code: str, fund_name: Optional[str] = None) -> Optional[str]:
    """
    自动检测基金跟踪的指数代码。
    优先从缓存映射读取，否则从天天基金详情页/API解析。
    """
    # 1. 缓存命中
    if fund_code in _index_map:
        return _index_map[fund_code]

    # 2. 从天天基金 tsdata 页面解析跟踪指数
    try:
        url = f"https://fundf10.eastmoney.com/tsdata_{fund_code}.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": f"https://fundf10.eastmoney.com/{fund_code}.html",
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        # 页面中"跟踪标的"后面通常有指数代码链接
        idx_match = re.search(r'跟踪标的.*?(\d{6})', content, re.DOTALL)
        if idx_match:
            index_code = idx_match.group(1)
            _index_map[fund_code] = index_code
            _save_index_map()
            print(f"[IndexMap] 检测到跟踪指数: {fund_code} -> {index_code}")
            return index_code
        else:
            # 页面获取成功但正则未匹配，打印片段辅助排查
            snippet = content[:500].replace('\n', ' ')
            print(f"[IndexMap] tsdata未匹配到指数代码 {fund_code}, 页面片段: {snippet[:200]}")
    except Exception as e:
        print(f"[IndexMap] tsdata页面解析失败 {fund_code}: {e}")

    # 3. 从基金详情API解析 INDEXCODE
    try:
        url = (
            f"https://fundmobapi.eastmoney.com/FundMNewApi/"
            f"FundMNNBasicInformation?FCODE={fund_code}"
            f"&deviceid=&plat=Iphone&product=EFund&Version=1"
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://fund.eastmoney.com/",
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")
        data = json.loads(content)
        datas = data.get("Datas") or {}
        index_code = datas.get("INDEXCODE", "")
        if index_code and index_code != "--" and len(index_code) == 6:
            _index_map[fund_code] = index_code
            _save_index_map()
            print(f"[IndexMap] 从API检测跟踪指数: {fund_code} -> {index_code}")
            return index_code
        else:
            print(f"[IndexMap] API INDEXCODE无效 {fund_code}: INDEXCODE='{index_code}'")

        # 4. 从业绩比较基准(BENCH/PERFCMP)中提取指数代码
        bench = datas.get("BENCH", "") or datas.get("PERFCMP", "")
        if bench:
            # 提取所有6位数字，取第一个看起来像指数代码的
            all_codes = re.findall(r'(\d{6})', bench)
            for idx in all_codes:
                if idx[0] in ('0', '3', '9'):
                    _index_map[fund_code] = idx
                    _save_index_map()
                    print(f"[IndexMap] 从BENCH提取跟踪指数: {fund_code} -> {idx}")
                    return idx
            print(f"[IndexMap] BENCH未找到有效指数代码 {fund_code}: BENCH='{bench}', 提取到的代码={all_codes}")
        else:
            print(f"[IndexMap] BENCH字段为空 {fund_code}")
    except Exception as e:
        print(f"[IndexMap] API解析失败 {fund_code}: {e}")

    print(f"[IndexMap] 所有检测方法均失败 {fund_code}")
    return None


# ============================================================
# 中证指数成分股权重下载
# ============================================================

def _fetch_csindex_weights(index_code: str) -> Optional[List[dict]]:
    """
    从中证指数官网下载完整的成分股权重数据(xls文件)。
    URL: https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/
         static/html/csindex/public/uploads/file/autofile/closeweight/
         {index_code}closeweight.xls
    返回: [{"stock_code": "601899", "stock_name": "紫金矿业", "weight": 15.3}, ...]
    """
    url = (
        f"https://csi-web-dev.oss-cn-shanghai-finance-1-pub.aliyuncs.com/"
        f"static/html/csindex/public/uploads/file/autofile/closeweight/"
        f"{index_code}closeweight.xls"
    )
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    print(f"[CSIndex] 正在下载 {index_code} 权重文件: {url}")

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"[CSIndex] {index_code} 非200响应: {resp.status}")
                return None
            raw = resp.read()
            print(f"[CSIndex] {index_code} 下载成功, 大小={len(raw)} bytes")

        # 尝试用 xlrd 解析真正的 .xls
        try:
            import xlrd
            wb = xlrd.open_workbook(file_contents=raw)
            result = _parse_xls_with_xlrd(wb, index_code)
            if result is None:
                print(f"[CSIndex] {index_code} xlrd解析返回空, 尝试HTML回退")
            else:
                return result
        except ImportError:
            print(f"[CSIndex] xlrd未安装, 回退到HTML解析")
        except Exception as e:
            print(f"[CSIndex] {index_code} xlrd解析异常: {e}, 回退到HTML解析")

        # 回退: 很多 .xls 实际上是 HTML 表格格式
        result = _parse_xls_as_html(raw, index_code)
        if result is None:
            print(f"[CSIndex] {index_code} HTML回退解析也返回空")
        return result

    except HTTPError as e:
        if e.code == 404:
            print(f"[CSIndex] {index_code} 权重文件不存在(404)")
        else:
            print(f"[CSIndex] {index_code} HTTP错误: {e.code}")
    except Exception as e:
        print(f"[CSIndex] {index_code} 下载失败: {e}")

    return None


def _parse_xls_with_xlrd(wb, index_code: str) -> Optional[List[dict]]:
    """用 xlrd 解析真正的 xls 二进制文件"""
    sheet = wb.sheet_by_index(0)
    if sheet.nrows < 2:
        return None

    header_row = [str(sheet.cell_value(0, c)).strip() for c in range(sheet.ncols)]
    code_col = name_col = weight_col = -1
    for i, h in enumerate(header_row):
        h_lower = h.lower()
        if '代码' in h or 'code' in h_lower:
            code_col = i
        elif '名称' in h or 'name' in h_lower:
            name_col = i
        elif '权重' in h or 'weight' in h_lower:
            weight_col = i

    if code_col == -1 or weight_col == -1:
        print(f"[CSIndex] 无法识别列头: {header_row}")
        return None

    holdings = []
    for r in range(1, sheet.nrows):
        try:
            raw_code = sheet.cell_value(r, code_col)
            if isinstance(raw_code, float):
                raw_code = str(int(raw_code))
            else:
                raw_code = str(raw_code).strip()
            stock_code = raw_code.zfill(6)

            stock_name = ""
            if name_col >= 0:
                stock_name = str(sheet.cell_value(r, name_col)).strip()

            weight_val = sheet.cell_value(r, weight_col)
            if isinstance(weight_val, str):
                weight_val = weight_val.replace('%', '').strip()
            weight = float(weight_val)

            if weight <= 0 or len(stock_code) != 6:
                continue

            holdings.append({
                "stock_code": stock_code,
                "stock_name": stock_name or stock_code,
                "weight": round(weight, 4)
            })
        except (ValueError, TypeError):
            continue

    if holdings:
        holdings.sort(key=lambda x: x["weight"], reverse=True)
        total = sum(h["weight"] for h in holdings)
        print(f"[CSIndex] {index_code} xlrd解析: {len(holdings)} 只, 总权重 {total:.2f}%")
        return holdings
    return None


def _parse_xls_as_html(raw: bytes, index_code: str) -> Optional[List[dict]]:
    """有些 .xls 文件实际上是 HTML 格式的表格"""
    try:
        content = None
        for encoding in ('utf-8', 'gbk', 'gb2312', 'latin-1'):
            try:
                content = raw.decode(encoding)
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if not content:
            return None

        if '<table' not in content.lower() and '<tr' not in content.lower():
            return None

        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL | re.IGNORECASE)
        if len(rows) < 2:
            return None

        header_cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', rows[0], re.DOTALL | re.IGNORECASE)
        header_cells = [re.sub(r'<[^>]+>', '', c).strip() for c in header_cells]

        code_col = name_col = weight_col = -1
        for i, h in enumerate(header_cells):
            if '代码' in h:
                code_col = i
            elif '名称' in h:
                name_col = i
            elif '权重' in h:
                weight_col = i

        if code_col == -1 or weight_col == -1:
            return None

        holdings = []
        for row in rows[1:]:
            cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL | re.IGNORECASE)
            cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
            if len(cells) <= max(code_col, weight_col):
                continue

            code_match = re.search(r'(\d{5,6})', cells[code_col])
            if not code_match:
                continue
            stock_code = code_match.group(1).zfill(6)

            stock_name = ""
            if 0 <= name_col < len(cells):
                stock_name = cells[name_col].strip()

            try:
                weight = float(cells[weight_col].replace('%', '').strip())
            except (ValueError, TypeError):
                continue

            if weight <= 0:
                continue

            holdings.append({
                "stock_code": stock_code,
                "stock_name": stock_name or stock_code,
                "weight": round(weight, 4)
            })

        if holdings:
            holdings.sort(key=lambda x: x["weight"], reverse=True)
            total = sum(h["weight"] for h in holdings)
            print(f"[CSIndex] {index_code} HTML解析: {len(holdings)} 只, 总权重 {total:.2f}%")
            return holdings

    except Exception as e:
        print(f"[CSIndex] HTML解析失败 {index_code}: {e}")
    return None


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


def _is_index_fund(fund_name: Optional[str]) -> bool:
    """判断是否为指数基金(ETF/指数增强/被动跟踪等)"""
    if not fund_name:
        return False
    keywords = ["ETF", "指数"]
    return any(kw in fund_name for kw in keywords)


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


def _fetch_eastmoney_holdings(fund_code: str, headers: dict) -> Tuple[Optional[List[dict]], Optional[str], float]:
    """
    从天天基金 FundArchivesDatas 获取持仓列表。
    返回 (holdings_list, report_date, parsed_weight)
    """
    try:
        url = (
            f"https://fundf10.eastmoney.com/FundArchivesDatas.aspx?"
            f"type=jjcc&code={fund_code}&topline=100&year=&month=&rt={int(time.time()*1000)}"
        )
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            content = resp.read().decode("utf-8")

        # 解析报告日期
        report_date = None
        quarter_match = re.search(r'(\d{4})年(\d)季度', content)
        if quarter_match:
            year = quarter_match.group(1)
            quarter = int(quarter_match.group(2))
            quarter_end = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
            report_date = f"{year}-{quarter_end.get(quarter, '12-31')}"

        # 解析第一个表格（最新一期持仓）
        first_table = re.search(
            r'<table[^>]*class="w782 comm tzxq"[^>]*>(.*?)</table>',
            content, re.DOTALL
        )
        if not first_table:
            first_table = re.search(r'<table[^>]*>(.*?)</table>', content, re.DOTALL)

        if not first_table:
            return None, report_date, 0.0

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

        if holdings:
            holdings.sort(key=lambda x: x["weight"], reverse=True)

        return holdings, report_date, parsed_weight

    except Exception as e:
        print(f"[Holdings] 天天基金持仓抓取失败 {fund_code}: {e}")
        return None, None, 0.0


def _fetch_holdings_combined(fund_code: str, depth: int = 0) -> Optional[dict]:
    """
    组合数据源获取持仓，优先级：
    1. ETF联接基金穿透（用户映射）
    2. 指数基金/ETF -> 中证指数权重下载（全部成分股+精确权重）
    3. 普通基金 -> 天天基金持仓
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

    if fund_name:
        print(f"[Holdings] {fund_code} fund_name={fund_name}, is_index={_is_index_fund(fund_name)}")
    else:
        print(f"[Holdings] {fund_code} fund_name=None (获取名称失败)")

    # === 第二步：用户手动设置的ETF映射 ===
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

    # === 第三步：指数基金/ETF -> 尝试中证指数权重 ===
    csindex_result = None
    is_idx = _is_index_fund(fund_name)
    if is_idx:
        print(f"[Holdings] {fund_code} 识别为指数基金({fund_name}), 尝试获取中证指数权重...")
        index_code = _detect_tracking_index(fund_code, fund_name)
        if index_code:
            csindex_holdings = _fetch_csindex_weights(index_code)
            if csindex_holdings and len(csindex_holdings) > 0:
                total_weight = sum(h["weight"] for h in csindex_holdings)
                csindex_result = {
                    "fund_code": fund_code,
                    "fund_name": fund_name,
                    "holdings_asof_date": datetime.now().strftime("%Y-%m-%d"),
                    "stock_total_weight": round(stock_position_ratio, 2),
                    "parsed_weight": round(total_weight, 2),
                    "missing_weight": round(max(0, stock_position_ratio - total_weight), 2),
                    "positions": csindex_holdings,
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "data_source": "csindex",
                    "tracking_index": index_code,
                }

    # === 第四步：天天基金持仓（始终获取，作为对比/回退） ===
    em_holdings, report_date, em_parsed_weight = _fetch_eastmoney_holdings(fund_code, headers)

    # === 第五步：选择最优数据源 ===
    if csindex_result:
        cs_weight = csindex_result["parsed_weight"]
        if cs_weight > em_parsed_weight:
            print(
                f"[Holdings] {fund_code} 使用中证指数 "
                f"({len(csindex_result['positions'])}只, {cs_weight:.1f}%) "
                f"> 天天基金 ({len(em_holdings or [])}只, {em_parsed_weight:.1f}%)"
            )
            if report_date:
                csindex_result["holdings_asof_date"] = report_date
            return csindex_result

    # 回退到天天基金
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
    print(f"数据源: {h.get('data_source', 'unknown')}")

    if h.get("is_etf_link"):
        print(f"ETF联接穿透: {h.get('etf_target')}")
    if h.get("tracking_index"):
        print(f"跟踪指数: {h.get('tracking_index')}")

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
    """
    _ensure_cache_dir()
    fund_codes = _get_tracked_fund_codes()

    refreshed = []
    failed = []
    skipped = 0

    for code in fund_codes:
        remaining = _holdings_cache_remaining_days(code)

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
                src = data.get("data_source", "?")
                n = len(data["positions"])
                pw = data.get("parsed_weight", 0)
                print(f"[AutoRefresh] {code} OK, {n} positions, {pw:.1f}% (src={src})")
            else:
                failed.append(code)
                print(f"[AutoRefresh] {code} FAILED: no positions returned")
        except Exception as e:
            failed.append(code)
            print(f"[AutoRefresh] {code} ERROR: {e}")

        if code != fund_codes[-1]:
            time.sleep(_REFRESH_INTERVAL_SEC)

    summary = {"refreshed": refreshed, "failed": failed, "skipped": skipped}
    print(f"[AutoRefresh] Done: {len(refreshed)} refreshed, {len(failed)} failed, {skipped} skipped")
    return summary