#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search missing ETF links by querying fund data"""
import json
import re
import sys
from urllib.request import Request, urlopen, install_opener, build_opener, ProxyHandler
install_opener(build_opener(ProxyHandler({})))

# Load missing funds
missing = [
    ("026613", "国泰上证科创板人工智能ETF联接A"),
    ("024561", "博时上证科创板人工智能ETF发起式C"),
    ("025653", "易方达创业板人工智能ETF发起式C"),
    ("013446", "华安中证芯片ETF发起式C"),
    ("020227", "国泰中证全指集成电路ETF发起联接C"),
    ("020705", "南方中证通信服务ETF发起联接C"),
    ("019237", "广发中证通信ETF发起式C"),
    ("019072", "国泰中证通信ETF发起联接C"),
    ("015895", "平安中证消费电子主题ETF发起式C"),
    ("015877", "广发中证消费电子主题ETF发起式C"),
    ("021098", "南方中证创新药产业ETF发起联接C"),
    ("012782", "华安中证创新药产业ETF发起式C"),
    ("025791", "新华中证云计算50ETF发起式C"),
    ("021398", "广发中证科技龙头ETF发起式C"),
    ("021717", "华宝中证科技龙头ETF发起式C"),
    ("021091", "易方达中证科技龙头ETF发起式C"),
    ("019331", "国泰创业板大盘龙头ETF发起式C"),
    ("020336", "华宝中证大数据产业ETF发起式C"),
    ("018135", "易方达中证大数据产业ETF发起式C"),
    ("021856", "国泰中证光伏产业ETF发起式C"),
    ("014605", "嘉实中证光伏产业ETF发起联接C"),
    ("021297", "国联中证有色金属产业ETF发起式C"),
    ("026478", "国联中证有色金属产业ETF发起式C"),
    ("022084", "国联中证有色金属产业ETF发起式C"),
    ("017141", "华宝中证有色金属ETF发起式C"),
    ("017030", "国泰标普500ETF发起联接(QDII)C美元"),
    ("018065", "万家中证500指数增强(QDII)C"),
]

# Method: Use eastmoney fund API to search for what each fund tracks
# The FundArchivesDatas API contains position info
# But we need the ETF code, not holdings
# Let's try getting the fund's subject/index info from the pingzhongdata

def search_etf_code(fund_name, fund_code):
    """For a fund name like 'XXX ETF联接C', search for corresponding ETF code"""
    name = fund_name
    
    # Remove ETF-related suffixes
    for suffix in ['ETF联接C', 'ETF联接A', 'ETF联接式C', 'ETF联接式A',
                   'ETF联接', '联接C', '联接A', '联接式C', '联接式A',
                   '(QDII)C', '(QDII)A',
                   '发起式', '发起联接',
                   'C份额', 'A份额', 'ETF']:
        if suffix in name:
            name = name[:name.index(suffix)]
    
    # Remove company prefix
    companies = ['国泰', '华宝', '南方', '华夏', '广发', '富国', '嘉实', '易方达',
                 '博时', '招商', '鹏华', '汇添富', '工银', '天弘', '华安', '景顺长城',
                 '银华', '万家', '中欧', '建信', '交银', '兴全', '大成', '平安',
                 '中金', '长城', '中信保诚', '光大保德信', '前海开源',
                 '华泰柏瑞', '申万菱信', '方正富邦', '中银', '金鹰',
                 '民生加银', '兴业', '国联安', '国投瑞银', '华富', '诺德',
                 '融通', '泰信', '新华', '信达澳亚', '华商', '东方',
                 '创金合信', '九泰', '泓德', '永赢', '红土创新', '中航', '摩根',
                 '贝莱德', '太平', '富荣', '国融', '金信', '恒越', '兴银',
                 '汇安', '中庚', '华泰保兴', '达诚', '百嘉',
                 '汇泉', '西部利得', '易米', '上银']
    
    for comp in companies:
        if name.startswith(comp):
            name = name[len(comp):]
            break
    
    # Clean up
    name = name.strip()
    return name

def search_via_pingzhongdata(fund_code):
    """Search pingzhongdata JS for subject/index info"""
    try:
        url = f'http://fund.eastmoney.com/pingzhongdata/{fund_code}.js'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Referer': f'http://fund.eastmoney.com/{fund_code}.html'
        }
        req = Request(url, headers=headers)
        with urlopen(req, timeout=8) as resp:
            content = resp.read().decode('utf-8')
        
        # Look for subject/tracking info
        info = {}
        
        # Fund name
        m = re.search(r'var\s+fS_name\s*=\s*"([^"]+)"', content)
        if m: info['name'] = m.group(1)
        
        # Subject code
        m = re.search(r'fS_subject\s*=\s*"([^"]+)"', content)
        if m: info['subject'] = m.group(1)
        
        # ETF code
        m = re.search(r'var\s+etf_code\s*=\s*"([^"]+)"', content)
        if m: info['etf_code'] = m.group(1)
        
        # Base code (sometimes has linked ETF)
        m = re.search(r'var\s+base_code\s*=\s*"([^"]+)"', content)
        if m: info['base_code'] = m.group(1)
        
        return info
    except Exception as e:
        return {'error': str(e)}

# Load full fund catalog
with open(r'E:\Git\valuation_grid\cache\etf_catalog.json', 'r', encoding='utf-8') as f:
    all_funds = json.load(f)

# Also load ETF only catalog - filter SH/SZ ETF codes
# ETF codes: 51xxxx (SH), 159xxx (SZ), 588xxx (科创)
etf_only = {}
for code, name in all_funds.items():
    if (code.startswith('51') or code.startswith('159') or code.startswith('588') or
        code.startswith('56') or code.startswith('52') or code.startswith('16') or
        code.startswith('11')) and 'ETF' in name:
        etf_only[code] = name

print(f"ETF catalog: {len(etf_only)} ETFs")

for fc, fn in missing:
    print(f"\n{'='*60}")
    print(f"Searching: {fc} {fn}")
    
    # Get fund info from pingzhongdata
    info = search_via_pingzhongdata(fc)
    print(f"  PingZhongData: {json.dumps(info, ensure_ascii=False)}")
    
    # Try to find ETF by name matching
    search_key = search_etf_code(fn, fc)
    print(f"  Search key: '{search_key}'")
    
    # Search in ETF catalog
    candidates = []
    for code, name in etf_only.items():
        # Remove company prefix from ETF name for matching
        if search_key and search_key in name:
            candidates.append((code, name))
    
    if candidates:
        print(f"  Candidates: {candidates[:5]}")
    else:
        print(f"  No match found in catalog")
