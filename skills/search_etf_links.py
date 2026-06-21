#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search ETF links using eastmoney fund search data
The fundcode_search.js contains all fund codes, names, types
"""
import json
import re
import sys
from urllib.request import Request, urlopen, install_opener, build_opener, ProxyHandler
install_opener(build_opener(ProxyHandler({})))

def fetch_all_funds():
    """Fetch all funds from eastmoney fund search"""
    print("Fetching all funds from eastmoney...")
    
    url = 'http://fund.eastmoney.com/js/fundcode_search.js'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://fund.eastmoney.com/'
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=15) as resp:
        content = resp.read().decode('utf-8-sig')  # UTF-8 with BOM
    
    # Parse JS array: var r = [[code, type, name, name2, pinyin], ...]
    match = re.search(r'var\s+r\s*=\s*(\[.*?\])\s*;', content, re.DOTALL)
    if not match:
        print("Failed to parse fund list")
        return {}
    
    all_funds = json.loads(match.group(1))
    
    # Build ETF catalog: code -> name for ETFs (type 5 = ETF联接? Let's check)
    etf_catalog = {}
    for f in all_funds:
        code, ftype, fname, fname2, pinyin = f[0], f[1], f[2], f[3], f[4]
        # ETF type codes: search for ETFs
        # Type 1 = stock fund, 2 = mixed, 3 = bond, 4 = index, 5 = ETF link, etc.
        etf_catalog[code] = fname
    
    print(f"Total funds: {len(all_funds)}")
    print(f"Total in catalog: {len(etf_catalog)}")
    
    # Save catalog
    with open(r'E:\Git\valuation_grid\cache\etf_catalog.json', 'w', encoding='utf-8') as f:
        json.dump(etf_catalog, f, ensure_ascii=False, indent=2)
    
    return etf_catalog, all_funds

def extract_etf_name(fund_name):
    """Extract the underlying ETF name from an ETF link fund name"""
    name = fund_name
    
    # Remove suffixes
    for suffix in ['ETF联接C', 'ETF联接A', 'ETF联接式C', 'ETF联接式A',
                   'ETF联接', '联接C', '联接A', '联接式C', '联接式A',
                   '(QDII)C', '(QDII)A', '(FOF)C', '(FOF)A',
                   'C份额', 'A份额', 'C', 'A',
                   '(QDII-LOF)C', '(QDII-LOF)A', '(LOF)C', '(LOF)A']:
        if suffix in name:
            name = name[:name.index(suffix)]
    
    # Remove company prefix (known Chinese fund companies)
    companies = ['国泰', '华宝', '南方', '华夏', '广发', '富国', '嘉实', '易方达',
                 '博时', '招商', '鹏华', '汇添富', '工银', '天弘', '华安', '景顺长城',
                 '银华', '万家', '中欧', '建信', '交银', '兴全', '大成', '平安',
                 '中金', '长城', '中信保诚', '光大保德信', '德邦', '前海开源',
                 '华泰柏瑞', '申万菱信', '方正富邦', '中银', '金鹰', '浦银安盛',
                 '民生加银', '兴业', '国联安', '国投瑞银', '华富', '诺德',
                 '融通', '泰信', '新华', '信达澳亚', '华商', '东方', '东海', '东吴',
                 '创金合信', '九泰', '泓德', '永赢', '红土创新', '中航', '摩根',
                 '贝莱德', '路博迈', '太平', '富荣', '国融', '金信', '恒越', '兴银',
                 '汇安', '先锋', '中庚', '华泰保兴', '达诚', '百嘉', '长江',
                 '汇泉', '西部利得', '易米', '百嘉', '中科沃土', '上银']
    
    for comp in companies:
        if name.startswith(comp):
            name = name[len(comp):]
            break
    
    # Remove 'ETF' remaining
    name = name.replace('ETF', '').strip()
    
    return name

def main():
    etf_catalog, all_funds = fetch_all_funds()
    
    # Load link funds to search
    with open(r'E:\Git\valuation_grid\cache\etf_links_to_search.json', 'r', encoding='utf-8') as f:
        link_funds = json.load(f)
    
    # Load existing etf_links
    existing = {}
    try:
        with open(r'E:\Git\valuation_grid\cache\etf_links.json', 'r', encoding='utf-8') as f:
            existing = json.load(f)
    except:
        existing = {}
    
    print(f"Existing ETF links: {len(existing)}")
    print(f"ETF link funds to search: {len(link_funds)}")
    
    new_links = {}
    
    for fund in link_funds:
        code = fund['code']
        name = fund['name']
        
        if code in existing:
            print(f"  [KEEP] {code} -> {existing[code]}")
            continue
        
        # Extract underlying ETF name
        etf_name_key = extract_etf_name(name)
        
        if not etf_name_key:
            print(f"  [FAIL] {code} {name} -> cannot extract ETF name")
            continue
        
        # Search ETF catalog for match
        best_match = None
        best_score = 0
        
        for etf_code, etf_fullname in etf_catalog.items():
            # ETF code: 6 digits starting with 1 (SH), 5 (SZ) or 0 (SZ)
            if not (len(etf_code) == 6):
                continue
                
            # Check if extracted name is in ETF full name
            if etf_name_key in etf_fullname:
                # Prefer exact match (shorter extracted name vs longer ETF name)
                score = len(etf_name_key) / max(len(etf_fullname), 1)
                if score > best_score:
                    best_score = score
                    best_match = etf_code
        
        if best_match:
            new_links[code] = best_match
            print(f"  [MATCH] {code} {name}")
            print(f"          -> {best_match} {etf_catalog[best_match]}")
        else:
            print(f"  [MISS]  {code} {name}")
            print(f"          search_key: '{etf_name_key}'")
    
    print(f"\nNew ETF links found: {len(new_links)}")
    
    # Merge and save
    merged = {**existing, **new_links}
    with open(r'E:\Git\valuation_grid\cache\etf_links.json', 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    
    print(f"Updated etf_links.json: {len(existing)} -> {len(merged)} entries")
    
    # Print summary of missing
    missing = [f for f in link_funds if f['code'] not in merged]
    if missing:
        print(f"\nStill missing ETF links ({len(missing)}):")
        for f in missing:
            print(f"  {f['code']} {f['name']}")

if __name__ == '__main__':
    main()
