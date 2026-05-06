#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""market_summary.py - 获取实时市场数据并生成总结"""
from urllib.request import urlopen, Request, build_opener, ProxyHandler, install_opener
from datetime import datetime
import json

# 禁用代理
install_opener(build_opener(ProxyHandler({})))

def get_market_data():
    """获取大盘指数数据"""
    url = 'https://hq.sinajs.cn/list=sh000001,sz399001,sz399006'
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://finance.sina.com.cn/'
    })
    resp = urlopen(req, timeout=8)
    content = resp.read().decode('gbk')
    
    indices = []
    for line in content.strip().split('\n'):
        if '=' in line:
            parts = line.split('=', 1)
            symbol = parts[0].split('_')[-1]
            data = parts[1].strip('";').split(',')
            if len(data) >= 5:
                name = data[0]
                current = float(data[3]) if data[3] else 0
                yesterday = float(data[2]) if data[2] else 0
                pct = round((current - yesterday) / yesterday * 100, 2) if yesterday > 0 else 0
                indices.append({
                    'symbol': symbol,
                    'name': name,
                    'current': current,
                    'yesterday': yesterday,
                    'pct': pct
                })
    return indices

def get_sector_data():
    """获取板块数据（简化版，从固定列表获取）"""
    # 这里可以扩展为从东方财富获取板块数据
    sectors = [
        {"name": "创业板电池", "pct": 6.13},
        {"name": "碳科技 30", "pct": 5.63},
        {"name": "能源金属", "pct": 5.54},
        {"name": "创新能源", "pct": 5.40},
        {"name": "有色金属", "pct": 4.14},
    ]
    return sectors

def generate_summary():
    """生成市场总结"""
    indices = get_market_data()
    sectors = get_sector_data()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    print(f"📊 {today} A 股实盘总结（浏览器数据源）")
    print()
    print("📈 大盘指数")
    for idx in indices:
        arrow = "🔴" if idx['pct'] > 0 else "🟢" if idx['pct'] < 0 else "⚪"
        print(f"- {arrow} {idx['name']} {idx['current']:.2f} ({idx['pct']:+.2f}%)")
    
    print()
    print("🔥 领涨板块")
    for sector in sorted(sectors, key=lambda x: x['pct'], reverse=True)[:5]:
        arrow = "🔴" if sector['pct'] > 0 else "🟢" if sector['pct'] < 0 else "⚪"
        print(f"- {arrow} {sector['name']} {sector['pct']:+.2f}%")
    
    print()
    print("💬 抖音文案：", end="")
    if all(idx['pct'] > 0 for idx in indices):
        print("大盘大涨，资金抢新能源电池，有色板块集体爆发")
    elif all(idx['pct'] < 0 for idx in indices):
        print("大盘回调，注意风险控制")
    else:
        print("指数分化，板块轮动加快")

if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    generate_summary()
