#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Grid Signals Skill (Text Only)

Fetches grid signals from local FastAPI backend,
outputs text summary of buy/sell/hold recommendations.

Usage:
    from export_grid_signals import export_grid_signals
    result = export_grid_signals()

Returns:
    {"success": bool, "count": int, "signals": [{"owner": str, "fund_code": str, "action": str, "amount/shares": float, "reason": str}]}
"""

import sys
import urllib.request
import urllib.error
import json
import socket
import subprocess
import time
import os
from datetime import datetime

# Fix Windows console encoding - force UTF-8 output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


API_URL = "http://localhost:8000/v1/strategy/signals"
POSITIONS_FILE = r"E:\Git\valuation_grid\data\positions.json"
TIMEOUT = 180
BACKEND_DIR = r"E:\Git\valuation_grid"
BACKEND_PORT = 8000


def is_backend_running():
    """Check if the backend service is running on port 8000."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', BACKEND_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_backend():
    """Start the FastAPI backend service in background."""
    try:
        os.chdir(BACKEND_DIR)
        process = subprocess.Popen(
            ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        print(f"正在启动后端服务 (端口 {BACKEND_PORT})...")
        for i in range(30):
            time.sleep(1)
            if is_backend_running():
                print(f"后端服务已启动 (耗时 {i+1} 秒)")
                return {"success": True, "process": process}
        
        return {"success": False, "error": "后端服务启动超时", "process": None}
    
    except FileNotFoundError:
        return {"success": False, "error": "未找到 uvicorn，请先安装：pip install uvicorn"}
    except Exception as e:
        return {"success": False, "error": f"启动失败：{str(e)}", "process": None}


def ensure_backend_running():
    """Ensure backend is running, start it if needed."""
    if is_backend_running():
        return {"success": True, "error": None}
    
    print("后端服务未运行，正在自动启动...")
    result = start_backend()
    
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    
    return {"success": True, "error": None}


def fetch_grid_signals():
    """Fetch grid signals from the FastAPI backend."""
    try:
        req = urllib.request.Request(
            API_URL,
            headers={
                'Accept': 'application/json',
                'User-Agent': 'OpenClaw-Grid-Signals/1.0'
            }
        )
        
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8'))
            return {"success": True, "data": data}
    
    except urllib.error.URLError as e:
        return {"success": False, "error": f"网络错误：{str(e.reason)}"}
    except urllib.error.HTTPError as e:
        return {"success": False, "error": f"HTTP 错误 {e.code}: {e.reason}"}
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"JSON 解析错误：{str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"未知错误：{str(e)}"}


def load_positions_data():
    """Load positions data from local file."""
    try:
        with open(POSITIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"funds": {}}


def get_fund_holding(fund_code_full, positions_data):
    """Get holding value and profit for a fund."""
    funds = positions_data.get("funds", {})
    fund_data = funds.get(fund_code_full, {})
    
    # Calculate total holding value and profit
    batches = fund_data.get("batches", [])
    total_value = 0
    total_profit = 0
    
    for batch in batches:
        if batch.get("status") == "holding":
            shares = batch.get("shares", 0)
            nav = batch.get("latest_nav", 0) or batch.get("nav", 0)
            cost_nav = batch.get("nav", 0)
            value = shares * nav
            profit = shares * (nav - cost_nav)
            total_value += value
            total_profit += profit
    
    return total_value, total_profit


def load_state_data():
    """Load state data from local file to get sector mapping."""
    state_file = r"E:\Git\valuation_grid\data\state.json"
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {"sectors": []}


def build_code_to_sector_mapping(state_data):
    """Build a mapping from fund code to sector name."""
    code_to_sector = {}
    sectors = state_data.get("sectors", [])
    
    for sector in sectors:
        sector_name = sector.get("name", "")
        for fund in sector.get("funds", []):
            code = fund.get("code", "")
            if code:
                code_to_sector[code] = sector_name
    
    return code_to_sector


def format_messages_by_owner(signals):
    """
    Format signal data into separate messages by owner.
    Returns dict: {owner: message_text}
    
    Args:
        signals: List of signal dicts
    
    Returns:
        dict: {owner: str, message: str, count: int}
    """
    if not signals:
        return {"default": {"owner": "default", "message": "今日暂无网格信号", "count": 0}}
    
    time_str = datetime.now().strftime("%m-%d %H:%M")
    
    # Load state data to get fund names
    state_data = load_state_data()
    code_to_name = {}
    for sector in state_data.get("sectors", []):
        for fund in sector.get("funds", []):
            code = fund.get("code", "")
            alias = fund.get("alias", "")
            if code:
                code_to_name[code] = alias
    
    # Group by owner (from fund_code format: "017193__老婆")
    by_owner = {}
    for sig in signals:
        fund_code_full = sig.get("fund_code", "")
        if "__" in fund_code_full:
            owner = fund_code_full.split("__")[1]
        else:
            owner = "默认"
        
        if owner not in by_owner:
            by_owner[owner] = []
        by_owner[owner].append(sig)
    
    # Format each owner's message
    messages = {}
    for owner, owner_signals in by_owner.items():
        lines = []
        lines.append(f"📋 {owner} · 网格信号 {time_str}")
        lines.append("-" * 40)
        
        for sig in owner_signals:
            fund_code_full = sig.get("fund_code", "")
            fund_code = fund_code_full.split("__")[0] if "__" in fund_code_full else fund_code_full
            fund_name = code_to_name.get(fund_code, "")
            action = sig.get("action", "hold")
            
            # Display format: "022084 华安中证有色金属矿业主题指数 C"
            display_text = f"{fund_code} {fund_name}" if fund_name else fund_code
            
            if action == "buy":
                amount = sig.get("amount", 0) or 0
                reason = sig.get("reason", "")
                lines.append(f"  [买入] {display_text}")
                lines.append(f"     建议：买入 {amount:.2f} 元")
                if reason:
                    lines.append(f"     原因：{reason}")
            
            elif action == "sell":
                sell_shares = sig.get("sell_shares", 0) or 0
                sell_pct = sig.get("sell_pct", 0) or 0
                target_batch = sig.get("target_batch_id", "")
                reason = sig.get("reason", "")
                lines.append(f"  [卖出] {display_text}")
                lines.append(f"     建议：卖出 {sell_shares} 份 (该批次{sell_pct}%)")
                if target_batch:
                    lines.append(f"     批次：{target_batch}")
                if reason:
                    lines.append(f"     原因：{reason}")
            
            else:  # hold
                lines.append(f"  [持有] {display_text}")
                lines.append(f"     建议：持有等待")
            
            lines.append("")
        
        messages[owner] = {
            "owner": owner,
            "message": "\n".join(lines),
            "count": len(owner_signals)
        }
    
    return messages


def export_grid_signals(skip_time_check=False, auto_start_backend=True):
    """
    Main export function.
    
    Fetches grid signals from backend, returns messages by owner.
    
    Args:
        skip_time_check: If True, skip time window validation (for testing)
        auto_start_backend: If True, automatically start backend if not running
    
    Returns:
        dict: {"success": bool, "count": int, "signals": [...], "messages": {owner: {owner, message, count}}}
    """
    # Ensure backend is running
    if auto_start_backend:
        backend_status = ensure_backend_running()
        if not backend_status["success"]:
            return backend_status
    
    # Fetch signals from API
    fetch_result = fetch_grid_signals()
    
    if not fetch_result["success"]:
        return fetch_result
    
    # Check if there's data
    data = fetch_result.get("data", {})
    signals = data.get("signals", [])
    
    if not signals:
        return {
            "success": True,
            "count": 0,
            "signals": [],
            "messages": {"default": {"owner": "default", "message": "今日暂无网格信号", "count": 0}}
        }
    
    # Format messages by owner
    messages = format_messages_by_owner(signals)
    
    return {
        "success": True,
        "count": len(signals),
        "signals": signals,
        "messages": messages
    }


if __name__ == "__main__":
    import sys
    
    skip_time_check = "--test" in sys.argv
    
    print("=" * 50)
    print("网格信号导出工具（文字版）")
    print("=" * 50)
    print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模式：{'是' if skip_time_check else '否'}")
    print("-" * 50)
    
    result = export_grid_signals(
        skip_time_check=skip_time_check,
        auto_start_backend=True
    )
    
    print("-" * 50)
    print(f"成功：{result.get('success')}")
    print(f"数量：{result.get('count')}")
    print("-" * 50)
    
    # Handle Unicode encoding on Windows console
    message = result.get('message', '')
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: encode to GBK with ignore
        print(message.encode('gbk', 'ignore').decode('gbk'))
