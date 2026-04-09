#!/usr/bin/env python3
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

import urllib.request
import urllib.error
import json
import socket
import subprocess
import time
import os
from datetime import datetime


API_URL = "http://localhost:8000/v1/strategy/signals"
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


def format_text_message(signals):
    """
    Format signal data into text message.
    
    Args:
        signals: List of signal dicts
    
    Returns:
        str: Formatted text message
    """
    if not signals:
        return "今日暂无网格信号"
    
    time_str = datetime.now().strftime("%m-%d %H:%M")
    lines = []
    
    # Group by owner
    by_owner = {}
    for sig in signals:
        # Parse owner from fund_code (format: "017193__老婆")
        fund_code_full = sig.get("fund_code", "")
        if "__" in fund_code_full:
            owner = fund_code_full.split("__")[1]
        else:
            owner = "默认"
        
        if owner not in by_owner:
            by_owner[owner] = []
        by_owner[owner].append(sig)
    
    for owner, owner_signals in by_owner.items():
        lines.append(f"📋 {owner} · 网格信号 {time_str}")
        lines.append("-" * 40)
        
        for sig in owner_signals:
            fund_code = sig.get("fund_code", "").split("__")[0]
            fund_name = sig.get("fund_name", "")
            action = sig.get("action", "hold")
            signal_name = sig.get("signal_name", "观望")
            
            if action == "buy":
                amount = sig.get("amount", 0)
                reason = sig.get("reason", "")
                lines.append(f"  [买入] {fund_code} {fund_name}")
                lines.append(f"     建议：买入 {amount:.2f} 元")
                if reason:
                    lines.append(f"     原因：{reason}")
            
            elif action == "sell":
                sell_shares = sig.get("sell_shares", 0)
                sell_pct = sig.get("sell_pct", 0)
                target_batch = sig.get("target_batch_id", "")
                reason = sig.get("reason", "")
                lines.append(f"  [卖出] {fund_code} {fund_name}")
                lines.append(f"     建议：卖出 {sell_shares} 份 (该批次{sell_pct}%)")
                if target_batch:
                    lines.append(f"     批次：{target_batch}")
                if reason:
                    lines.append(f"     原因：{reason}")
            
            else:  # hold
                lines.append(f"  [持有] {fund_code} {fund_name}")
                lines.append(f"     建议：持有等待")
            
            lines.append("")
        
        lines.append("")
    
    return "\n".join(lines)


def export_grid_signals(skip_time_check=False, auto_start_backend=True):
    """
    Main export function.
    
    Fetches grid signals from backend, returns text summary.
    
    Args:
        skip_time_check: If True, skip time window validation (for testing)
        auto_start_backend: If True, automatically start backend if not running
    
    Returns:
        dict: {"success": bool, "count": int, "signals": [...], "message": str}
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
            "message": "今日暂无网格信号"
        }
    
    # Format text message
    message = format_text_message(signals)
    
    return {
        "success": True,
        "count": len(signals),
        "signals": signals,
        "message": message
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
