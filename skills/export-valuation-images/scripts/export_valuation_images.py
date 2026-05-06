#!/usr/bin/env python3
"""
Export Valuation Images Skill

Fetches real-time fund valuation images from local FastAPI backend,
decodes base64 PNG images, and saves them to a temporary directory.

Usage:
    from export_valuation_images import export_valuation_images
    result = export_valuation_images()

Returns:
    {"success": bool, "count": int, "images": [{"sector": str, "filepath": str}]}
    or
    {"success": False, "error": "错误信息"}
"""

import urllib.request
import urllib.error
import json
import base64
import tempfile
import os
import subprocess
import time
import socket
from datetime import datetime, timedelta
import calendar


API_URL = "http://localhost:8000/v1/export/images"
TIMEOUT = 120  # 秒
BACKEND_DIR = r"E:\Git\valuation_grid"
BACKEND_PORT = 8000

# 图片保存到 OpenClaw workspace 下（允许 media 发送）
OUTPUT_BASE_DIR = r"C:\Users\Administrator\.openclaw\workspace-dragapult\valuation_exports"


def is_backend_running():
    """
    Check if the backend service is running on port 8000.
    
    Returns:
        bool: True if service is running, False otherwise
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('127.0.0.1', BACKEND_PORT))
        sock.close()
        return result == 0
    except Exception:
        return False


def start_backend():
    """
    Start the FastAPI backend service in background.
    
    Returns:
        dict: {"success": bool, "error": str or None, "process": Popen or None}
    """
    try:
        # Change to backend directory
        os.chdir(BACKEND_DIR)
        
        # Start uvicorn in background
        process = subprocess.Popen(
            ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )
        
        # Wait for service to be ready (up to 30 seconds)
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
    """
    Ensure backend is running, start it if needed.
    
    Returns:
        dict: {"success": bool, "error": str or None}
    """
    if is_backend_running():
        return {"success": True, "error": None}
    
    print("后端服务未运行，正在自动启动...")
    result = start_backend()
    
    if not result["success"]:
        return {"success": False, "error": result["error"]}
    
    return {"success": True, "error": None}


def is_trading_day(date=None):
    """
    Check if the given date is a trading day (Monday-Friday, not a Chinese holiday).
    For simplicity, we only check weekdays here.
    Add holiday logic if needed.
    
    Args:
        date: datetime.date object, defaults to today
    
    Returns:
        bool: True if trading day, False otherwise
    """
    if date is None:
        date = datetime.now().date()
    
    # Check if weekend
    if date.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    # TODO: Add Chinese holiday calendar check if needed
    # For now, assume all weekdays are trading days
    
    return True


def is_valid_execution_time():
    """
    Check if current time is within valid execution window:
    - Trading day 15:00 - 22:00 (北京时间)
    
    Returns:
        bool: True if should execute, False otherwise
    """
    now = datetime.now()
    
    # Check if trading day
    if not is_trading_day(now.date()):
        return False
    
    # Check time window: 15:00 - 22:00 (Asia/Shanghai)
    hour = now.hour
    
    # Allow execution between 15:00 and 22:59 (inclusive of both 15:00 and 22:00)
    if 15 <= hour <= 22:
        return True
    
    return False


def fetch_images():
    """
    Fetch valuation images from the FastAPI backend.
    
    Returns:
        dict: API response data or error dict
    """
    try:
        req = urllib.request.Request(
            API_URL,
            headers={
                'Accept': 'application/json',
                'User-Agent': 'OpenClaw-Valuation-Export/1.0'
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


def decode_and_save_images(images_data):
    """
    Decode base64 images and save to temporary directory.
    
    Args:
        images_data: List of image dicts from API
    
    Returns:
        list: [{"sector": str, "filepath": str}, ...]
    """
    # Save to workspace directory (allowed for Feishu media upload)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(OUTPUT_BASE_DIR, timestamp)
    os.makedirs(base_dir, exist_ok=True)
    
    saved_images = []
    
    for img in images_data:
        sector = img.get("sector", "未知板块")
        filename = img.get("filename", f"{sector}.png")
        base64_data = img.get("image_base64", "")
        
        if not base64_data:
            continue
        
        try:
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            # Save to file
            filepath = os.path.join(base_dir, filename)
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            saved_images.append({
                "sector": sector,
                "filepath": filepath
            })
        
        except Exception as e:
            # Log error but continue with other images
            print(f"保存 {sector} 图片失败：{str(e)}")
            continue
    
    return saved_images


def export_valuation_images(skip_time_check=False, auto_start_backend=True):
    """
    Main export function.
    
    Fetches valuation images from backend, decodes and saves them,
    returns structured result for agent to send to Feishu.
    
    Args:
        skip_time_check: If True, skip time window validation (for testing)
        auto_start_backend: If True, automatically start backend if not running
    
    Returns:
        dict: {"success": bool, "count": int, "images": [...]}
              or {"success": False, "error": "错误信息"}
    """
    # Check if this is a valid execution time
    if not skip_time_check and not is_valid_execution_time():
        return {
            "success": False,
            "error": "非执行时间：只允许在交易日 15:00 到 22:59 执行"
        }
    
    # Ensure backend is running
    if auto_start_backend:
        backend_status = ensure_backend_running()
        if not backend_status["success"]:
            return backend_status
    
    # Fetch images from API
    fetch_result = fetch_images()
    
    if not fetch_result["success"]:
        return fetch_result
    
    # Check if there's data
    data = fetch_result.get("data", {})
    count = data.get("count", 0)
    images_data = data.get("images", [])
    
    if count == 0 or not images_data:
        return {
            "success": True,
            "count": 0,
            "images": []
        }
    
    # Decode and save images
    saved_images = decode_and_save_images(images_data)
    
    return {
        "success": True,
        "count": len(saved_images),
        "images": saved_images
    }


if __name__ == "__main__":
    import sys
    
    # Check for --test flag to skip time check
    skip_time_check = "--test" in sys.argv
    
    print("=" * 50)
    print("基金估值图片导出工具")
    print("=" * 50)
    print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试模式：{'是' if skip_time_check else '否'}")
    print("-" * 50)
    
    result = export_valuation_images(
        skip_time_check=skip_time_check,
        auto_start_backend=True
    )
    
    print("-" * 50)
    print(f"结果：{json.dumps(result, ensure_ascii=False, indent=2)}")
