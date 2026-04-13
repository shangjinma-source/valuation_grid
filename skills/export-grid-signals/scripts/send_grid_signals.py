#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send Grid Signals to Feishu (by owner)

Executes export_grid_signals and sends each owner's message separately.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_grid_signals import export_grid_signals
import urllib.request
import json

FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")
CHAT_ID = "oc_0a4ded16f2f509c984a6dd9cbd691682"


def send_to_feishu(message_text):
    """Send message to Feishu group chat."""
    if not FEISHU_WEBHOOK_URL:
        print(f"[DRY RUN] Would send to Feishu:\n{message_text}")
        return True
    
    try:
        payload = {
            "msg_type": "text",
            "content": {
                "text": message_text
            }
        }
        
        req = urllib.request.Request(
            FEISHU_WEBHOOK_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            if result.get("code") == 0 or result.get("StatusCode") == 0 or result.get("success"):
                print(f"✅ Sent to Feishu: {message_text[:50]}...")
                return True
            else:
                print(f"❌ Feishu API error: {result}")
                return False
    
    except Exception as e:
        print(f"❌ Failed to send: {e}")
        return False


def main():
    print("=" * 50)
    print("网格信号发送工具（按 owner 分组发送）")
    print("=" * 50)
    
    # Export signals
    result = export_grid_signals(skip_time_check=False, auto_start_backend=True)
    
    if not result.get("success"):
        print(f"❌ 导出失败：{result.get('error', 'Unknown error')}")
        sys.exit(1)
    
    messages = result.get("messages", {})
    total_count = result.get("count", 0)
    
    print(f"✅ 成功导出 {total_count} 条信号")
    print(f"📧 按 {len(messages)} 个 owner 分组发送")
    print("-" * 50)
    
    # Send each owner's message
    sent_count = 0
    for owner, msg_data in messages.items():
        message_text = msg_data.get("message", "")
        count = msg_data.get("count", 0)
        
        print(f"\n📋 {owner}: {count} 条信号")
        
        if send_to_feishu(message_text):
            sent_count += 1
        else:
            print(f"  ❌ 发送失败")
    
    print("-" * 50)
    print(f"✅ 完成：发送 {sent_count}/{len(messages)} 条消息")
    
    return 0 if sent_count == len(messages) else 1


if __name__ == "__main__":
    sys.exit(main())
