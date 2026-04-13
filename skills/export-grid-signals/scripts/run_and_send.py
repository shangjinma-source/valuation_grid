#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run export_grid_signals and send to Feishu via OpenClaw message tool.
Each owner gets a separate message.
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_grid_signals import export_grid_signals

CHAT_ID = "chat:oc_0a4ded16f2f509c984a6dd9cbd691682"


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
    print(f"📧 按 {len(messages)} 个 owner 分组")
    print("-" * 50)
    
    # Output message data for OpenClaw to process
    for owner, msg_data in messages.items():
        message_text = msg_data.get("message", "")
        count = msg_data.get("count", 0)
        
        print(f"\n📋 {owner}: {count} 条信号")
        print(f"---MESSAGE_START---")
        print(message_text)
        print(f"---MESSAGE_END---")
    
    print("-" * 50)
    print(f"✅ 完成：{len(messages)} 个 owner 的消息已准备发送")
    
    # Return structured data for OpenClaw to send
    return {
        "success": True,
        "count": total_count,
        "messages": messages,
        "chat_id": CHAT_ID
    }


if __name__ == "__main__":
    result = main()
    # Print JSON for OpenClaw to parse
    import json
    print("\n===JSON_RESULT===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
