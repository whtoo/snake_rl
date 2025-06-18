#!/usr/bin/env python3
"""
测试输入屏蔽功能
"""

import time
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.input_shield import input_shield_context, setup_signal_handlers

def test_input_shield():
    """
    测试输入屏蔽功能
    """
    print("=== 输入屏蔽功能测试 ===")
    print("测试说明:")
    print("1. 程序将启动输入屏蔽")
    print("2. 在测试期间，请尝试输入任何内容（应该被忽略）")
    print("3. 按 Ctrl+C 可以安全终止测试")
    print("4. 测试将持续30秒")
    print()
    
    stop_requested = False
    
    def stop_callback():
        nonlocal stop_requested
        print("\n收到停止信号，准备退出测试...")
        stop_requested = True
    
    # 设置信号处理器
    setup_signal_handlers(stop_callback)
    
    print("开始测试...")
    
    # 使用输入屏蔽上下文
    with input_shield_context():
        start_time = time.time()
        test_duration = 30  # 30秒测试
        
        while not stop_requested:
            elapsed = time.time() - start_time
            remaining = test_duration - elapsed
            
            if remaining <= 0:
                print("\n测试时间结束！")
                break
            
            print(f"\r测试进行中... 剩余时间: {remaining:.1f}秒 (按 Ctrl+C 提前退出)", end="", flush=True)
            time.sleep(0.5)
    
    print("\n\n=== 测试完成 ===")
    print("如果在测试期间您的键盘输入被忽略，说明输入屏蔽功能正常工作！")
    print("如果您能够通过 Ctrl+C 安全退出，说明信号处理正常工作！")

if __name__ == "__main__":
    test_input_shield()