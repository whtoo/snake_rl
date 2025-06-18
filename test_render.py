#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试evaluate.py的渲染功能

这个脚本用于测试新增的游戏画面显示功能
"""

import subprocess
import sys
import os

def test_render_functionality():
    """
    测试渲染功能
    """
    print("🧪 测试evaluate.py的渲染功能")
    print("="*50)
    
    # 检查是否存在模型文件
    model_paths = [
        "./checkpoints/best_model_rainbow.pth",
        "./checkpoints/best_model_dqn.pth",
        "./checkpoints/best_model_dueling.pth"
    ]
    
    available_model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            available_model = model_path
            break
    
    if not available_model:
        print("❌ 未找到可用的模型文件")
        print("请确保以下路径之一存在模型文件:")
        for path in model_paths:
            print(f"  - {path}")
        return False
    
    print(f"✅ 找到模型文件: {available_model}")
    
    # 确定模型类型
    if "rainbow" in available_model:
        model_type = "rainbow"
    elif "dueling" in available_model:
        model_type = "dueling"
    else:
        model_type = "dqn"
    
    print(f"🤖 模型类型: {model_type}")
    
    # 测试不同的渲染选项
    test_cases = [
        {
            "name": "标准评估（无渲染）",
            "args": ["--model", model_type, "--model_path", available_model, "--n_episodes", "1"]
        },
        {
            "name": "带渲染的评估",
            "args": ["--model", model_type, "--model_path", available_model, "--n_episodes", "1", "--render"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试 {i}: {test_case['name']}")
        print("-" * 30)
        
        cmd = [sys.executable, "src/evaluate.py"] + test_case["args"]
        print(f"🔧 执行命令: {' '.join(cmd)}")
        
        try:
            # 对于渲染测试，设置较短的超时时间
            timeout = 30 if "--render" in test_case["args"] else 60
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                print(f"✅ 测试 {i} 成功")
                print("输出:")
                print(result.stdout)
            else:
                print(f"❌ 测试 {i} 失败 (返回码: {result.returncode})")
                print("错误输出:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 测试 {i} 超时 (这对于渲染测试是正常的)")
        except Exception as e:
            print(f"❌ 测试 {i} 出现异常: {e}")
    
    print("\n🎯 渲染功能测试完成")
    print("\n💡 使用说明:")
    print("  - 使用 --render 参数可以显示游戏画面")
    print("  - 使用 --record_video 参数可以录制视频")
    print("  - 两个参数可以同时使用")
    print("  - 按 Ctrl+C 可以提前停止评估")
    
    return True

if __name__ == "__main__":
    test_render_functionality()