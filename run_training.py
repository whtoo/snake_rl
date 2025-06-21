#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
强化学习训练启动脚本
使用方法: python run_training.py [模型类型] [回合数]
"""

import os
import subprocess
import sys
import asyncio

# 导入环境修复模块
try:
    from fix_environment import setup_environment, check_dependencies
except ImportError:
    print("警告: 无法导入环境修复模块")
    def setup_environment():
        pass
    def check_dependencies():
        return True

def check_conda_env(env_name):
    """检查 conda 环境是否存在"""
    try:
        result = subprocess.run(['conda', 'info', '--envs'],
                              capture_output=True,
                              text=True,
                              check=True)
        return env_name in result.stdout
    except subprocess.CalledProcessError:
        print("错误: 无法检查 conda 环境")
        return False


def create_directories():
    """创建必要的目录"""
    directories = ['checkpoints', 'logs', 'videos']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"确保目录存在: {directory}")

async def _stream_output(stream, prefix):
    """异步读取并打印流的内容"""
    while True:
        line = await stream.readline()
        if line:
            # 使用 sys.stdout.write 和 flush 来确保实时性，尤其是在某些缓冲环境下
            sys.stdout.write(f"{prefix}{line.decode(errors='replace').strip()}\n")
            sys.stdout.flush()
        else:
            break

async def run_training(model_type, episodes):
    """异步运行训练并实时显示日志"""
    # 使用模块方式运行以避免相对导入错误
    cmd_list = [
        'conda', 'run', '-n', 'snake_rl', 'python', '-m', 'src.train',
        '--model', model_type,
        '--episodes', str(episodes),
        '--save_dir', 'checkpoints',
        '--log_dir', 'logs',
        '--save_interval', '100',
        '--eval_interval', '50'
    ]

    if sys.platform == "win32":
        cmd_str = subprocess.list2cmdline(cmd_list)
    else:
        cmd_str = ' '.join(cmd_list) # 注意：如果参数包含空格或特殊shell字符，这可能需要更复杂的引用

    print(f"执行异步命令: {cmd_str}")

    process = await asyncio.create_subprocess_shell(
        cmd_str,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    print("训练日志:")
    stdout_task = asyncio.create_task(_stream_output(process.stdout, "LOG: "))
    stderr_task = asyncio.create_task(_stream_output(process.stderr, "ERR: "))

    await asyncio.gather(stdout_task, stderr_task)
    return_code = await process.wait()

    if return_code == 0:
        print("\n训练过程成功完成。")
        return True
    else:
        print(f"\n训练过程中发生错误，返回码: {return_code}")
        return False

async def main():
    # 默认参数
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'dqn'
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    print("======================================")
    print("强化学习 Atari 游戏训练")
    print("======================================")
    print(f"模型类型: {model_type}")
    print(f"训练回合: {episodes}")
    print("游戏环境: ALE/Assault-v5") 
    print("======================================")
    
    # 设置环境变量以解决常见问题
    print("设置环境变量...")
    setup_environment()
    
    # 检查依赖
    print("检查依赖...")
    if not check_dependencies():
        print("依赖检查失败，请检查安装")
        sys.exit(1)

    # 检查 conda 环境
    env_name = 'snake_rl'
    if not check_conda_env(env_name):
        print(f"错误: 未找到 {env_name} conda 环境")
        print("请先运行: conda env create -f environment.yml")
        sys.exit(1)

    # 创建必要的目录
    print("创建输出目录...")
    create_directories()
    
    print("验证目录创建:")
    for directory in ['checkpoints', 'logs', 'videos']:
        if os.path.exists(directory):
            print(f"  {directory}: 存在")
        else:
            print(f"  {directory}: 不存在")

    # 开始训练
    print("准备异步训练命令...")
    if await run_training(model_type, episodes):
        print("======================================")
        print("训练完成！")
        print("模型保存在: checkpoints/")
        print("日志保存在: logs/")
        print("======================================")
        print("使用以下命令评估模型:")
        print(f"python src\\evaluate.py --model_path checkpoints\\best_model_{model_type}.pth")
        print("======================================")
    else:
        print("训练未完成")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())