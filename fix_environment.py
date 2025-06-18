#!/usr/bin/env python3
"""
环境修复脚本
用于解决OpenMP、Qt和pybind11相关的错误，同时优化性能
"""

import os
import sys
import warnings
import multiprocessing

def get_optimal_thread_count():
    """
    获取最优的线程数设置
    
    返回:
        int: 推荐的线程数
    """
    try:
        # 获取逻辑CPU核心数
        logical_cores = multiprocessing.cpu_count()
        
        # 对于机器学习任务，通常使用物理核心数或逻辑核心数的一半到全部
        # 这里我们使用逻辑核心数，但限制最大值避免过度并行化
        if logical_cores <= 4:
            return logical_cores
        elif logical_cores <= 8:
            return logical_cores - 1  # 保留一个核心给系统
        else:
            return min(logical_cores - 2, 32)  # 保留2个核心，最大16线程
            
    except Exception:
        # 如果无法获取核心数，返回保守值
        return 4

def setup_environment(force_single_thread=False, custom_thread_count=None):
    """
    设置环境变量以解决常见的库冲突问题，同时优化性能
    
    参数:
        force_single_thread (bool): 是否强制使用单线程（最安全但性能最低）
        custom_thread_count (int): 自定义线程数
    """
    # 解决OpenMP冲突问题
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # 设置线程数
    if force_single_thread:
        thread_count = 1
        print("使用单线程模式（最安全，但性能较低）")
    elif custom_thread_count:
        thread_count = custom_thread_count
        print(f"使用自定义线程数: {thread_count}")
    else:
        thread_count = get_optimal_thread_count()
        print(f"使用优化的线程数: {thread_count}")
    
    os.environ['OMP_NUM_THREADS'] = str(thread_count)
    
    # 设置其他OpenMP相关环境变量以提高性能
    os.environ['MKL_NUM_THREADS'] = str(thread_count)
    os.environ['NUMEXPR_NUM_THREADS'] = str(thread_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(thread_count)
    
    # 设置Qt相关环境变量
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
    
    # 设置PyTorch相关环境变量
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # 禁用一些警告
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    print("环境变量已设置完成:")
    print(f"  KMP_DUPLICATE_LIB_OK = {os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print(f"  OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"  MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
    print(f"  系统逻辑核心数 = {multiprocessing.cpu_count()}")

def check_dependencies():
    """
    检查关键依赖是否正确安装
    """
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        
        # 检查PyTorch线程设置
        torch_threads = torch.get_num_threads()
        print(f"PyTorch线程数: {torch_threads}")
        
        import gymnasium
        print(f"Gymnasium版本: {gymnasium.__version__}")
        
        # torch-directml已从项目中移除
        print("注意: 项目已移除DirectML支持，使用CUDA或CPU进行训练")
        
        return True
    except ImportError as e:
        print(f"依赖检查失败: {e}")
        return False

def benchmark_performance():
    """
    简单的性能基准测试
    """
    try:
        import torch
        import time
        
        print("\n执行性能基准测试...")
        
        # 创建测试张量
        size = 1000
        a = torch.randn(size, size)
        b = torch.randn(size, size)
        
        # 预热
        for _ in range(3):
            _ = torch.mm(a, b)
        
        # 测试矩阵乘法性能
        start_time = time.time()
        for _ in range(10):
            result = torch.mm(a, b)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        print(f"矩阵乘法平均耗时: {avg_time:.4f}秒")
        
        if avg_time < 0.1:
            print("性能: 优秀")
        elif avg_time < 0.5:
            print("性能: 良好")
        else:
            print("性能: 需要优化")
            
    except Exception as e:
        print(f"性能测试失败: {e}")

def main():
    """
    主函数
    """
    print("=== 智能环境修复脚本 ===")
    
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='智能环境修复脚本')
    parser.add_argument('--single-thread', action='store_true', 
                       help='强制使用单线程（最安全但性能最低）')
    parser.add_argument('--threads', type=int, 
                       help='自定义线程数')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    
    args = parser.parse_args()
    
    # 设置环境变量
    setup_environment(
        force_single_thread=args.single_thread,
        custom_thread_count=args.threads
    )
    
    # 检查依赖
    if check_dependencies():
        print("\n依赖检查通过。")
        
        # 运行性能测试
        if args.benchmark:
            benchmark_performance()
        
        print("\n使用方法:")
        print("1. 运行此脚本设置环境")
        print("2. 然后运行: python train.py [参数]")
        print("\n或者直接运行: python run_training.py")
        print("\n性能优化选项:")
        print("  --single-thread    : 单线程模式（最安全）")
        print("  --threads N        : 使用N个线程")
        print("  --benchmark        : 运行性能测试")
    else:
        print("\n依赖检查失败，请检查安装。")
        sys.exit(1)

if __name__ == "__main__":
    main()