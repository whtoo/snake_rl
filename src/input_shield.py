#!/usr/bin/env python3
"""
输入屏蔽模块
用于在训练期间屏蔽标准输入，只允许Ctrl+C中断
"""

import sys
import os
import threading
import signal
import time
from contextlib import contextmanager

class InputShield:
    """
    输入屏蔽类，用于在训练期间屏蔽不必要的输入
    """
    
    def __init__(self):
        self.original_stdin = None
        self.shield_active = False
        self.input_thread = None
        self.stop_thread = False
        
    def _input_consumer(self):
        """
        后台线程，消费并丢弃标准输入
        """
        while not self.stop_thread:
            try:
                if sys.stdin.readable():
                    # 非阻塞读取，丢弃输入
                    if os.name == 'nt':  # Windows
                        import msvcrt
                        while msvcrt.kbhit():
                            msvcrt.getch()  # 消费键盘输入
                    else:  # Unix/Linux/Mac
                        import select
                        if select.select([sys.stdin], [], [], 0)[0]:
                            sys.stdin.read(1)  # 消费一个字符
                time.sleep(0.01)  # 避免CPU占用过高
            except (OSError, IOError):
                # 输入流可能被关闭或不可用
                break
            except Exception:
                # 忽略其他异常，继续运行
                pass
    
    def start_shield(self):
        """
        启动输入屏蔽
        """
        if self.shield_active:
            return
            
        print("\n=== 训练输入屏蔽已启动 ===")
        print("训练期间将忽略键盘和鼠标输入")
        print("按 Ctrl+C 可以安全终止训练")
        print("=" * 35)
        
        self.shield_active = True
        self.stop_thread = False
        
        # 启动输入消费线程
        self.input_thread = threading.Thread(
            target=self._input_consumer, 
            daemon=True
        )
        self.input_thread.start()
        
        # 设置标准输入为非阻塞模式（Unix系统）
        if os.name != 'nt':
            try:
                import fcntl
                flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
                fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
            except (ImportError, OSError):
                pass
    
    def stop_shield(self):
        """
        停止输入屏蔽
        """
        if not self.shield_active:
            return
            
        self.shield_active = False
        self.stop_thread = True
        
        # 等待输入线程结束
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)
        
        # 恢复标准输入为阻塞模式（Unix系统）
        if os.name != 'nt':
            try:
                import fcntl
                flags = fcntl.fcntl(sys.stdin.fileno(), fcntl.F_GETFL)
                fcntl.fcntl(sys.stdin.fileno(), fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
            except (ImportError, OSError):
                pass
        
        print("\n=== 训练输入屏蔽已停止 ===")

# 全局输入屏蔽实例
_input_shield = InputShield()

@contextmanager
def input_shield_context():
    """
    输入屏蔽上下文管理器
    
    使用方法:
    with input_shield_context():
        # 训练代码
        pass
    """
    try:
        _input_shield.start_shield()
        yield
    finally:
        _input_shield.stop_shield()

def start_input_shield():
    """
    启动输入屏蔽（函数接口）
    """
    _input_shield.start_shield()

def stop_input_shield():
    """
    停止输入屏蔽（函数接口）
    """
    _input_shield.stop_shield()

def setup_signal_handlers(stop_callback=None):
    """
    设置信号处理器，只允许Ctrl+C中断
    
    参数:
        stop_callback: 接收到Ctrl+C时的回调函数
    """
    def signal_handler(sig, frame):
        print("\n\n=== 接收到 Ctrl+C 信号 ===")
        print("正在安全终止训练...")
        
        # 停止输入屏蔽
        stop_input_shield()
        
        # 调用用户提供的回调
        if stop_callback:
            stop_callback()
        
        print("训练已安全终止。")
        sys.exit(0)
    
    # 注册Ctrl+C信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    
    # 在Unix系统上，也处理SIGTERM
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("信号处理器已设置，按 Ctrl+C 可安全终止训练")

if __name__ == "__main__":
    # 测试代码
    print("测试输入屏蔽功能...")
    print("请尝试输入任何内容，应该被忽略")
    print("按 Ctrl+C 退出测试")
    
    def test_callback():
        print("测试回调函数被调用")
    
    setup_signal_handlers(test_callback)
    
    with input_shield_context():
        try:
            while True:
                print("训练中... (按 Ctrl+C 退出)")
                time.sleep(2)
        except KeyboardInterrupt:
            print("捕获到 KeyboardInterrupt")