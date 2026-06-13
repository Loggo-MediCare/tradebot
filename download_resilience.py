#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载弹性模块 - 提供抗中断和重试功能
用于处理 Hugging Face 模型下载的网络不稳定问题
"""

import os
import signal
import functools
from typing import Any, Callable, TypeVar, Optional
import time

# 设置 Hugging Face 下载环境
def setup_hf_download_env():
    """设置 Hugging Face 下载环境以提高稳定性"""
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    # 增加超时时间（秒）
    os.environ['HF_HUB_TIMEOUT'] = '300'  # 5分钟超时
    # 禁用进度条以减少输出噪音
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '0'
    return


class DownloadInterruptHandler:
    """处理下载过程中的中断"""
    
    def __init__(self, timeout: Optional[int] = None):
        self.timeout = timeout or 600  # 默认10分钟
        self.interrupted = False
    
    def __enter__(self):
        if self.timeout:
            signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.timeout)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # 取消闹钟
        if exc_type is KeyboardInterrupt:
            self.interrupted = True
            return True  # 抑制异常
        return False
    
    def _timeout_handler(self, signum, frame):
        raise TimeoutError(f"Download timeout after {self.timeout} seconds")


def resilient_download(max_retries: int = 3, timeout: int = 600):
    """
    下载弹性装饰器 - 用于包装可能被中断的下载操作
    
    Args:
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    
    Usage:
        @resilient_download(max_retries=3, timeout=300)
        def download_model():
            # 下载代码
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    print(f"   🔄 下载尝试 {attempt + 1}/{max_retries}...")
                    
                    with DownloadInterruptHandler(timeout=timeout) as handler:
                        result = func(*args, **kwargs)
                        if handler.interrupted:
                            print("   ⚠️  下载被中断，正在重试...")
                            time.sleep(2)  # 等待2秒后重试
                            continue
                        return result
                
                except KeyboardInterrupt:
                    print("   ⚠️  收到中断信号...")
                    if attempt < max_retries - 1:
                        print(f"   🔄 重新尝试... ({attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        print("   ❌ 下载最终失败，将跳过此功能")
                        return None
                
                except TimeoutError as e:
                    print(f"   ⏱️  下载超时: {e}")
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"   🔄 重新尝试... ({attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        print("   ❌ 下载最终失败，将跳过此功能")
                        return None
                
                except Exception as e:
                    print(f"   ❌ 下载出错: {e}")
                    last_error = e
                    if attempt < max_retries - 1:
                        print(f"   🔄 重新尝试... ({attempt + 1}/{max_retries})")
                        time.sleep(2)
                        continue
                    else:
                        print("   ❌ 下载最终失败，将跳过此功能")
                        return None
            
            return None
        
        return wrapper
    return decorator


# 初始化环境
setup_hf_download_env()
