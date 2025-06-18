#!/usr/bin/env python3
"""
数据下载脚本
使用方法:
    python download_data.py             # 下载训练数据和特征元数据
    python download_data.py live        # 下载live数据
    python download_data.py all         # 下载所有数据
"""

import os
import sys
sys.path.append('src')

from data_fetch import download_latest_parquet
from predict import download_live_data

def main():
    os.makedirs("data", exist_ok=True)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "live":
            print("下载 live 数据...")
            download_live_data()
        elif command == "all":
            print("下载训练数据...")
            download_latest_parquet()
            print("下载 live 数据...")
            download_live_data()
        else:
            print(f"未知命令: {command}")
            print("请使用: python download_data.py [live|all]")
    else:
        print("下载训练数据和特征元数据...")
        download_latest_parquet()

if __name__ == "__main__":
    main()