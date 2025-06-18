#!/usr/bin/env python3
"""简化的测试训练脚本，验证优化效果"""
import os, sys
sys.path.append('src')

import numpy as np, pandas as pd
from loguru import logger
from data_fetch import load_filtered_df
from preprocess import rank_gauss, pca_reduce

def test_pipeline():
    # 配置日志
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/test.log", rotation="50 MB", level="INFO")
    logger.info("开始测试流程")
    
    # 测试数据加载
    logger.info("测试数据加载...")
    df = load_filtered_df()
    logger.info(f"数据加载成功: {df.shape}")
    
    # 测试特征预处理
    logger.info("测试特征预处理...")
    feature_cols = [c for c in df.columns if c.startswith("feature_")][:50]  # 只取前50个特征进行测试
    df_feat = df[feature_cols].head(1000)  # 只取前1000行进行测试
    
    logger.info(f"测试数据: {df_feat.shape}")
    
    # 测试 rank_gauss
    logger.info("测试 rank_gauss 优化...")
    import time
    start = time.time()
    gauss_feat = rank_gauss(df_feat)
    rank_time = time.time() - start
    logger.info(f"rank_gauss 完成，耗时: {rank_time:.2f}秒")
    
    # 测试 PCA
    logger.info("测试 PCA 降维...")
    start = time.time()
    pca_feat = pca_reduce(gauss_feat)
    pca_time = time.time() - start
    logger.info(f"PCA 完成，耗时: {pca_time:.2f}秒")
    
    logger.info(f"所有测试完成！总体形状: {pca_feat.shape}")
    print(f"✅ 测试成功！数据形状: {pca_feat.shape}")
    print(f"⚡ rank_gauss 耗时: {rank_time:.2f}秒")
    print(f"⚡ PCA 耗时: {pca_time:.2f}秒")

if __name__ == "__main__":
    test_pipeline()