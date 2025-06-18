import glob, joblib, numpy as np, pandas as pd, os, time, json
from loguru import logger
from preprocess import rank_gauss, pca_reduce
from data_fetch import get_feature_set
import numerapi

# 全局模型缓存
_model_cache = {}

def download_live_data(feature_set="medium", outfile="data/live.parquet"):
    """下載最新的 live 數據"""
    napi = numerapi.NumerAPI()
    
    # 下載 live 數據
    print("正在下載 live 數據...")
    napi.download_dataset("v5.0/live.parquet", outfile)
    
    # 確保 features.json 存在
    features_file = "data/features.json"
    if not os.path.exists(features_file):
        print("正在下載特徵元數據...")
        napi.download_dataset("v5.0/features.json", features_file)
    
    print("Live 數據下載完成！")

def predict_live(live_path="data/live.parquet", feature_set=None):
    """
    對 live 數據進行預測
    live_path: live 數據文件路径
    feature_set: 特徵集 ("small", "medium", "all")，如果為 None 則從模型目錄讀取
    """
    # 配置日志
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/predict.log", rotation="50 MB", level="INFO")
    logger.info("開始預測流程")
    
    # 如果沒有指定特徵集，嘗試從訓練時保存的資訊讀取
    if feature_set is None:
        feature_set_file = "models/feature_set.txt"
        if os.path.exists(feature_set_file):
            with open(feature_set_file, 'r') as f:
                feature_set = f.readline().strip()
            logger.info(f"從模型目錄讀取特徵集: {feature_set}")
        else:
            feature_set = "medium"  # 預設值
            logger.info(f"使用預設特徵集: {feature_set}")
    
    # 獲取特徵列表
    features = get_feature_set(feature_set)
    if features is None:
        logger.error("無法獲取特徵集，停止預測")
        return
    
    # 載入 live 數據，只載入需要的特徵
    logger.info(f"正在載入 live 數據...")
    try:
        live_df = pd.read_parquet(live_path, columns=features)
        logger.info(f"成功載入 live 數據，特徵數量: {len(features)}, 預測行數: {len(live_df)}")
    except Exception as e:
        logger.error(f"載入 live 數據失敗: {e}")
        return
    
    feature_cols = [c for c in live_df.columns if c.startswith("feature_")]
    
    start_time = time.time()
    # 使用共享的 PCA 模型（如果存在）
    if os.path.exists("models/pca_model.pkl"):
        logger.info("使用共享的 PCA 模型")
        pca = joblib.load("models/pca_model.pkl")
        feat_gauss = rank_gauss(live_df[feature_cols])
        feat = pca.transform(feat_gauss).astype(np.float32)
    else:
        logger.info("PCA 模型不存在，使用 fallback 方法")
        feat = pca_reduce(rank_gauss(live_df[feature_cols]))
    
    preprocess_time = time.time() - start_time
    logger.info(f"特徵預處理耗時: {preprocess_time:.2f}秒")
    
    model_files = glob.glob("models/lgbm_seed*.pkl")
    logger.info(f"找到 {len(model_files)} 個模型文件")
    
    if len(model_files) == 0:
        logger.error("未找到模型文件，請先執行訓練")
        return
    
    preds = np.zeros(len(live_df), dtype=np.float32)
    for i, pkl in enumerate(model_files):
        model_start = time.time()
        
        # 使用模型缓存
        if pkl not in _model_cache:
            logger.info(f"載入模型到緩存: {os.path.basename(pkl)}")
            _model_cache[pkl] = joblib.load(pkl)
        else:
            logger.info(f"使用緩存模型: {os.path.basename(pkl)}")
        
        model = _model_cache[pkl]
        preds += model.predict(feat) / len(model_files)
        model_time = time.time() - model_start
        logger.info(f"模型 {i+1}/{len(model_files)} ({os.path.basename(pkl)}) 預測耗時: {model_time:.2f}秒")
    
    # 創建預測結果 DataFrame
    submission = pd.DataFrame({
        "prediction": preds
    }, index=live_df.index)
    
    submission.to_csv("predictions.csv")
    
    total_time = time.time() - start_time
    logger.info(f"預測流程完成，總耗時: {total_time:.2f}秒")
    logger.info(f"預測結果已保存到 predictions.csv")
    
    return submission

if __name__ == "__main__":
    import sys
    
    # 檢查是否需要下載 live 數據
    if len(sys.argv) > 1 and sys.argv[1] == "download":
        feature_set = sys.argv[2] if len(sys.argv) > 2 else "medium"
        download_live_data(feature_set)
    else:
        # 執行預測
        feature_set = sys.argv[1] if len(sys.argv) > 1 else None
        predict_live(feature_set=feature_set)