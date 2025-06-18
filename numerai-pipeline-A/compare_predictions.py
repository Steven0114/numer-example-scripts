"""
比較原始25個模型集成方法與最終模型的預測結果
確保兩種方法產生一致的結果
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import cloudpickle
from loguru import logger
from scipy import stats

# 添加src目錄到Python路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import rank_gauss
from neutralize import neutralize


def neutralize_single_era(predictions, features, neutralizers, proportion=0.85):
    """
    對單一era進行neutralization（不依賴era分組）
    這與create_final_official_model.py中的方法一致
    """
    scores = predictions.rank(method="first").values
    scores = stats.norm.ppf((scores - 0.5) / scores.shape[0])
    exposures = features[neutralizers].fillna(features[neutralizers].median()).values
    scores = scores - proportion * exposures @ np.linalg.pinv(exposures, rcond=1e-6) @ scores
    scores = scores / np.std(scores, axis=0, ddof=0)
    return pd.Series(scores.flatten(), index=predictions.index)


def load_original_models():
    """載入原始的25個LightGBM模型和PCA模型"""
    models_dir = "models"
    
    # 載入PCA模型
    pca_model = joblib.load(os.path.join(models_dir, "pca_model.pkl"))
    
    # 載入所有LightGBM模型
    import glob
    model_files = glob.glob(os.path.join(models_dir, "lgbm_seed*.pkl"))
    lgb_models = []
    for model_file in sorted(model_files):
        model = joblib.load(model_file)
        lgb_models.append(model)
    
    return pca_model, lgb_models


def predict_with_original_method(live_features, feature_cols):
    """使用原始的25個模型集成方法進行預測"""
    
    # 載入模型
    pca_model, lgb_models = load_original_models()
    
    # 1. Rank-Gaussian變換
    feat_gauss = rank_gauss(live_features[feature_cols])
    
    # 2. PCA降維
    feat = pca_model.transform(feat_gauss).astype(np.float32)
    
    # 3. 模型集成預測
    preds = np.zeros(len(live_features), dtype=np.float32)
    for model in lgb_models:
        preds += model.predict(feat) / len(lgb_models)
    
    # 4. 轉換為pandas Series進行neutralization
    pred_series = pd.Series(preds, index=live_features.index)
    
    # 5. Neutralization（與create_final_official_model.py一致：前30個特徵，proportion=0.85）
    neutral_cols = feature_cols[:30]
    neutralized_predictions = neutralize_single_era(
        pred_series, 
        live_features, 
        neutral_cols, 
        proportion=0.85
    )
    
    return neutralized_predictions.values


def predict_with_final_model(live_features, model_file="numerai_final_model_v3.pkl"):
    """使用最終模型進行預測"""
    
    # 載入最終模型
    with open(model_file, "rb") as f:
        predict_func = cloudpickle.loads(f.read())
    
    # 創建空的benchmark_models（官方格式要求）
    live_benchmark_models = pd.DataFrame(index=live_features.index)
    
    # 預測
    predictions = predict_func(live_features, live_benchmark_models)
    
    return predictions["prediction"].values


def compare_predictions(n_samples=1000, feature_set="medium"):
    """比較兩種預測方法的結果"""
    
    logger.info(f"比較兩種預測方法，樣本數: {n_samples}, 特徵集: {feature_set}")
    
    # 載入特徵元數據
    with open("data/features.json", 'r') as f:
        feature_metadata = json.load(f)
    features = feature_metadata["feature_sets"][feature_set]
    feature_cols = [c for c in features if c.startswith("feature_")]
    
    # 創建測試數據
    np.random.seed(42)  # 確保可重現性
    test_live_features = pd.DataFrame(
        np.random.randn(n_samples, len(features)),
        columns=features,
        index=range(n_samples)
    )
    
    logger.info("使用原始25個模型集成方法進行預測...")
    original_predictions = predict_with_original_method(test_live_features, feature_cols)
    
    logger.info("使用最終模型進行預測...")
    final_predictions = predict_with_final_model(test_live_features)
    
    # 比較結果
    logger.info("比較預測結果...")
    
    # 計算差異統計
    diff = original_predictions - final_predictions
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_diff = np.max(np.abs(diff))
    correlation = np.corrcoef(original_predictions, final_predictions)[0, 1]
    
    logger.info("="*50)
    logger.info("預測結果比較:")
    logger.info(f"  平均絕對誤差 (MAE): {mae:.8f}")
    logger.info(f"  均方誤差 (MSE): {mse:.8f}")
    logger.info(f"  均方根誤差 (RMSE): {rmse:.8f}")
    logger.info(f"  最大絕對差異: {max_diff:.8f}")
    logger.info(f"  相關係數: {correlation:.8f}")
    logger.info("="*50)
    
    # 統計摘要
    logger.info("原始方法預測統計:")
    logger.info(f"  平均值: {np.mean(original_predictions):.6f}")
    logger.info(f"  標準差: {np.std(original_predictions):.6f}")
    logger.info(f"  最小值: {np.min(original_predictions):.6f}")
    logger.info(f"  最大值: {np.max(original_predictions):.6f}")
    
    logger.info("最終模型預測統計:")
    logger.info(f"  平均值: {np.mean(final_predictions):.6f}")
    logger.info(f"  標準差: {np.std(final_predictions):.6f}")
    logger.info(f"  最小值: {np.min(final_predictions):.6f}")
    logger.info(f"  最大值: {np.max(final_predictions):.6f}")
    
    # 判斷是否一致
    if mae < 1e-6 and correlation > 0.999:
        logger.info("✅ 兩種方法的預測結果高度一致！")
        return True
    elif mae < 1e-4 and correlation > 0.99:
        logger.info("⚠️ 兩種方法的預測結果基本一致，存在微小差異")
        return True
    else:
        logger.error("❌ 兩種方法的預測結果存在明顯差異！")
        return False


if __name__ == "__main__":
    import sys
    
    # 設置日誌
    logger.add("logs/compare_predictions.log", rotation="50 MB", level="INFO")
    
    # 創建日誌目錄
    os.makedirs("logs", exist_ok=True)
    
    # 檢查參數
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    feature_set = sys.argv[2] if len(sys.argv) > 2 else "medium"
    
    try:
        success = compare_predictions(n_samples, feature_set)
        if not success:
            sys.exit(1)
        logger.info("🎉 預測方法一致性驗證完成！")
    except Exception as e:
        logger.error(f"比較失敗: {e}")
        sys.exit(1)