import os, joblib, gc, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
import lightgbm as lgb
from rich.progress import track
from loguru import logger
from config import SEEDS, N_SPLITS, LIGHTGBM_PARAMS, NUM_FEATURES
from data_fetch import load_filtered_df
from preprocess import rank_gauss, pca_reduce
from neutralize import neutralize


def train(feature_set="medium"):
    """
    訓練模型
    feature_set: "small", "medium", "all" - 選擇要使用的特徵集
    """
    # 配置日志
    os.makedirs("logs", exist_ok=True)
    logger.add("logs/train.log", rotation="50 MB", level="INFO")
    logger.info(f"開始訓練流程，使用 '{feature_set}' 特徵集")
    
    logger.info("正在加載和預處理數據...")
    df = load_filtered_df(feature_set=feature_set)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    logger.info(f"特徵數量: {len(feature_cols)}, 數據行數: {len(df)}")
    
    df_feat = rank_gauss(df[feature_cols])
    pca_feat = pca_reduce(df_feat)
    target = df["target"].astype(np.float32)
    groups = df["era"].values
    logger.info("數據預處理完成")

    oof_preds = np.zeros(len(df), dtype=np.float32)
    pca_saved = False
    total_models = len(SEEDS) * N_SPLITS
    
    model_count = 0
    for seed_idx, seed in enumerate(SEEDS):
        logger.info(f"開始訓練 Seed {seed} ({seed_idx+1}/{len(SEEDS)})")
        lgb_params = LIGHTGBM_PARAMS | {"random_state": seed}
        kf = GroupKFold(n_splits=N_SPLITS)
        
        for fold, (tr_idx, val_idx) in enumerate(kf.split(pca_feat, target, groups)):
            model_count += 1
            logger.info(f"訓練模型 {model_count}/{total_models}: seed={seed}, fold={fold}")
            
            X_tr, y_tr = pca_feat[tr_idx], target.iloc[tr_idx]
            X_val, y_val = pca_feat[val_idx], target.iloc[val_idx]
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      eval_metric="l1", callbacks=[lgb.early_stopping(50)])
            
            val_score = model.best_score_['valid_0']['l1']
            logger.info(f"模型 seed{seed}_f{fold} 驗證 L1: {val_score:.6f}")
            
            joblib.dump(model, f"models/lgbm_seed{seed}_f{fold}.pkl")
            oof_preds[val_idx] += model.predict(X_val) / len(SEEDS)
            
            # 保存 PCA 模型（只在第一個 seed 第一個 fold 時保存）
            if not pca_saved and seed == SEEDS[0] and fold == 0:
                pca = PCA(n_components=NUM_FEATURES, random_state=0)
                pca.fit(rank_gauss(df[feature_cols]))
                joblib.dump(pca, "models/pca_model.pkl", compress=3)
                pca_saved = True
                logger.info("PCA 模型已保存")
        logger.info(f"Seed {seed} 完成")
        gc.collect()

    logger.info("訓練完成，開始後處理...")
    df["pred"] = oof_preds
    # selective neutralize: top 30 loading features
    neutral_cols = feature_cols[:30]
    df = neutralize(df, ["pred"], neutral_cols, proportion=0.85)
    df[["era", "target", "pred"]].to_csv("models/oof_with_pred.csv", index=False)
    
    # 保存使用的特徵集信息
    with open("models/feature_set.txt", "w") as f:
        f.write(f"{feature_set}\n")
        f.write(f"特徵數量: {len(feature_cols)}\n")
        f.write(f"特徵列表:\n")
        for feature in feature_cols:
            f.write(f"{feature}\n")
    
    logger.info("訓練流程全部完成！")


if __name__ == "__main__":
    import sys
    os.makedirs("models", exist_ok=True)
    
    # 從命令行參數獲取特徵集，預設為 medium
    feature_set = sys.argv[1] if len(sys.argv) > 1 else "medium"
    
    if feature_set not in ["small", "medium", "all"]:
        print(f"錯誤: 無效的特徵集 '{feature_set}'")
        print("請選擇: small, medium, all")
        sys.exit(1)
    
    train(feature_set)