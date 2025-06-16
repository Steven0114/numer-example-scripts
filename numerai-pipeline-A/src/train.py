import os, joblib, gc, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
from config import SEEDS, N_SPLITS, LIGHTGBM_PARAMS
from data_fetch import load_filtered_df
from preprocess import rank_gauss, pca_reduce
from neutralize import neutralize


def train():
    df = load_filtered_df()
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    df_feat = rank_gauss(df[feature_cols])
    pca_feat = pca_reduce(df_feat)
    target = df["target"].astype(np.float32)
    groups = df["era"].values

    oof_preds = np.zeros(len(df), dtype=np.float32)
    for seed in SEEDS:
        lgb_params = LIGHTGBM_PARAMS | {"random_state": seed}
        kf = GroupKFold(n_splits=N_SPLITS)
        for fold, (tr_idx, val_idx) in enumerate(kf.split(pca_feat, target, groups)):
            X_tr, y_tr = pca_feat[tr_idx], target.iloc[tr_idx]
            X_val, y_val = pca_feat[val_idx], target.iloc[val_idx]
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      eval_metric="l1", callbacks=[lgb.early_stopping(50)])
            joblib.dump(model, f"models/lgbm_seed{seed}_f{fold}.pkl")
            oof_preds[val_idx] += model.predict(X_val) / len(SEEDS)
        gc.collect()

    df["pred"] = oof_preds
    # selective neutralize: top 30 loading features
    neutral_cols = feature_cols[:30]
    df = neutralize(df, ["pred"], neutral_cols, proportion=0.85)
    df[["era", "target", "pred"]].to_csv("models/oof_with_pred.csv", index=False)


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    train()
