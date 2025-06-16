import glob, joblib, numpy as np, pandas as pd
from preprocess import rank_gauss, pca_reduce

def predict_live(live_df: pd.DataFrame):
    feature_cols = [c for c in live_df.columns if c.startswith("feature_")]
    feat = pca_reduce(rank_gauss(live_df[feature_cols]))
    preds = np.zeros(len(live_df), dtype=np.float32)
    for pkl in glob.glob("models/lgbm_seed*.pkl"):
        model = joblib.load(pkl)
        preds += model.predict(feat) / len(glob.glob("models/lgbm_seed*.pkl"))
    live_df["prediction"] = preds
    live_df.to_csv("predictions.csv", index=False)
