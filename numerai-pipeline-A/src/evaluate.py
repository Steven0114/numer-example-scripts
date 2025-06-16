import pandas as pd, numpy as np

def era_corr(df):
    corrs = df.groupby("era").apply(lambda d: np.corrcoef(d["pred"], d["target"])[0,1])
    return corrs.mean(), corrs.std()

if __name__ == "__main__":
    df = pd.read_csv("models/oof_with_pred.csv")
    m, s = era_corr(df)
    print(f"Era-wise corr mean={m:.4f}, std={s:.4f}")
