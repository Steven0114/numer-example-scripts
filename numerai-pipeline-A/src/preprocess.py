import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from config import NUM_FEATURES

def rank_gauss(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Rank → Gaussian transformation (per column)."""
    gauss = []
    for col in df_feat.columns:
        r = df_feat[col].rank(method="first").values
        r = (r - 0.5) / len(r)
        gauss.append(stats.norm.ppf(r))
    return pd.DataFrame(np.column_stack(gauss), columns=df_feat.columns, index=df_feat.index)


def pca_reduce(feat_df: pd.DataFrame, n=NUM_FEATURES, seed=0):
    pca = PCA(n_components=n, random_state=seed)
    return pca.fit_transform(feat_df).astype(np.float32)
