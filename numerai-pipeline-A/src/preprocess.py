import pandas as pd, numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import rankdata
from config import NUM_FEATURES

def rank_gauss(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Rank → Gaussian transformation (per column) - 内存优化版本."""
    n_rows, n_cols = df_feat.shape
    result = np.zeros((n_rows, n_cols), dtype=np.float32)
    
    # 分块处理以节省内存
    chunk_size = min(100, n_cols)  # 每次处理100列或更少
    
    for i in range(0, n_cols, chunk_size):
        end_idx = min(i + chunk_size, n_cols)
        chunk_cols = df_feat.columns[i:end_idx]
        chunk_data = df_feat[chunk_cols]
        
        # 对当前块进行转换
        r = chunk_data.apply(lambda x: rankdata(x, method="ordinal"), axis=0).astype(np.float32)
        r = (r - 0.5) / n_rows
        gauss_chunk = stats.norm.ppf(r).astype(np.float32)
        
        # 存储结果
        result[:, i:end_idx] = gauss_chunk
    
    return pd.DataFrame(result, columns=df_feat.columns, index=df_feat.index)


def pca_reduce(feat_df: pd.DataFrame, n=NUM_FEATURES, seed=0):
    pca = PCA(n_components=n, random_state=seed)
    return pca.fit_transform(feat_df).astype(np.float32)