import numpy as np, pandas as pd
from scipy import stats

def neutralize(df, columns, neutralizers, proportion=0.85, era_col="era"):
    """Selective neutralize on given columns vs. neutralizer set."""
    out = []
    for era, era_df in df.groupby(era_col):
        scores = era_df[columns].rank(method="first").values
        scores = stats.norm.ppf((scores - 0.5) / scores.shape[0])
        exposures = era_df[neutralizers].fillna(era_df[neutralizers].median()).values
        scores = scores - proportion * exposures @ np.linalg.pinv(exposures, rcond=1e-6) @ scores
        scores = scores / np.std(scores, axis=0, ddof=0)
        out.append(scores)
    clean = pd.DataFrame(np.vstack(out), columns=columns, index=df.index)
    df[columns] = clean
    return df
