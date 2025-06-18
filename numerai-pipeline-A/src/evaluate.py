"""Utility script for evaluating out-of-fold predictions."""

from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from numerai_tools.scoring import correlation_contribution


def era_corr(df: pd.DataFrame, pred_col: str = "pred", target_col: str = "target"):
    """Return the mean and std of era-wise correlations."""
    corrs = df.groupby("era").apply(
        lambda d: np.corrcoef(d[pred_col], d[target_col])[0, 1]
    )
    return corrs.mean(), corrs.std()


def enhanced_era_metrics(
    df: pd.DataFrame, pred_col: str = "pred", target_col: str = "target"
) -> tuple[dict[str, float], pd.Series]:
    """Calculate enhanced era level metrics."""

    corrs = df.groupby("era").apply(
        lambda d: np.corrcoef(d[pred_col], d[target_col])[0, 1]
    )

    cumsum = corrs.cumsum()
    metrics = {
        "mean": corrs.mean(),
        "std": corrs.std(),
        "sharpe": corrs.mean() / corrs.std() if corrs.std() > 0 else 0,
        "p5": corrs.quantile(0.05),
        "p95": corrs.quantile(0.95),
        "min": corrs.min(),
        "max": corrs.max(),
        "count": len(corrs),
        "positive_ratio": (corrs > 0).mean(),
        "max_drawdown": (cumsum.expanding(min_periods=1).max() - cumsum).max(),
    }

    return metrics, corrs


def mmc_metrics(
    df: pd.DataFrame,
    pred_col: str,
    meta_col: str,
    target_col: str,
) -> tuple[dict[str, float], pd.Series]:
    """Calculate MMC metrics if meta model predictions are provided."""

    mmc = df.dropna(subset=[meta_col]).groupby("era").apply(
        lambda d: correlation_contribution(
            d[[pred_col]], d[meta_col], d[target_col]
        )[pred_col]
    )

    cumsum = mmc.cumsum()
    metrics = {
        "mean": mmc.mean(),
        "std": mmc.std(),
        "sharpe": mmc.mean() / mmc.std() if mmc.std() > 0 else 0,
        "max_drawdown": (cumsum.expanding(min_periods=1).max() - cumsum).max(),
    }

    return metrics, mmc


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions")
    parser.add_argument("--file", default="models/oof_with_pred.csv")
    parser.add_argument("--pred-col", default="pred")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--meta-col", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.file)

    # basic correlation metrics
    m, s = era_corr(df, args.pred_col, args.target_col)
    print(f"Era-wise corr mean={m:.4f}, std={s:.4f}")

    metrics, corrs = enhanced_era_metrics(df, args.pred_col, args.target_col)
    print("\n=== Enhanced Metrics ===")
    print(f"Correlation Mean:     {metrics['mean']:.6f}")
    print(f"Correlation Std:      {metrics['std']:.6f}")
    print(f"Sharpe Ratio:         {metrics['sharpe']:.6f}")
    print(f"5% Percentile:        {metrics['p5']:.6f}")
    print(f"95% Percentile:       {metrics['p95']:.6f}")
    print(f"Min Correlation:      {metrics['min']:.6f}")
    print(f"Max Correlation:      {metrics['max']:.6f}")
    print(f"Era Count:            {metrics['count']}")
    print(f"Positive Era Ratio:   {metrics['positive_ratio']:.2%}")
    print(f"Max Drawdown:         {metrics['max_drawdown']:.6f}")

    risk_level = (
        "低" if metrics["p5"] > 0 else "中" if metrics["p5"] > -0.02 else "高"
    )
    print("\n=== Risk Assessment ===")
    print(f"Risk Level:           {risk_level}")
    print(f"Tail Risk (p5):       {metrics['p5']:.6f}")
    print(f"Upside Potential:     {metrics['p95']:.6f}")

    if args.meta_col and args.meta_col in df.columns:
        mmc_m, mmc_scores = mmc_metrics(
            df, args.pred_col, args.meta_col, args.target_col
        )
        print("\n=== MMC Metrics ===")
        print(f"MMC Mean:             {mmc_m['mean']:.6f}")
        print(f"MMC Std:              {mmc_m['std']:.6f}")
        print(f"MMC Sharpe:           {mmc_m['sharpe']:.6f}")
        print(f"MMC Max Drawdown:     {mmc_m['max_drawdown']:.6f}")


if __name__ == "__main__":
    main()