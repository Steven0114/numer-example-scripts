NUM_FEATURES          = 50          # PCA 主成分個數
TRAIN_YEARS_BACK      = 4           # 只用最近 N 年資料
SEEDS                 = [0,1,2,3,4]
N_SPLITS              = 5           # era‐wise KFold
LIGHTGBM_PARAMS = dict(
    objective       = "regression",
    n_estimators    = 400,
    learning_rate   = 0.03,
    max_depth       = 7,
    num_leaves      = 64,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_lambda      = 1.0,
    min_child_samples = 20
)
