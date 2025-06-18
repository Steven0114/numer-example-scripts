import numerapi, pyarrow.parquet as pq, pyarrow.dataset as ds, os, pandas as pd, json
from config import TRAIN_YEARS_BACK

def download_latest_parquet(outfile="data/v5_full.parquet"):
    """下載最新的訓練資料和特徵元數據"""
    napi = numerapi.NumerAPI()
    
    # 確保 data 目錄存在
    os.makedirs("data", exist_ok=True)
    
    # 下載訓練資料
    print("正在下載訓練資料...")
    napi.download_dataset("v5.0/train.parquet", outfile)
    
    # 下載特徵元數據
    features_file = "data/features.json"
    print("正在下載特徵元數據...")
    napi.download_dataset("v5.0/features.json", features_file)
    
    print("資料下載完成！")

def get_feature_set(feature_set="medium", features_file=None):
    """
    從 features.json 讀取指定的特徵集
    feature_set: "small", "medium", "all"
    features_file: features.json 的路徑，如果為 None 則自動尋找
    """
    if features_file is None:
        # 嘗試從不同的路徑找到 features.json
        possible_paths = [
            "data/features.json",
            "../data/features.json",
            "../../data/features.json"
        ]
        features_file = None
        for path in possible_paths:
            if os.path.exists(path):
                features_file = path
                break
        
        if features_file is None:
            print("錯誤: 找不到 features.json 文件")
            print("請確保已下載特徵元數據文件")
            return None
    
    if not os.path.exists(features_file):
        print(f"特徵元數據文件不存在: {features_file}")
        print("請先執行 download_latest_parquet() 下載數據")
        return None
    
    print(f"使用特徵元數據文件: {features_file}")
    
    with open(features_file, 'r') as f:
        feature_metadata = json.load(f)
    
    available_sets = list(feature_metadata["feature_sets"].keys())
    print(f"可用的特徵集: {available_sets}")
    
    if feature_set not in available_sets:
        print(f"錯誤: 特徵集 '{feature_set}' 不可用")
        print(f"請選擇以下之一: {available_sets}")
        return None
    
    features = feature_metadata["feature_sets"][feature_set]
    print(f"使用 '{feature_set}' 特徵集，共 {len(features)} 個特徵")
    
    return features

def load_filtered_df(path=None, feature_set="medium"):
    """
    載入並過濾資料，使用指定的特徵集
    feature_set: "small", "medium", "all"
    path: 資料文件路徑，如果為 None 則自動尋找
    """
    print("正在加載數據...")
    
    if path is None:
        # 嘗試從不同的路徑找到 parquet 文件
        possible_paths = [
            "data/v5_full.parquet",
            "../data/v5_full.parquet",
            "../../data/v5_full.parquet"
        ]
        path = None
        for p in possible_paths:
            if os.path.exists(p):
                path = p
                break
        
        if path is None:
            print("錯誤: 找不到訓練資料文件")
            print("請先執行 download_latest_parquet() 下載數據")
            return None
    
    print(f"使用資料文件: {path}")
    
    # 獲取特徵列表
    features = get_feature_set(feature_set)
    if features is None:
        # 如果無法獲取特徵集，回退到使用所有 feature_ 開頭的欄位
        print("警告: 無法載入特徵集，將使用所有 feature_ 開頭的欄位")
        dataset = ds.dataset(path, format="parquet")
        schema = dataset.schema
        features = [name for name in schema.names if name.startswith("feature_")]
        print(f"發現 {len(features)} 個特徵欄位")
    
    # 指定要載入的欄位：era, target, 和選定的特徵
    columns_to_load = ["era", "target"] + features
    
    # 先只读取era列以进行过滤
    dataset = ds.dataset(path, format="parquet")
    
    # 读取era列进行过滤
    era_table = dataset.to_table(columns=['era'])
    era_df = era_table.to_pandas()
    recent_eras = sorted(era_df["era"].unique())[-52*TRAIN_YEARS_BACK:]
    print(f"目標era數量: {len(recent_eras)}")
    
    # 使用era过滤来减少内存使用，並只載入需要的欄位
    filter_expr = ds.field('era').isin(recent_eras)
    filtered_table = dataset.to_table(columns=columns_to_load, filter=filter_expr)
    
    print("正在转换为pandas DataFrame...")
    df = filtered_table.to_pandas(use_threads=True)
    
    print("正在优化数据类型...")
    # 將 float 列转换为 float16 以节省内存
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float16')
    
    print(f"最終數據形狀: {df.shape}")
    print(f"載入的特徵數量: {len([c for c in df.columns if c.startswith('feature_')])}")
    
    return df