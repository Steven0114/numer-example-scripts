import numerapi, pyarrow.parquet as pq, os, pandas as pd
from config import TRAIN_YEARS_BACK

def download_latest_parquet(outfile="data/v5_full.parquet"):
    napi = numerapi.NumerAPI()
    file_id = napi.get_dataset_id("v5/train.parquet")
    napi.download_dataset(file_id, outfile)


def load_filtered_df(path="data/v5_full.parquet"):
    df = pq.read_table(path).to_pandas(types_mapper={"float": "float16"})
    recent_eras = sorted(df["era"].unique())[-52*TRAIN_YEARS_BACK:]
    return df[df["era"].isin(recent_eras)].copy()
