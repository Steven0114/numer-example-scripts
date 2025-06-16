# Numerai Example Scripts

A collection of scripts and notebooks to help you get started quickly. 

Need help? Find us on Discord:

[![](https://dcbadge.vercel.app/api/server/numerai)](https://discord.gg/numerai)


## Notebooks 

Try running these notebooks on Google Colab's free tier!

### Hello Numerai
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/hello_numerai.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Start here if you are new! Explore the dataset and build your first model. 

### Feature Neutralization
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/feature_neutralization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Learn how to measure feature risk and control it with feature neutralization.

### Target Ensemble
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/target_ensemble.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Learn how to create an ensemble trained on different targets.

### Model Upload
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/example_model.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A barebones example of how to build and upload your model to Numerai.
## numerai-pipeline-A Usage

### 執行流程
1. 下載資料  
   ```powershell
   python - <<EOF
   from src.data_fetch import download_latest_parquet
   download_latest_parquet()
   EOF
   ```
2. 訓練
   ```powershell
   python src/train.py
   ```
3. 檢視交叉驗證
   ```powershell
   python src/evaluate.py
   ```
4. 產生 live 預測（在 Compute Heavy / 本機均可）
   ```powershell
   python src/predict.py
   ```

> ✅ **說明**  
> - `train.py` 會自動：
>   1. 只讀近4年 era  
>   2. Rank→Gaussian  
>   3. PCA 取50 維  
>   4. 5 × 5 = 25 模型（各自5 seed × 5 fold）訓練並存在 `models/`
>   5. 對 OOF 預測做 **selective neutralize (proportion 0.85)**  
> - `evaluate.py` 讀取 OOF CSV，列卷 era-wise 平均 corr 與波動。  
> - `predict.py` 只需在 Compute Heavy 任務中下載最新 live parquet，跑一次即可輸出 `predictions.csv` 並上傳。
