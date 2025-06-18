# Numerai Example Scripts

A collection of scripts and notebooks to help you get started quickly. 

Need help? Find us on Discord:

[![](https://dcbadge.vercel.app/api/server/numerai)](https://discord.gg/numerai)

---

## 📋 Numerai 模型上傳標準流程

> ✅ **成功上傳關鍵** - 不論您使用何種模型設計，以下標準流程是成功上傳到 Numerai 官網的必要步驟

### 🎯 **第一步：模型結構標準**
您的模型**必須**是一個預測函數，具有以下標準格式：

```python
def predict(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame
) -> pd.DataFrame:
    """
    標準預測函數
    
    參數:
    live_features: 包含特徵的DataFrame
    live_benchmark_models: Numerai提供的基準模型
    
    返回:
    包含'prediction'列的DataFrame
    """
    # 您的預測邏輯
    predictions = your_model.predict(live_features[your_features])
    
    # 標準返回格式
    submission = pd.Series(predictions, index=live_features.index)
    return submission.to_frame("prediction")
```

### 🎯 **第二步：序列化標準**
**必須**使用 `cloudpickle` 進行序列化：

```python
import cloudpickle

# 序列化預測函數
p = cloudpickle.dumps(predict)
with open("your_model.pkl", "wb") as f:
    f.write(p)
```

### 🎯 **第三步：驗證標準**
上傳前**必須**通過以下測試：

```python
# 載入測試
with open("your_model.pkl", "rb") as f:
    predict_func = cloudpickle.loads(f.read())

# 創建測試數據
test_features = pd.DataFrame(...)  # 您的特徵格式
test_benchmark = pd.DataFrame(...)  # 空的基準模型

# 預測測試
predictions = predict_func(test_features, test_benchmark)

# 驗證格式
assert isinstance(predictions, pd.DataFrame)
assert "prediction" in predictions.columns
assert len(predictions) == len(test_features)
```

### 🎯 **第四步：常見錯誤避免**

#### ❌ **絕對不要做的事情**
- ❌ 使用自定義類作為模型結構
- ❌ 混合使用 `joblib` 和 `cloudpickle`
- ❌ 在函數內使用複雜的依賴鏈
- ❌ 忽略函數簽名標準
- ❌ 返回非 DataFrame 格式

#### ✅ **必須遵循的原則**
- ✅ 純函數結構，避免類依賴
- ✅ 最小化外部依賴
- ✅ 標準輸入輸出格式
- ✅ 使用官方推薦的 `cloudpickle==2.2.1`
- ✅ 測試通過後再上傳

### 🎯 **第五步：文件檢查**
上傳前確認：
- ✅ 文件大小 < 10 MB
- ✅ 文件名以 `.pkl` 結尾
- ✅ 通過本地預測測試
- ✅ 無警告或錯誤信息

---

## 📚 Notebooks 

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

---

## 🏭 numerai-pipeline-A 高級訓練流程

> **完整的端對端 Numerai 模型訓練與上傳解決方案**

### 🎯 **設計理念**
- **25 模型集成**: 5 seeds × 5 folds = 25 個 LightGBM 模型
- **特徵工程**: Rank-Gaussian 變換 + PCA 降維
- **風險控制**: Selective feature neutralization
- **官方兼容**: 完全符合 Numerai 上傳標準

### 🚀 **完整執行流程**

#### **步驟 1: 數據準備**
```powershell
# 下載最新數據（支援特徵集選擇）  
python - <<EOF
from src.data_fetch import download_latest_parquet
download_latest_parquet()
EOF
```

#### **步驟 2: 模型訓練**
```powershell
# 使用預設的 medium 特徵集（推薦）
python src/train.py

# 或指定其他特徵集
python src/train.py small    # 使用 small 特徵集 (~42 features)
python src/train.py medium   # 使用 medium 特徵集 (~700 features)  
python src/train.py all      # 使用 all 特徵集 (~2300 features)
```

#### **步驟 3: 模型評估**
```powershell
# 檢視交叉驗證結果
python src/evaluate.py
```

#### **步驟 4: 生成上傳模型**
```powershell
# 生成符合官方標準的模型文件
python src/create_final_official_model.py numerai_final_model.pkl medium

# 測試模型兼容性
python src/create_final_official_model.py test numerai_final_model.pkl
```

#### **步驟 5: Live 預測**
```powershell
# 下載 live 數據
python src/predict.py download

# 執行預測（自動使用訓練時的特徵集）
python src/predict.py
```

### 🔧 **技術架構**

#### **特徵處理流程**
1. **特徵選擇**: 支援 `small`/`medium`/`all` 三種特徵集
2. **數據篩選**: 載入近 4 年 era 數據
3. **Rank-Gaussian**: 將特徵轉換為正態分佈
4. **PCA 降維**: 降維至 50 維以提升效率

#### **模型訓練架構**
1. **GroupKFold**: 以 era 為組進行交叉驗證
2. **多種子訓練**: 5 個不同隨機種子確保穩定性
3. **集成策略**: 25 個模型的平均集成
4. **特徵中和**: 對前 30 個特徵進行 85% 中和處理

### 📊 **性能特點**

#### **内存优化**
- **分块处理**: 避免大数据集内存溢出
- **数据过滤**: 只加载必要的 era 数据
- **类型优化**: 使用 float16 减少内存占用

#### **训练效率**
- **并行训练**: 充分利用多核 CPU
- **早停机制**: 防止过拟合
- **进度监控**: 详细的日志记录

### 🎯 **输出文件**

完成训练后将生成：
- **25 个 LightGBM 模型**: `models/lgbm_seed{X}_f{Y}.pkl`
- **PCA 模型**: `models/pca_model.pkl`
- **交叉验证结果**: `models/oof_with_pred.csv`
- **特征集信息**: `models/feature_set.txt`
- **官方兼容模型**: 通过 `create_final_official_model.py` 生成

### 📋 **故障排除**

#### **常见问题解决**
- ✅ **内存不足**: 已实现分块处理和数据过滤
- ✅ **API 兼容性**: 移除不支持的参数
- ✅ **进度条冲突**: 使用日志替代嵌套进度条
- ✅ **上传格式**: 严格按照 Numerai 标准

#### **技术文档**
- 📋 [BUGFIX_SUMMARY.md](numerai-pipeline-A/BUGFIX_SUMMARY.md) - 问题修复总结
- 🔧 [FIXED_MODEL_GUIDE.md](numerai-pipeline-A/FIXED_MODEL_GUIDE.md) - 模型修正指南
- 📊 [OPTIMIZATION_SUMMARY.md](numerai-pipeline-A/OPTIMIZATION_SUMMARY.md) - 优化总结
- 🎯 [OFFICIAL_MODEL_SOLUTION.md](numerai-pipeline-A/OFFICIAL_MODEL_SOLUTION.md) - 官方模型解决方案

---

**开始使用 numerai-pipeline-A 构建你的 Numerai 模型！** 🚀