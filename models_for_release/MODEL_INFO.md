# 智能果园检测系统 - 模型文件

## 📋 模型列表

### 🍊 柑橘检测模型
- **文件**: `best-baseed-yolov11s.pt`
- **类型**: YOLOv11s
- **用途**: 柑橘果实检测
- **大小**: 19M
- **类别**: Citrus Fruit, Ground Fruit, Tree Fruit

### 🌱 作物推荐模型
- **文件**: `best_xgb_model.pkl`
- **类型**: XGBoost
- **用途**: 基于土壤和气候条件推荐作物
- **大小**: 1.6M

### 📊 数据缩放器
- **文件**: `scaler.pkl`
- **类型**: StandardScaler
- **用途**: 数据预处理和标准化
- **大小**: 4.0K

### 📈 产量预测模型
- **文件夹**: `AutogluonModels/`
- **类型**: AutoGluon TabularPredictor
- **用途**: 农作物产量预测
- **大小**: 5.7G

## 🚀 使用方法

1. 下载所有模型文件
2. 解压到项目的 `model/` 目录下
3. 确保目录结构如下：
   ```
   smart-orchard-system/
   ├── model/
   │   ├── best-baseed-yolov11s.pt
   │   ├── best_xgb_model.pkl
   │   ├── scaler.pkl
   │   └── AutogluonModels/
   │       └── ag-20250703_165505/
   ```
4. 运行系统: `./launch.sh`

## ⚠️  注意事项

- 模型文件总大小约为几百MB到几GB
- 确保有足够的磁盘空间
- 首次加载模型可能需要一些时间
- 建议使用GPU加速（如果可用）

## 📞 支持

如果遇到模型加载问题，请：
1. 检查文件完整性
2. 确认Python环境和依赖
3. 查看系统日志
4. 提交Issue到GitHub仓库
