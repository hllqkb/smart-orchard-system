#!/bin/bash

# 打包模型文件用于GitHub Release
# 使用方法: ./package_models.sh

echo "📦 开始打包模型文件..."

# 创建临时目录
TEMP_DIR="models_for_release"
mkdir -p "$TEMP_DIR"

# 检查并复制模型文件
echo "🔍 检查模型文件..."

# YOLO柑橘检测模型
if [ -f "model/best-baseed-yolov11s.pt" ]; then
    echo "✅ 找到YOLO柑橘检测模型"
    cp "model/best-baseed-yolov11s.pt" "$TEMP_DIR/"
    MODEL_COUNT=$((MODEL_COUNT + 1))
else
    echo "⚠️  未找到YOLO柑橘检测模型: model/best-baseed-yolov11s.pt"
fi

# 作物推荐模型
if [ -f "model/best_xgb_model.pkl" ]; then
    echo "✅ 找到作物推荐模型"
    cp "model/best_xgb_model.pkl" "$TEMP_DIR/"
    MODEL_COUNT=$((MODEL_COUNT + 1))
else
    echo "⚠️  未找到作物推荐模型: model/best_xgb_model.pkl"
fi

# 缩放器
if [ -f "model/scaler.pkl" ]; then
    echo "✅ 找到缩放器文件"
    cp "model/scaler.pkl" "$TEMP_DIR/"
    MODEL_COUNT=$((MODEL_COUNT + 1))
else
    echo "⚠️  未找到缩放器文件: model/scaler.pkl"
fi

# AutoGluon产量预测模型
# if [ -d "model/AutogluonModels" ]; then
#     echo "✅ 找到AutoGluon产量预测模型"
#     cp -r "model/AutogluonModels" "$TEMP_DIR/"
#     MODEL_COUNT=$((MODEL_COUNT + 1))
# else
#     echo "⚠️  未找到AutoGluon产量预测模型: model/AutogluonModels/"
# fi

# 创建模型信息文件
cat > "$TEMP_DIR/MODEL_INFO.md" << EOF
# 智能果园检测系统 - 模型文件

## 📋 模型列表

### 🍊 柑橘检测模型
- **文件**: \`best-baseed-yolov11s.pt\`
- **类型**: YOLOv11s
- **用途**: 柑橘果实检测
- **大小**: $(du -h model/best-baseed-yolov11s.pt 2>/dev/null | cut -f1 || echo "未知")
- **类别**: Citrus Fruit, Ground Fruit, Tree Fruit

### 🌱 作物推荐模型
- **文件**: \`best_xgb_model.pkl\`
- **类型**: XGBoost
- **用途**: 基于土壤和气候条件推荐作物
- **大小**: $(du -h model/best_xgb_model.pkl 2>/dev/null | cut -f1 || echo "未知")

### 📊 数据缩放器
- **文件**: \`scaler.pkl\`
- **类型**: StandardScaler
- **用途**: 数据预处理和标准化
- **大小**: $(du -h model/scaler.pkl 2>/dev/null | cut -f1 || echo "未知")

### 📈 产量预测模型
- **文件夹**: \`AutogluonModels/\`
- **类型**: AutoGluon TabularPredictor
- **用途**: 农作物产量预测
- **大小**: $(du -sh model/AutogluonModels 2>/dev/null | cut -f1 || echo "未知")

## 🚀 使用方法

1. 下载所有模型文件
2. 解压到项目的 \`model/\` 目录下
3. 确保目录结构如下：
   \`\`\`
   smart-orchard-system/
   ├── model/
   │   ├── best-baseed-yolov11s.pt
   │   ├── best_xgb_model.pkl
   │   ├── scaler.pkl
   │   └── AutogluonModels/
   │       └── ag-20250703_165505/
   \`\`\`
4. 运行系统: \`./launch.sh\`

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
EOF

# 打包文件
echo "📦 创建压缩包..."
tar -czf "smart-orchard-models-v1.0.0.tar.gz" -C "$TEMP_DIR" .

# 计算文件大小和哈希
ARCHIVE_SIZE=$(du -h "smart-orchard-models-v1.0.0.tar.gz" | cut -f1)
ARCHIVE_HASH=$(sha256sum "smart-orchard-models-v1.0.0.tar.gz" | cut -d' ' -f1)

echo ""
echo "✅ 模型文件打包完成！"
echo ""
echo "📋 打包信息:"
echo "  📦 文件名: smart-orchard-models-v1.0.0.tar.gz"
echo "  📏 大小: $ARCHIVE_SIZE"
echo "  🔐 SHA256: $ARCHIVE_HASH"
echo "  📁 包含文件: $(ls -la "$TEMP_DIR" | wc -l) 个"
echo ""
echo "📤 上传到GitHub Release的步骤:"
echo "1. 访问您的GitHub仓库"
echo "2. 点击 'Releases' -> 'Create a new release'"
echo "3. 标签版本: v1.0.0"
echo "4. 发布标题: Smart Orchard Detection System v1.0.0"
echo "5. 上传文件: smart-orchard-models-v1.0.0.tar.gz"
echo "6. 添加发布说明并发布"
echo ""

# 清理临时目录
rm -rf "$TEMP_DIR"

echo "🎉 打包完成！模型文件已准备好上传到GitHub Release。"
