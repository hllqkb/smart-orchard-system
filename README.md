# 🍊 智能果园检测系统

## 项目简介

智能果园检测系统是一个综合性的农业智能化平台，集成了多种先进技术，为果园管理提供全方位的智能化解决方案。

## 主要功能

### 🔐 用户认证系统
- **传统登录**: 用户名/密码登录
- **人脸识别登录**: 免密码人脸识别登录
- **用户注册**: 支持新用户注册和人脸信息录入

### 🍊 果园智能检测
- **实时图像检测**: 基于YOLO-MECD的柑橘检测
- **批量图像处理**: 支持多张图像同时检测
- **检测结果分析**: 详细的检测统计和可视化

### 📹 视频内容理解
- **视频分析**: 基于Visual-Vision-RAG的视频内容理解
- **智能查询**: 支持自然语言查询视频内容
- **关键帧提取**: 自动提取和分析关键帧

### 🌱 农业预测与建议
- **作物推荐**: 基于土壤和气候条件推荐适合的作物
- **产量预测**: 预测农作物产量
- **智能建议**: AI生成的种植建议和优化方案

### 📊 数据管理
- **用户数据**: 安全的用户信息存储
- **检测历史**: 检测结果的历史记录
- **数据可视化**: 丰富的图表和统计信息

## 技术架构

### 前端技术
- **Streamlit**: 主要的Web界面框架
- **Plotly**: 数据可视化
- **PIL/OpenCV**: 图像处理
- **CSS**: 自定义样式

### 后端技术
- **Python**: 主要开发语言
- **YOLO**: 目标检测模型
- **Transformers**: 自然语言处理
- **AutoGluon**: 机器学习预测
- **SQLite**: 数据存储

### AI模型
- **YOLO-MECD**: 柑橘检测模型
- **BLIP**: 图像描述生成
- **DETR**: 目标检测
- **XGBoost**: 农作物推荐
- **文心一言**: 智能对话和建议

## 项目结构

```
smart-orchard-system/
├── main.py                 # 主应用入口
├── config/                 # 配置文件
│   ├── settings.py         # 系统配置
│   └── database.py         # 数据库配置
├── modules/                # 功能模块
│   ├── auth/              # 认证模块
│   ├── detection/         # 检测模块
│   ├── video_analysis/    # 视频分析模块
│   ├── prediction/        # 预测模块
│   └── utils/             # 工具模块
├── models/                # 模型文件
├── data/                  # 数据文件
├── static/                # 静态资源
├── templates/             # 模板文件
└── requirements.txt       # 依赖包
```

## 快速开始

### 🚀 推荐启动方式

```bash
# 使用启动脚本 (自动激活conda环境)
./launch.sh
```

### 🔧 手动启动方式

```bash
# 激活conda环境
source /home/hllqk/miniconda3/etc/profile.d/conda.sh
conda activate deeplearn

# 启动系统
streamlit run main.py
```

### 📋 系统检查

```bash
# 检查所有模型和功能
source /home/hllqk/miniconda3/etc/profile.d/conda.sh
conda activate deeplearn
python demo.py
```

### 📦 模型文件位置

- **柑橘检测模型**: `model/best-baseed-yolov11s.pt`
- **作物推荐模型**: `model/best_xgb_model.pkl`
- **作物缩放器**: `model/scaler.pkl`
- **产量预测模型**: `model/AutogluonModels/ag-20250703_165505`

### 📋 手动安装步骤

1. **环境要求**
   - Python 3.8+
   - 8GB+ RAM
   - 10GB+ 磁盘空间
   - CUDA (可选，用于GPU加速)

2. **克隆项目**
```bash
git clone <repository-url>
cd smart-orchard-system
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **配置模型路径**
```bash
# 编辑 config/settings.py 中的模型路径
# 确保以下模型文件存在：
# - YOLO柑橘检测模型: /path/to/yolo/best.pt
# - 作物推荐模型: /path/to/crop/model.pkl
# - 产量预测模型: /path/to/yield/model
```

5. **运行应用**
```bash
streamlit run main.py
# 或者
python run.py
```

## 使用说明

### 首次使用
1. 访问系统主页
2. 注册新用户账号
3. 可选择录入人脸信息以启用人脸登录
4. 登录后即可使用各项功能

### 功能使用
- **果园检测**: 上传果园图像进行智能检测
- **视频分析**: 上传果园视频进行内容分析
- **农业预测**: 输入土壤和气候参数获取建议
- **历史查看**: 查看之前的检测和分析结果

## 开发团队

本项目整合了以下开源项目的优秀功能：
- Visual-Vision-RAG: 视频内容理解
- YOLO-MECD: 柑橘检测
- Agriculture-Prediction: 农业预测

## 许可证

MIT License

## 🎯 核心特性

### 🔐 智能认证系统
- **双重登录方式**: 传统密码 + 人脸识别
- **安全数据存储**: 加密用户信息和人脸特征
- **会话管理**: 自动超时和安全登出

### 🍊 果园智能检测
- **专用柑橘检测**: 基于YOLOv11s训练的专用柑橘检测模型
- **高精度识别**: 针对柑橘果实优化的检测算法
- **实时检测**: 快速准确的柑橘果实识别和定位
- **批量处理**: 支持多张图像同时检测
- **详细分析**: 检测统计、置信度分析、结果可视化
- **历史记录**: 完整的检测历史和数据导出

### 📹 视频内容理解
- **智能分析**: 自动提取关键帧并进行内容分析
- **自然语言查询**: 支持中文问答式视频内容检索
- **事件时间线**: 生成视频事件摘要和时间轴
- **RAG技术**: 基于向量数据库的智能检索

### 🌱 农业预测建议
- **作物推荐**: 基于土壤和气候条件的智能作物推荐
- **产量预测**: 机器学习驱动的作物产量预测
- **AI建议**: 个性化的种植建议和优化方案
- **数据分析**: 详细的农业数据分析和可视化

### 🤖 模型训练中心
- **自定义数据集训练**: 支持上传ZIP格式数据集
- **多种网络架构**: LeNet、AlexNet、ResNet、VGG、EfficientNet、R-CNN
- **深度学习框架**: 基于PyTorch的现代化训练流程
- **实时训练监控**: 训练过程可视化和性能监控
- **模型管理**: 训练历史、模型下载和部署
- **智能优化**: 自动学习率调整和数据增强

### 💾 数据管理系统
- **安全存储**: 用户数据、检测结果、分析历史
- **数据导出**: 支持JSON、CSV、Excel多种格式
- **备份恢复**: 完整的数据备份和恢复功能
- **存储监控**: 实时存储使用情况监控

### 🔧 系统监控
- **性能监控**: CPU、内存、GPU使用率实时监控
- **应用统计**: 用户活动、功能使用统计
- **日志管理**: 完整的系统操作日志
- **健康检查**: 自动系统健康状态检测

## 🛠️ 技术栈

### 前端技术
- **Streamlit**: 现代化Web界面框架
- **Plotly**: 交互式数据可视化
- **CSS3**: 响应式界面设计
- **JavaScript**: 动态交互效果

### 后端技术
- **Python 3.8+**: 主要开发语言
- **SQLite**: 轻量级数据库
- **OpenCV**: 图像处理
- **NumPy/Pandas**: 数据处理

### AI/ML技术
- **YOLO**: 目标检测
- **Transformers**: 自然语言处理
- **LangChain**: RAG框架
- **FAISS**: 向量数据库
- **AutoGluon**: 自动机器学习
- **Face Recognition**: 人脸识别

### 部署技术
- **Docker**: 容器化部署 (计划中)
- **Nginx**: 反向代理 (计划中)
- **SSL/TLS**: 安全传输 (计划中)

## 📊 系统架构

```
智能果园检测系统
├── 用户认证层
│   ├── 传统登录
│   └── 人脸识别
├── 业务逻辑层
│   ├── 果园检测
│   ├── 视频分析
│   ├── 农业预测
│   └── 数据管理
├── 数据存储层
│   ├── 用户数据
│   ├── 检测结果
│   ├── 分析历史
│   └── 系统日志
└── 基础设施层
    ├── 模型管理
    ├── 缓存系统
    ├── 文件存储
    └── 监控告警
```

## 🔧 配置说明

### 模型配置
编辑 `config/settings.py` 文件中的模型路径：

```python
MODEL_PATHS = {
    "yolo_citrus": "/path/to/yolo/best.pt",
    "crop_recommendation": "/path/to/crop/model.pkl",
    "crop_scaler": "/path/to/scaler.pkl",
    "yield_prediction": "/path/to/yield/model"
}
```

### API配置
配置AI服务API密钥：

```python
API_CONFIG = {
    "ernie_api_key": "your_ernie_api_key",
    "openrouter_api_key": "your_openrouter_api_key"
}
```

### 系统配置
调整系统参数：

```python
# 检测参数
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_file_size": 10 * 1024 * 1024  # 10MB
}

# 视频参数
VIDEO_CONFIG = {
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "frame_extraction_interval": 5,  # 秒
    "max_frames": 20
}
```

## 📱 使用指南

### 首次使用
1. **注册账号**: 创建新用户账号
2. **设置人脸**: 可选择录入人脸信息
3. **开始使用**: 登录后即可使用各项功能

### 果园检测
1. **上传图像**: 选择果园图像文件
2. **调整参数**: 设置检测置信度等参数
3. **开始检测**: 点击检测按钮进行分析
4. **查看结果**: 查看检测结果和统计信息

### 视频分析
1. **上传视频**: 选择果园视频文件
2. **设置参数**: 配置帧提取间隔等参数
3. **开始分析**: 系统自动提取关键帧并分析
4. **智能查询**: 使用自然语言查询视频内容

### 农业预测
1. **输入参数**: 填写土壤和气候数据
2. **选择功能**: 作物推荐或产量预测
3. **获取结果**: 查看预测结果和AI建议
4. **保存记录**: 结果自动保存到历史记录

## 🚨 故障排除

### 常见问题

**Q: 模型加载失败**
A: 确保使用正确的conda环境: `conda activate deeplearn`

**Q: JSON序列化错误**
A: 已修复numpy类型的JSON序列化问题

**Q: AutoGluon版本警告**
A: 已添加版本兼容性参数，可正常使用

**Q: 产量预测失败 (No module named 'lightgbm')**
A: 已安装lightgbm依赖，问题已解决

**Q: 视频分析功能**
A: 集成了Visual-Vision-RAG系统，提供专业视频分析

**Q: 人脸识别不工作**
A: 确保已安装dlib和face_recognition库，检查摄像头权限

**Q: API调用失败**
A: 检查网络连接和API密钥配置，确保密钥有效

**Q: 界面显示异常**
A: 清除浏览器缓存，或尝试无痕模式访问

### 日志查看
```bash
# 查看应用日志
tail -f data/logs/app.log

# 查看系统监控
python -c "from modules.utils.system_monitor import system_monitor; system_monitor.show_system_dashboard()"
```

### 性能优化
- 启用GPU加速（需要CUDA环境）
- 调整图像处理参数
- 定期清理缓存和临时文件
- 监控系统资源使用情况

## 🤝 贡献指南

### 开发环境设置
```bash
# 克隆仓库
git clone <repository-url>
cd smart-orchard-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装开发依赖
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 如果有开发依赖
```

### 代码规范
- 遵循PEP 8代码风格
- 添加适当的注释和文档字符串
- 编写单元测试
- 提交前运行代码检查

### 提交流程
1. Fork项目仓库
2. 创建功能分支
3. 提交代码更改
4. 创建Pull Request
5. 代码审查和合并

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

本项目整合了以下优秀开源项目的功能：
- **Visual-Vision-RAG**: 视频内容理解技术
- **YOLO-MECD**: 柑橘检测模型
- **Agriculture-Prediction**: 农业预测算法

感谢所有贡献者和开源社区的支持！

## 📞 联系我们

- **项目主页**: [GitHub Repository]
- **问题反馈**: [GitHub Issues]
- **技术讨论**: [GitHub Discussions]
- **邮箱联系**: [your-email@example.com]

## 🔄 更新日志

### v1.0.0 (2025-01-24)
- ✨ 初始版本发布
- 🔐 集成用户认证和人脸识别系统
- 🍊 实现YOLO-MECD柑橘检测功能
- 📹 集成Visual-Vision-RAG视频分析
- 🌱 添加农业预测和建议功能
- 💾 完整的数据管理和存储系统
- 🔧 系统监控和性能优化
- 📱 响应式Web界面设计
- 📚 完整的文档和使用指南

### 计划中的功能
- 🐳 Docker容器化部署
- 🌐 多语言支持
- 📊 高级数据分析和报表
- 🔔 实时通知和告警
- 📱 移动端适配
- 🤖 更多AI模型集成
