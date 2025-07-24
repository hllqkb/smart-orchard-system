#!/usr/bin/env python3
"""
智能果园检测系统 - 安装脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """安装依赖包"""
    print("📦 安装依赖包...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ 错误: 找不到requirements.txt文件")
        return False
    
    try:
        # 升级pip
        print("⬆️ 升级pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        
        # 安装依赖
        print("📦 安装项目依赖...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        
        print("✅ 依赖包安装完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {str(e)}")
        return False

def setup_environment():
    """设置环境"""
    print("🔧 设置环境...")
    
    # 创建必要的目录
    project_root = Path(__file__).parent
    directories = [
        "data",
        "data/user_images",
        "data/video_frames", 
        "data/face_encodings",
        "data/cache",
        "data/exports",
        "data/backups",
        "data/logs",
        "temp",
        "temp/video_frames",
        "models",
        "static"
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ 创建目录: {dir_name}")
    
    print("✅ 环境设置完成")
    return True

def check_optional_dependencies():
    """检查可选依赖"""
    print("🔍 检查可选依赖...")
    
    optional_packages = {
        "autogluon.tabular": "产量预测功能",
        "dlib": "人脸识别功能",
        "paddlenlp": "高级NLP功能"
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package.replace('-', '_').replace('.', '_'))
            print(f"  ✅ {package} - {description}")
        except ImportError:
            print(f"  ⚠️ {package} - {description} (可选，未安装)")
    
    return True

def create_config_template():
    """创建配置模板"""
    print("📝 创建配置模板...")
    
    config_template = """# 智能果园检测系统配置文件
# 请根据实际情况修改以下配置

# 模型路径配置
YOLO_MODEL_PATH = "/path/to/yolo/model/best.pt"
CROP_MODEL_PATH = "/path/to/crop/model/best_xgb_model.pkl"
CROP_SCALER_PATH = "/path/to/crop/scaler/scaler.pkl"
YIELD_MODEL_PATH = "/path/to/yield/model/ag-20250703_165505"

# API配置
ERNIE_API_KEY = "your_ernie_api_key_here"
OPENROUTER_API_KEY = "your_openrouter_api_key_here"

# 系统配置
DEBUG_MODE = False
MAX_FILE_SIZE = 100  # MB
CACHE_EXPIRY = 7  # days
"""
    
    config_file = Path(__file__).parent / "config.env"
    
    if not config_file.exists():
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_template)
        print(f"  ✅ 创建配置文件: {config_file}")
    else:
        print(f"  ℹ️ 配置文件已存在: {config_file}")
    
    return True

def show_installation_summary():
    """显示安装总结"""
    print("\n" + "=" * 60)
    print("🎉 智能果园检测系统安装完成!")
    print("=" * 60)
    
    print("\n📋 安装总结:")
    print("  ✅ Python依赖包已安装")
    print("  ✅ 目录结构已创建")
    print("  ✅ 配置文件已生成")
    
    print("\n🚀 启动系统:")
    print("  方法1: python run.py")
    print("  方法2: streamlit run main.py")
    
    print("\n⚙️ 配置说明:")
    print("  1. 编辑 config/settings.py 配置模型路径")
    print("  2. 编辑 config.env 配置API密钥")
    print("  3. 确保模型文件路径正确")
    
    print("\n📚 功能说明:")
    print("  🍊 果园智能检测 - 基于YOLO的柑橘检测")
    print("  📹 视频内容理解 - 智能视频分析和查询")
    print("  🌱 农业预测建议 - 作物推荐和产量预测")
    print("  👤 用户认证系统 - 支持人脸识别登录")
    print("  💾 数据管理 - 完整的数据存储和导出")
    
    print("\n⚠️ 注意事项:")
    print("  • 确保有足够的磁盘空间存储图像和视频")
    print("  • 人脸识别功能需要摄像头支持")
    print("  • GPU加速需要CUDA环境")
    print("  • 部分功能需要网络连接")
    
    print("\n🆘 获取帮助:")
    print("  • 查看 README.md 了解详细使用说明")
    print("  • 检查 requirements.txt 确认依赖版本")
    print("  • 运行 python run.py 进行系统检查")
    
    print("\n" + "=" * 60)

def main():
    """主函数"""
    print("🍊 智能果园检测系统 - 安装程序")
    print("=" * 50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python版本: {sys.version}")
    
    # 询问是否继续安装
    try:
        response = input("\n是否继续安装? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '是', '']:
            print("👋 安装已取消")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\n👋 安装已取消")
        sys.exit(0)
    
    print("\n🚀 开始安装...")
    
    # 安装依赖包
    if not install_dependencies():
        print("❌ 依赖包安装失败")
        sys.exit(1)
    
    # 设置环境
    if not setup_environment():
        print("❌ 环境设置失败")
        sys.exit(1)
    
    # 检查可选依赖
    check_optional_dependencies()
    
    # 创建配置模板
    create_config_template()
    
    # 显示安装总结
    show_installation_summary()

if __name__ == "__main__":
    main()
