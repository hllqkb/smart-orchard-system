#!/usr/bin/env python3
"""
智能果园检测系统 - 启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version}")
    return True

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    required_packages = [
        'streamlit',
        'opencv-python',
        'torch',
        'ultralytics',
        'transformers',
        'face-recognition',
        'plotly',
        'pandas',
        'numpy',
        'pillow',
        'requests',
        'bcrypt',
        'langchain',
        'faiss-cpu',
        'xgboost',
        'scikit-learn',
        'torchvision',
        'matplotlib',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ 所有依赖包检查通过")
    return True

def check_model_files():
    """检查模型文件"""
    print("🔍 检查模型文件...")
    
    model_paths = {
        "YOLO柑橘检测模型": "/home/hllqk/projects/smart-orchard-system/model/best-baseed-yolov11s.pt",
        "作物推荐模型": "/home/hllqk/projects/smart-orchard-system/model/best_xgb_model.pkl",
        "作物推荐缩放器": "/home/hllqk/projects/smart-orchard-system/model/scaler.pkl",
        "产量预测模型": "/home/hllqk/projects/smart-orchard-system/model/AutogluonModels/ag-20250703_165505"
    }
    
    missing_models = []
    
    for name, path in model_paths.items():
        if os.path.exists(path):
            print(f"  ✅ {name}")

            # 对YOLO模型进行额外检查
            if "YOLO" in name:
                try:
                    from ultralytics import YOLO
                    model = YOLO(path)
                    print(f"    📋 模型架构: YOLOv11s (柑橘专用)")
                    if hasattr(model, 'names') and model.names:
                        print(f"    🏷️ 检测类别: {list(model.names.values())}")
                        print(f"    📊 类别数量: {len(model.names)}")
                    print(f"    💾 文件大小: {os.path.getsize(path) / (1024*1024):.1f} MB")
                except Exception as e:
                    print(f"    ⚠️ 模型验证失败: {str(e)}")
        else:
            print(f"  ❌ {name}: {path}")
            missing_models.append(name)
    
    if missing_models:
        print(f"\n⚠️ 缺少以下模型文件: {', '.join(missing_models)}")
        print("部分功能可能不可用，但系统仍可启动")
    else:
        print("✅ 所有模型文件检查通过")
    
    return True

def setup_directories():
    """设置必要的目录"""
    print("📁 设置目录结构...")
    
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
    
    project_root = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {dir_name}")
    
    print("✅ 目录结构设置完成")
    return True

def initialize_database():
    """初始化数据库"""
    print("🗄️ 初始化数据库...")
    
    try:
        # 导入数据库管理器来初始化数据库
        sys.path.insert(0, str(Path(__file__).parent))
        from config.database import db_manager
        
        # 数据库会在导入时自动初始化
        print("✅ 数据库初始化完成")
        return True
    except Exception as e:
        print(f"❌ 数据库初始化失败: {str(e)}")
        return False

def run_system():
    """运行系统"""
    print("🚀 启动智能果园检测系统...")
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    main_file = project_root / "main.py"
    
    if not main_file.exists():
        print("❌ 错误: 找不到main.py文件")
        return False
    
    try:
        # 使用streamlit运行应用
        cmd = [sys.executable, "-m", "streamlit", "run", str(main_file)]
        
        print(f"执行命令: {' '.join(cmd)}")
        print("=" * 50)
        print("🍊 智能果园检测系统正在启动...")
        print("📱 请在浏览器中访问显示的URL")
        print("🛑 按 Ctrl+C 停止系统")
        print("=" * 50)
        
        # 运行streamlit应用
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 系统已停止")
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    print("🍊 智能果园检测系统 - 启动检查")
    print("=" * 50)

    # 检查Python版本
    if not check_python_version():
        sys.exit(1)

    # 检查依赖包
    if not check_dependencies():
        print("\n💡 提示: 运行以下命令修复依赖问题:")
        print("python setup_dependencies.py")
        print("或手动安装: pip install -r requirements.txt")

        try:
            response = input("\n是否现在自动修复依赖? (y/n): ").lower().strip()
            if response in ['y', 'yes', '是', '']:
                print("🔧 正在修复依赖...")
                import subprocess
                result = subprocess.run([sys.executable, "setup_dependencies.py"],
                                      cwd=Path(__file__).parent)
                if result.returncode == 0:
                    print("✅ 依赖修复完成，请重新运行启动脚本")
                else:
                    print("❌ 依赖修复失败，请手动安装")
                sys.exit(result.returncode)
            else:
                print("👋 请先修复依赖问题")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n👋 启动已取消")
            sys.exit(1)

    # 检查模型文件
    check_model_files()

    # 设置目录
    if not setup_directories():
        sys.exit(1)

    # 初始化数据库
    if not initialize_database():
        sys.exit(1)

    print("\n✅ 所有检查完成，系统准备就绪!")
    print("=" * 50)

    # 显示启动选项
    print("\n🚀 启动选项:")
    print("1. 启动完整系统 (推荐)")
    print("2. 运行功能演示")
    print("3. 仅检查系统状态")
    print("4. 退出")

    try:
        choice = input("\n请选择 (1-4): ").strip()

        if choice == '1' or choice == '':
            run_system()
        elif choice == '2':
            print("🎯 运行功能演示...")
            subprocess.run([sys.executable, "demo.py"], cwd=Path(__file__).parent)
        elif choice == '3':
            print("✅ 系统状态检查完成")
        elif choice == '4':
            print("👋 退出")
        else:
            print("❌ 无效选择")

    except KeyboardInterrupt:
        print("\n👋 启动已取消")

if __name__ == "__main__":
    main()
