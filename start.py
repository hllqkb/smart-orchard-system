#!/usr/bin/env python3
"""
智能果园检测系统 - 快速启动脚本
"""

import os
import sys
import subprocess
from pathlib import Path

def quick_start():
    """快速启动系统"""
    print("🍊 智能果园检测系统 - 快速启动")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # 简单检查模型文件
    print("🔍 检查模型文件...")

    models_to_check = [
        ("柑橘检测模型", "model/best-baseed-yolov11s.pt"),
        ("作物推荐模型", "model/best_xgb_model.pkl"),
        ("作物缩放器", "model/scaler.pkl"),
        ("产量预测模型", "model/AutogluonModels/ag-20250703_165505")
    ]

    for model_name, model_path in models_to_check:
        full_path = project_root / model_path
        if full_path.exists():
            print(f"✅ {model_name}文件存在")
        else:
            print(f"⚠️ {model_name}文件不存在，部分功能可能受限")

    # 确保数据目录存在
    (project_root / "data").mkdir(exist_ok=True)
    
    # 直接启动Streamlit应用
    print("🚀 启动智能果园检测系统...")
    print("📱 系统将在浏览器中打开")
    print("🛑 按 Ctrl+C 停止系统")
    print("=" * 50)
    
    try:
        main_file = project_root / "main.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(main_file)]
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        print("\n👋 系统已停止")
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        print("💡 请尝试运行: python run.py")

if __name__ == "__main__":
    quick_start()
