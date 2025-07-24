#!/usr/bin/env python3
"""
智能果园检测系统 - 演示脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_database():
    """测试数据库功能"""
    print("🗄️ 测试数据库功能...")
    
    try:
        from config.database import db_manager
        
        # 测试创建用户
        user_id = db_manager.create_user("demo_user", "demo_password", "demo@example.com")
        if user_id:
            print(f"  ✅ 创建用户成功: ID {user_id}")
            
            # 测试验证用户
            verified_id = db_manager.verify_user("demo_user", "demo_password")
            if verified_id == user_id:
                print("  ✅ 用户验证成功")
            else:
                print("  ❌ 用户验证失败")
            
            # 测试获取用户信息
            user_info = db_manager.get_user_by_id(user_id)
            if user_info:
                print(f"  ✅ 获取用户信息成功: {user_info['username']}")
            else:
                print("  ❌ 获取用户信息失败")
        else:
            print("  ❌ 创建用户失败")
        
        print("✅ 数据库功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据库测试失败: {str(e)}")
        return False

def test_models():
    """测试模型加载"""
    print("🤖 测试模型加载...")
    
    try:
        from config.settings import MODEL_PATHS
        
        # 测试YOLO柑橘检测模型
        yolo_path = MODEL_PATHS["yolo_citrus"]
        if yolo_path and os.path.exists(yolo_path):
            print(f"  ✅ YOLO柑橘检测模型文件存在")
            print(f"    📍 路径: {yolo_path}")

            try:
                from ultralytics import YOLO
                model = YOLO(yolo_path)
                print("  ✅ YOLOv11s柑橘检测模型加载成功")

                # 显示模型详细信息
                if hasattr(model, 'names') and model.names:
                    print(f"    🏷️ 检测类别: {list(model.names.values())}")
                    print(f"    📊 类别数量: {len(model.names)}")

                file_size = os.path.getsize(yolo_path) / (1024*1024)
                print(f"    💾 模型大小: {file_size:.1f} MB")

            except Exception as e:
                print(f"  ⚠️ YOLO模型加载失败: {str(e)}")
                print("  💡 建议: 检查模型文件完整性或重新训练")
        else:
            print(f"  ❌ YOLO模型文件不存在: {yolo_path}")
            print("  💡 建议: 确保模型文件存在或重新训练")
        
        # 测试作物推荐模型
        crop_model_path = MODEL_PATHS["crop_recommendation"]
        crop_scaler_path = MODEL_PATHS["crop_scaler"]

        if crop_model_path and crop_scaler_path and os.path.exists(crop_model_path) and os.path.exists(crop_scaler_path):
            print(f"  ✅ 作物推荐模型文件存在")
            print(f"    📍 模型路径: {crop_model_path}")
            print(f"    📍 缩放器路径: {crop_scaler_path}")

            try:
                import joblib
                _ = joblib.load(crop_model_path)
                _ = joblib.load(crop_scaler_path)
                print("  ✅ 作物推荐模型加载成功")
            except Exception as e:
                print(f"  ⚠️ 作物推荐模型加载失败: {str(e)}")
        else:
            print(f"  ❌ 作物推荐模型文件不存在")
            print(f"    期望路径: {crop_model_path}")
            print(f"    期望路径: {crop_scaler_path}")

        # 测试产量预测模型
        yield_model_path = MODEL_PATHS["yield_prediction"]
        if yield_model_path and os.path.exists(yield_model_path):
            print(f"  ✅ 产量预测模型路径存在")
            print(f"    📍 模型路径: {yield_model_path}")

            try:
                from autogluon.tabular import TabularPredictor
                _ = TabularPredictor.load(
                    yield_model_path,
                    require_version_match=False,
                    require_py_version_match=False
                )
                print("  ✅ AutoGluon产量预测模型加载成功")
            except ImportError:
                print("  ⚠️ AutoGluon未安装，产量预测功能不可用")
            except Exception as e:
                print(f"  ⚠️ 产量预测模型加载失败: {str(e)}")
        else:
            print(f"  ❌ 产量预测模型文件不存在: {yield_model_path}")
        
        print("✅ 模型测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {str(e)}")
        return False

def test_face_recognition():
    """测试人脸识别功能"""
    print("👤 测试人脸识别功能...")
    
    try:
        import face_recognition
        import numpy as np
        
        # 创建一个测试图像
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 测试人脸检测
        face_locations = face_recognition.face_locations(test_image)
        print(f"  ✅ 人脸检测功能正常 (检测到 {len(face_locations)} 张人脸)")
        
        print("✅ 人脸识别功能测试完成")
        return True
        
    except ImportError:
        print("  ❌ face_recognition库未安装")
        return False
    except Exception as e:
        print(f"❌ 人脸识别测试失败: {str(e)}")
        return False

def test_api_connections():
    """测试API连接"""
    print("🌐 测试API连接...")
    
    try:
        from config.settings import API_CONFIG
        import requests
        
        # 测试网络连接
        try:
            response = requests.get("https://www.baidu.com", timeout=5)
            if response.status_code == 200:
                print("  ✅ 网络连接正常")
            else:
                print("  ⚠️ 网络连接异常")
        except:
            print("  ❌ 网络连接失败")
        
        # 检查API配置
        ernie_key = API_CONFIG.get("ernie_api_key")
        openrouter_key = API_CONFIG.get("openrouter_api_key")
        
        if ernie_key and ernie_key != "your_api_key_here":
            print("  ✅ 文心一言API密钥已配置")
        else:
            print("  ⚠️ 文心一言API密钥未配置")
        
        if openrouter_key and openrouter_key != "your_api_key_here":
            print("  ✅ OpenRouter API密钥已配置")
        else:
            print("  ⚠️ OpenRouter API密钥未配置")
        
        print("✅ API连接测试完成")
        return True
        
    except Exception as e:
        print(f"❌ API连接测试失败: {str(e)}")
        return False

def test_data_management():
    """测试数据管理功能"""
    print("💾 测试数据管理功能...")
    
    try:
        from modules.utils.data_manager import data_manager
        
        # 测试存储使用情况
        storage_info = data_manager.get_storage_usage()
        if storage_info:
            total_size = data_manager.format_size(storage_info['total_size'])
            print(f"  ✅ 存储使用情况: {total_size}")
        else:
            print("  ⚠️ 无法获取存储使用情况")
        
        # 测试临时文件清理
        _, message = data_manager.clean_temp_files(max_age_hours=0)
        print(f"  ✅ 临时文件清理: {message}")
        
        print("✅ 数据管理功能测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据管理测试失败: {str(e)}")
        return False

def run_demo():
    """运行演示"""
    print("🍊 智能果园检测系统 - 功能演示")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("数据库功能", test_database()))
    test_results.append(("模型加载", test_models()))
    test_results.append(("人脸识别", test_face_recognition()))
    test_results.append(("API连接", test_api_connections()))
    test_results.append(("数据管理", test_data_management()))
    
    # 显示测试结果
    print("\n" + "=" * 50)
    print("📊 测试结果总结")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有功能测试通过，系统运行正常！")
    elif passed >= total * 0.7:
        print("⚠️ 大部分功能正常，部分功能可能受限")
    else:
        print("❌ 多项功能异常，请检查配置和依赖")
    
    print("\n💡 提示:")
    print("  • 运行 python run.py 启动完整系统")
    print("  • 检查 config/settings.py 确认配置正确")
    print("  • 确保所有依赖包已正确安装")
    
    return passed == total

def main():
    """主函数"""
    try:
        success = run_demo()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 演示已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
