#!/usr/bin/env python3
"""
测试修复效果的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_citrus_categories():
    """测试柑橘检测类别配置"""
    print("🍊 测试柑橘检测类别配置...")
    
    from config.settings import CITRUS_CATEGORIES
    
    print(f"类别配置: {CITRUS_CATEGORIES}")
    
    # 检查是否为英文标签
    for category_id, info in CITRUS_CATEGORIES.items():
        display_name = info['display']
        print(f"  类别 {category_id}: {display_name}")
        
        # 检查是否包含中文字符
        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in display_name)
        if has_chinese:
            print(f"    ⚠️ 仍包含中文字符")
        else:
            print(f"    ✅ 英文标签正确")
    
    print("✅ 柑橘检测类别测试完成\n")

def test_yield_prediction():
    """测试产量预测功能"""
    print("📊 测试产量预测功能...")
    
    try:
        from modules.prediction.agriculture_predictor import agriculture_predictor
        
        # 测试参数
        test_params = {
            'nitrogen': 80,
            'phosphorus': 40, 
            'potassium': 60,
            'temperature': 25,
            'humidity': 70,
            'ph_value': 6.5,
            'rainfall': 800,
            'crop_code': 1  # 玉米
        }
        
        print(f"测试参数: {test_params}")
        
        # 执行预测
        result, error = agriculture_predictor.predict_yield(**test_params)
        
        if result:
            print(f"✅ 产量预测成功:")
            print(f"  预测产量: {result['predicted_yield']:.2f} {result['yield_unit']}")
        else:
            print(f"❌ 产量预测失败: {error}")
            
    except Exception as e:
        print(f"❌ 产量预测测试异常: {str(e)}")
    
    print("✅ 产量预测测试完成\n")

def test_json_serialization():
    """测试JSON序列化"""
    print("💾 测试JSON序列化...")
    
    import json
    import numpy as np
    
    # 模拟检测结果数据
    test_data = {
        "total_count": 3,
        "category_counts": {"Citrus Fruit": 2, "Tree Fruit": 1},
        "confidences": [0.85, 0.92, 0.78],
        "avg_confidence": 0.85,
        "max_confidence": 0.92,
        "detection_details": [
            {
                "序号": 1,
                "类别": "Citrus Fruit",
                "置信度": "85.00%",
                "边界框": "(100, 200, 300, 400)",
                "面积": "40000"
            }
        ]
    }
    
    try:
        json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
        print("✅ JSON序列化成功")
        print(f"序列化结果长度: {len(json_str)} 字符")
    except Exception as e:
        print(f"❌ JSON序列化失败: {str(e)}")
    
    print("✅ JSON序列化测试完成\n")

def main():
    """主测试函数"""
    print("🧪 开始测试修复效果")
    print("=" * 50)
    
    test_citrus_categories()
    test_yield_prediction() 
    test_json_serialization()
    
    print("=" * 50)
    print("🎉 所有测试完成！")

if __name__ == "__main__":
    main()
