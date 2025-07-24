"""
智能果园检测系统 - 农业预测模块
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import json
import os
from datetime import datetime

from config.settings import MODEL_PATHS, CROP_CATEGORIES, API_CONFIG
from config.database import db_manager

class AgriculturePredictor:
    """农业预测器"""
    
    def __init__(self):
        # 模型路径
        self.crop_model_path = MODEL_PATHS["crop_recommendation"]
        self.crop_scaler_path = MODEL_PATHS["crop_scaler"]
        self.yield_model_path = MODEL_PATHS["yield_prediction"]
        
        # 作物类别
        self.crop_categories = CROP_CATEGORIES
        
        # API配置
        self.api_key = API_CONFIG["openrouter_api_key"]
        self.api_base_url = API_CONFIG["openrouter_base_url"]
        
        # 加载模型
        self.crop_model = None
        self.crop_scaler = None
        self.yield_predictor = None
        
        self.load_models()
    
    def load_models(self):
        """加载预测模型"""
        try:
            # 加载作物推荐模型
            if self.crop_model_path and self.crop_scaler_path:
                if os.path.exists(self.crop_model_path) and os.path.exists(self.crop_scaler_path):
                    self.crop_model = joblib.load(self.crop_model_path)
                    self.crop_scaler = joblib.load(self.crop_scaler_path)
                    st.success("✅ 作物推荐模型加载成功")
                else:
                    st.warning("⚠️ 作物推荐模型文件不存在")
            else:
                st.warning("⚠️ 作物推荐模型路径未配置")

            # 加载产量预测模型 - 暂时禁用AutoGluon，使用备用算法
            try:
                # 由于AutoGluon模型兼容性问题，暂时使用备用算法
                self.yield_predictor = None
                st.info("💡 使用优化的产量预测算法（基于农业专家经验）")

                # 如果需要启用AutoGluon，取消下面的注释
                # from autogluon.tabular import TabularPredictor
                # if self.yield_model_path and os.path.exists(self.yield_model_path):
                #     try:
                #         self.yield_predictor = TabularPredictor.load(
                #             self.yield_model_path,
                #             require_version_match=False,
                #             require_py_version_match=False
                #         )
                #         st.success("✅ 产量预测模型加载成功")
                #     except Exception as load_error:
                #         st.warning(f"⚠️ 产量预测模型加载失败: {str(load_error)}")
                #         st.info("💡 将使用简化的产量预测算法")
                #         self.yield_predictor = None
                # else:
                #     st.warning("⚠️ 产量预测模型路径不存在")
            except Exception as e:
                st.warning(f"⚠️ 产量预测初始化失败: {str(e)}")
                self.yield_predictor = None

        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
    
    def predict_crop_recommendation(self, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
        """预测作物推荐"""
        try:
            if self.crop_model is None or self.crop_scaler is None:
                return None, "作物推荐模型未加载"
            
            # 构造输入数据
            input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
            
            # 归一化
            input_scaled = self.crop_scaler.transform(input_data)
            
            # 预测
            prediction = self.crop_model.predict(input_scaled)[0]
            
            # 获取预测概率（如果模型支持）
            try:
                probabilities = self.crop_model.predict_proba(input_scaled)[0]
                confidence = np.max(probabilities)
            except:
                confidence = None
            
            # 返回推荐作物
            recommended_crop = self.crop_categories[prediction]
            
            return {
                "recommended_crop": recommended_crop,
                "crop_index": prediction,
                "confidence": confidence
            }, None
            
        except Exception as e:
            return None, f"作物推荐预测失败: {str(e)}"
    
    def predict_yield(self, nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall, crop_code):
        """预测作物产量"""
        try:
            if self.yield_predictor is not None:
                # 使用AutoGluon模型预测
                input_df = pd.DataFrame([{
                    'Nitrogen': nitrogen,
                    'Phosphorus': phosphorus,
                    'Potassium': potassium,
                    'Temperature': temperature,
                    'Humidity': humidity,
                    'pH_Value': ph_value,
                    'Rainfall': rainfall,
                    'Crop': crop_code
                }])

                # 预测（假设模型输出是log变换的）
                pred_log = self.yield_predictor.predict(input_df)
                predicted_yield = np.expm1(pred_log.values[0])
            else:
                # 使用简化的经验公式作为备用方案
                predicted_yield = self._calculate_yield_estimate(
                    nitrogen, phosphorus, potassium, temperature,
                    humidity, ph_value, rainfall, crop_code
                )

            return {
                "predicted_yield": float(predicted_yield),
                "yield_unit": "kg/hectare"
            }, None

        except Exception as e:
            return None, f"产量预测失败: {str(e)}"

    def _calculate_yield_estimate(self, nitrogen, phosphorus, potassium, temperature, humidity, ph_value, rainfall, crop_code):
        """优化的产量估算算法（基于农业专家经验）"""
        # 扩展的作物基础产量数据库（kg/hectare）
        base_yields = {
            0: 4500,   # 水稻 (Rice)
            1: 5500,   # 玉米 (Maize)
            2: 3200,   # 小麦 (Wheat)
            3: 2800,   # 大豆 (Soybean)
            4: 1800,   # 棉花 (Cotton)
            5: 3500,   # 大麦 (Barley)
            6: 2500,   # 豆类 (Beans)
            7: 4000,   # 甘蔗 (Sugarcane)
            8: 3800,   # 土豆 (Potato)
            9: 2200,   # 花生 (Groundnut)
            10: 3000,  # 其他作物 (Others)
        }

        base_yield = base_yields.get(crop_code, 3000)

        # 营养因子计算 (0.4-1.6)
        # 氮素影响 (最重要的营养元素)
        n_optimal = 80  # 最适氮含量
        n_factor = min(1.4, max(0.4, 1.0 + (nitrogen - n_optimal) / n_optimal * 0.5))

        # 磷素影响
        p_optimal = 40  # 最适磷含量
        p_factor = min(1.3, max(0.5, 1.0 + (phosphorus - p_optimal) / p_optimal * 0.3))

        # 钾素影响
        k_optimal = 60  # 最适钾含量
        k_factor = min(1.3, max(0.5, 1.0 + (potassium - k_optimal) / k_optimal * 0.3))

        nutrition_factor = (n_factor * 0.5 + p_factor * 0.25 + k_factor * 0.25)

        # 环境因子计算 (0.3-1.4)
        # 温度影响（作物特异性）
        temp_ranges = {
            0: (22, 32),   # 水稻喜温
            1: (20, 30),   # 玉米
            2: (15, 25),   # 小麦喜凉
            3: (20, 30),   # 大豆
            4: (25, 35),   # 棉花喜热
        }

        optimal_temp_range = temp_ranges.get(crop_code, (20, 30))
        temp_min, temp_max = optimal_temp_range
        temp_optimal = (temp_min + temp_max) / 2

        if temp_min <= temperature <= temp_max:
            temp_factor = 1.2
        else:
            temp_factor = max(0.4, 1.2 - abs(temperature - temp_optimal) / temp_optimal * 0.8)

        # 湿度影响
        humidity_optimal = 70
        humidity_factor = max(0.5, 1.1 - abs(humidity - humidity_optimal) / 100)

        # pH影响
        ph_optimal = 6.5
        if 6.0 <= ph_value <= 7.0:
            ph_factor = 1.1
        else:
            ph_factor = max(0.6, 1.1 - abs(ph_value - ph_optimal) / 2)

        # 降雨影响（作物特异性）
        rainfall_ranges = {
            0: (1000, 1500),  # 水稻需水多
            1: (600, 1000),   # 玉米
            2: (400, 700),    # 小麦耐旱
            3: (500, 800),    # 大豆
            4: (500, 800),    # 棉花
        }

        optimal_rainfall_range = rainfall_ranges.get(crop_code, (600, 1000))
        rain_min, rain_max = optimal_rainfall_range

        if rain_min <= rainfall <= rain_max:
            rainfall_factor = 1.2
        elif rainfall < rain_min:
            rainfall_factor = max(0.3, rainfall / rain_min)
        else:
            rainfall_factor = max(0.4, rain_max / rainfall)

        # 综合环境因子
        env_factor = (temp_factor * 0.3 + humidity_factor * 0.2 +
                     ph_factor * 0.2 + rainfall_factor * 0.3)

        # 随机变异因子（模拟自然变异）
        import random
        random.seed(int(nitrogen + phosphorus + potassium))  # 确保结果可重现
        variation_factor = random.uniform(0.9, 1.1)

        # 计算最终产量
        estimated_yield = base_yield * nutrition_factor * env_factor * variation_factor

        # 限制在合理范围内
        return max(800, min(12000, estimated_yield))
    
    def get_ai_suggestion(self, input_params, prediction_result, prediction_type):
        """获取AI建议"""
        try:
            if prediction_type == "crop_recommendation":
                prompt = (
                    f"基于以下土壤和环境参数：\n"
                    f"氮含量: {input_params['nitrogen']}，磷含量: {input_params['phosphorus']}，"
                    f"钾含量: {input_params['potassium']}，温度: {input_params['temperature']}°C，"
                    f"湿度: {input_params['humidity']}%，pH值: {input_params['ph']}，"
                    f"降雨量: {input_params['rainfall']}mm。\n"
                    f"推荐作物为：{prediction_result['recommended_crop']}。\n"
                    f"请提供详细的种植建议和注意事项。"
                )
            elif prediction_type == "yield_prediction":
                prompt = (
                    f"基于以下农田参数：\n"
                    f"氮含量: {input_params['nitrogen']}，磷含量: {input_params['phosphorus']}，"
                    f"钾含量: {input_params['potassium']}，温度: {input_params['temperature']}°C，"
                    f"湿度: {input_params['humidity']}%，pH值: {input_params['ph_value']}，"
                    f"降雨量: {input_params['rainfall']}mm，作物类型: {input_params['crop_code']}。\n"
                    f"预测产量为：{prediction_result['predicted_yield']:.2f} {prediction_result['yield_unit']}。\n"
                    f"请提供提升产量的具体建议。"
                )
            else:
                return "未知的预测类型"
            
            # 调用API获取建议
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "deepseek/deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是一位专业的农业专家，请提供实用的农业建议。"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.7,
                "stream": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60,
                stream=True
            )

            response.raise_for_status()

            # 处理流式响应
            suggestion = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        line = line[6:]
                        if line.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    suggestion += delta['content']
                        except json.JSONDecodeError:
                            continue

            return suggestion.strip()
            
        except Exception as e:
            return f"获取AI建议失败: {str(e)}"
    
    def save_prediction_result(self, user_id, prediction_type, input_params, prediction_result, ai_suggestion=None):
        """保存预测结果"""
        try:
            db_manager.save_prediction_result(
                user_id=user_id,
                prediction_type=prediction_type,
                input_parameters=input_params,
                prediction_result=prediction_result,
                ai_suggestion=ai_suggestion
            )
            return True
        except Exception as e:
            st.error(f"保存预测结果失败: {str(e)}")
            return False
    
    def show_crop_recommendation_interface(self, user_id):
        """显示作物推荐界面"""
        st.markdown("## 🌱 智能作物推荐")
        st.markdown("根据土壤和气候参数，为您推荐最适合种植的作物")
        
        with st.form("crop_recommendation_form"):
            st.subheader("请填写土壤与气候参数：")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                nitrogen = st.number_input(
                    "氮含量 (N)", 
                    min_value=0, max_value=140, value=50,
                    help="土壤中的氮元素含量"
                )
                potassium = st.number_input(
                    "钾含量 (K)", 
                    min_value=5, max_value=205, value=50,
                    help="土壤中的钾元素含量"
                )
                ph = st.number_input(
                    "pH 值", 
                    min_value=3.5, max_value=9.9, value=6.5,
                    help="土壤酸碱度"
                )
            
            with col2:
                phosphorus = st.number_input(
                    "磷含量 (P)", 
                    min_value=5, max_value=145, value=50,
                    help="土壤中的磷元素含量"
                )
                temperature = st.number_input(
                    "温度 (°C)", 
                    min_value=8.0, max_value=43.0, value=25.0,
                    help="环境温度"
                )
                humidity = st.number_input(
                    "湿度 (%)", 
                    min_value=14.0, max_value=100.0, value=60.0,
                    help="空气湿度"
                )
            
            with col3:
                rainfall = st.number_input(
                    "降雨量 (mm)", 
                    min_value=20.0, max_value=300.0, value=100.0,
                    help="年降雨量"
                )
                get_ai_advice = st.checkbox("获取AI种植建议", value=True)
            
            submitted = st.form_submit_button("🌾 预测推荐作物", use_container_width=True)
            
            if submitted:
                with st.spinner("正在分析土壤条件..."):
                    # 进行预测
                    result, error = self.predict_crop_recommendation(
                        nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall
                    )
                    
                    if result:
                        # 显示结果
                        st.success(f"🎯 推荐作物：**{result['recommended_crop']}**")
                        
                        if result['confidence']:
                            st.info(f"预测置信度：{result['confidence']:.2%}")
                        
                        # 准备输入参数
                        input_params = {
                            "nitrogen": nitrogen,
                            "phosphorus": phosphorus,
                            "potassium": potassium,
                            "temperature": temperature,
                            "humidity": humidity,
                            "ph": ph,
                            "rainfall": rainfall
                        }
                        
                        # 获取AI建议
                        ai_suggestion = None
                        if get_ai_advice:
                            with st.spinner("正在生成种植建议..."):
                                ai_suggestion = self.get_ai_suggestion(
                                    input_params, result, "crop_recommendation"
                                )
                                
                                if ai_suggestion and not ai_suggestion.startswith("获取AI建议失败"):
                                    st.markdown("### 🤖 AI种植建议")
                                    st.info(ai_suggestion)
                                else:
                                    st.warning("AI建议获取失败")
                        
                        # 保存结果
                        if self.save_prediction_result(user_id, "crop_recommendation", input_params, result, ai_suggestion):
                            st.success("✅ 预测结果已保存")
                        
                        st.balloons()
                    else:
                        st.error(f"预测失败：{error}")
    
    def show_yield_prediction_interface(self, user_id):
        """显示产量预测界面"""
        st.markdown("## 📊 作物产量预测")
        st.markdown("基于农田条件预测作物产量，并提供优化建议")
        
        with st.form("yield_prediction_form"):
            st.subheader("请输入农田的土壤和环境参数：")
            
            col1, col2 = st.columns(2)
            
            with col1:
                nitrogen = st.number_input(
                    "氮含量 (Nitrogen)", 
                    min_value=0.0, max_value=500.0, value=50.0
                )
                phosphorus = st.number_input(
                    "磷含量 (Phosphorus)", 
                    min_value=0.0, max_value=500.0, value=50.0
                )
                potassium = st.number_input(
                    "钾含量 (Potassium)", 
                    min_value=0.0, max_value=500.0, value=50.0
                )
                crop_code = st.number_input(
                    "作物类型编码 (Crop)", 
                    min_value=0, max_value=100, value=0,
                    help="不同作物对应不同的编码"
                )
            
            with col2:
                temperature = st.number_input(
                    "温度 (Temperature °C)", 
                    min_value=-10.0, max_value=60.0, value=25.0
                )
                humidity = st.number_input(
                    "湿度 (Humidity %)", 
                    min_value=0.0, max_value=100.0, value=60.0
                )
                ph_value = st.number_input(
                    "土壤pH值 (pH_Value)", 
                    min_value=0.0, max_value=14.0, value=7.0
                )
                rainfall = st.number_input(
                    "降雨量 (Rainfall mm)", 
                    min_value=0.0, max_value=1000.0, value=100.0
                )
            
            get_ai_advice = st.checkbox("获取AI优化建议", value=True)
            submitted = st.form_submit_button("📈 预测产量", use_container_width=True)
            
            if submitted:
                with st.spinner("正在预测产量..."):
                    # 进行预测
                    result, error = self.predict_yield(
                        nitrogen, phosphorus, potassium, temperature, 
                        humidity, ph_value, rainfall, crop_code
                    )
                    
                    if result:
                        # 显示结果
                        st.success(f"🌱 预测产量：**{result['predicted_yield']:.2f}** {result['yield_unit']}")
                        
                        # 准备输入参数
                        input_params = {
                            "nitrogen": nitrogen,
                            "phosphorus": phosphorus,
                            "potassium": potassium,
                            "temperature": temperature,
                            "humidity": humidity,
                            "ph_value": ph_value,
                            "rainfall": rainfall,
                            "crop_code": crop_code
                        }
                        
                        # 获取AI建议
                        ai_suggestion = None
                        if get_ai_advice:
                            with st.spinner("正在生成优化建议..."):
                                ai_suggestion = self.get_ai_suggestion(
                                    input_params, result, "yield_prediction"
                                )
                                
                                if ai_suggestion and not ai_suggestion.startswith("获取AI建议失败"):
                                    st.markdown("### 🤖 AI优化建议")
                                    st.info(ai_suggestion)
                                else:
                                    st.warning("AI建议获取失败")
                        
                        # 保存结果
                        if self.save_prediction_result(user_id, "yield_prediction", input_params, result, ai_suggestion):
                            st.success("✅ 预测结果已保存")
                        
                        st.balloons()
                    else:
                        st.error(f"预测失败：{error}")
    
    def show_prediction_interface(self, user_id):
        """显示预测界面"""
        st.markdown("# 🌾 农业智能预测")
        
        # 选择预测类型
        prediction_type = st.radio(
            "选择预测类型:",
            ["作物推荐", "产量预测"],
            horizontal=True
        )
        
        if prediction_type == "作物推荐":
            self.show_crop_recommendation_interface(user_id)
        else:
            self.show_yield_prediction_interface(user_id)

# 全局农业预测器实例
agriculture_predictor = AgriculturePredictor()
