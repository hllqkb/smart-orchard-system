"""
智能果园检测系统 - 柑橘检测模块
"""

import streamlit as st
import cv2
import numpy as np
import time
import torch
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from ultralytics import YOLO
import tempfile
import os

from config.settings import MODEL_PATHS, DETECTION_CONFIG, CITRUS_CATEGORIES, MODEL_INFO
from config.database import db_manager

class CitrusDetector:
    """柑橘检测器"""
    
    def __init__(self):
        self.model = None
        self.model_path = MODEL_PATHS["yolo_citrus"]
        self.confidence_threshold = DETECTION_CONFIG["confidence_threshold"]
        self.iou_threshold = DETECTION_CONFIG["iou_threshold"]
        self.supported_formats = DETECTION_CONFIG["supported_formats"]
        self.max_file_size = DETECTION_CONFIG["max_file_size"]
        self.categories = CITRUS_CATEGORIES
        
        # 加载模型
        self.load_model()
    
    @st.cache_resource
    def load_model(_self):
        """加载YOLO柑橘检测模型"""
        try:
            if _self.model_path and os.path.exists(_self.model_path):
                model = YOLO(_self.model_path)

                # 打印模型信息用于调试
                print(f"模型类别名称: {model.names}")
                print(f"配置的类别: {_self.categories}")

                return model
            else:
                st.warning("⚠️ 柑橘检测模型文件不存在，请检查模型路径")
                return None
        except Exception as e:
            st.error(f"模型加载失败: {str(e)}")
            return None
    
    def detect_objects(self, image, confidence_threshold=None, iou_threshold=None):
        """检测图像中的柑橘"""
        if self.model is None:
            self.model = self.load_model()
            if self.model is None:
                return None, "模型未加载"
        
        try:
            # 使用指定的阈值或默认值
            conf = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
            iou = iou_threshold if iou_threshold is not None else self.iou_threshold
            
            # 记录开始时间
            start_time = time.perf_counter()
            
            # 进行检测
            results = self.model(
                image,
                conf=conf,
                iou=iou,
                verbose=False
            )
            
            # 记录结束时间
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            
            return results[0], inference_time
            
        except Exception as e:
            return None, f"检测失败: {str(e)}"
    
    def draw_detections(self, image, result, show_confidence=True, show_labels=True, show_boxes=True, line_width=2):
        """绘制检测结果"""
        if result.boxes is None or len(result.boxes) == 0:
            return np.array(image)
        
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 获取检测结果
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        
        # 绘制每个检测框
        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = box.astype(int)
            
            # 获取类别信息
            category_info = self.categories.get(cls, {"color": (128, 128, 128), "display": "未知"})
            color = category_info["color"]
            
            if show_boxes:
                # 绘制边界框
                cv2.rectangle(img_array, (x1, y1), (x2, y2), color, line_width)
            
            if show_labels or show_confidence:
                # 准备标签文本
                label_parts = []
                if show_labels:
                    label_parts.append(category_info["display"])
                if show_confidence:
                    label_parts.append(f"{conf:.2f}")
                
                label = " ".join(label_parts)
                
                # 计算文本尺寸
                font_scale = 0.6
                font_thickness = 2
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                
                # 文本位置
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10
                
                # 绘制文本背景
                cv2.rectangle(
                    img_array,
                    (text_x, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    img_array,
                    label,
                    (text_x + 2, text_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA
                )
        
        return img_array
    
    def analyze_detection_results(self, result):
        """分析检测结果"""
        if result.boxes is None or len(result.boxes) == 0:
            return {
                "total_count": 0,
                "category_counts": {},
                "confidences": [],
                "avg_confidence": 0.0,
                "max_confidence": 0.0,
                "detection_details": []
            }

        # 获取检测结果
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # 统计各类别数量
        category_counts = {}
        detection_details = []

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            # 确保类别索引正确
            category_info = self.categories.get(cls, {"display": f"类别{cls}"})
            category_name = category_info["display"]

            if category_name not in category_counts:
                category_counts[category_name] = 0
            category_counts[category_name] += 1

            # 详细信息 - 转换所有numpy类型为Python原生类型
            detection_details.append({
                "序号": int(i + 1),
                "类别": str(category_name),
                "置信度": f"{float(conf):.2%}",
                "边界框": f"({int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])})",
                "面积": f"{int((box[2] - box[0]) * (box[3] - box[1]))}"
            })

        # 转换所有numpy类型为Python原生类型以支持JSON序列化
        return {
            "total_count": int(len(boxes)),
            "category_counts": category_counts,
            "confidences": [float(c) for c in confidences],
            "avg_confidence": float(np.mean(confidences)),
            "max_confidence": float(np.max(confidences)),
            "detection_details": detection_details
        }
    
    def save_detection_result(self, user_id, image_path, analysis_result, confidence_threshold, inference_time):
        """保存检测结果到数据库"""
        try:
            # 准备保存的数据
            result_data = {
                "analysis": analysis_result,
                "inference_time": inference_time,
                "parameters": {
                    "confidence_threshold": confidence_threshold,
                    "iou_threshold": self.iou_threshold
                }
            }
            
            db_manager.save_detection_result(
                user_id=user_id,
                detection_type="citrus_detection",
                image_path=image_path,
                results=result_data,
                confidence_threshold=confidence_threshold,
                detection_count=analysis_result["total_count"]
            )
            
            return True
        except Exception as e:
            st.error(f"保存检测结果失败: {str(e)}")
            return False
    
    def show_detection_interface(self, user_id):
        """显示检测界面"""
        st.markdown("## 🍊 柑橘智能检测")

        # 显示模型状态
        if self.model is None:
            self.model = self.load_model()

        if self.model is not None:
            st.success("✅ 柑橘检测模型已加载 (YOLOv11s)")
        else:
            st.error("❌ 模型加载失败，请检查模型文件")

        # 检测参数设置
        with st.sidebar:
            st.markdown("### ⚙️ 检测参数")
            confidence_threshold = st.slider(
                "置信度阈值",
                min_value=0.1,
                max_value=1.0,
                value=self.confidence_threshold,
                step=0.05,
                help="只显示置信度高于此阈值的检测结果"
            )
            
            iou_threshold = st.slider(
                "IoU阈值",
                min_value=0.1,
                max_value=1.0,
                value=self.iou_threshold,
                step=0.05,
                help="非极大值抑制的IoU阈值"
            )
            
            # 显示选项
            st.markdown("### 🎨 显示选项")
            show_confidence = st.checkbox("显示置信度", value=True)
            show_labels = st.checkbox("显示标签", value=True)
            show_boxes = st.checkbox("显示边界框", value=True)
            line_width = st.slider("边界框线条宽度", min_value=1, max_value=5, value=2)
        
        # 检测模式选择
        detection_mode = st.radio(
            "选择检测模式:",
            ["单张图像检测", "批量图像检测"],
            horizontal=True
        )
        
        if detection_mode == "单张图像检测":
            self.show_single_image_detection(
                user_id, confidence_threshold, iou_threshold,
                show_confidence, show_labels, show_boxes, line_width
            )
        else:
            self.show_batch_image_detection(
                user_id, confidence_threshold, iou_threshold,
                show_confidence, show_labels, show_boxes, line_width
            )
    
    def show_single_image_detection(self, user_id, confidence_threshold, iou_threshold,
                                  show_confidence, show_labels, show_boxes, line_width):
        """显示单张图像检测界面"""
        st.markdown("### 🖼️ 单张图像检测")
        
        # 图像上传
        uploaded_file = st.file_uploader(
            "选择要检测的图像",
            type=self.supported_formats,
            help=f"支持格式: {', '.join(self.supported_formats).upper()}"
        )
        
        if uploaded_file is not None:
            # 检查文件大小
            if uploaded_file.size > self.max_file_size:
                st.error(f"文件大小超过限制 ({self.max_file_size / (1024*1024):.1f}MB)")
                return
            
            # 显示原始图像
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📷 原始图像")
                st.image(image, caption="上传的图像", use_container_width=True)
            
            # 检测按钮
            if st.button("🚀 开始检测", use_container_width=True):
                with st.spinner("🔄 正在进行检测..."):
                    # 进行检测
                    result, inference_time = self.detect_objects(image, confidence_threshold, iou_threshold)
                    
                    if result is not None and not isinstance(result, str):
                        with col2:
                            st.markdown("#### 🎯 检测结果")
                            
                            # 绘制检测结果
                            annotated_image = self.draw_detections(
                                image, result, show_confidence, show_labels, show_boxes, line_width
                            )
                            
                            # 显示结果图像
                            annotated_image_pil = Image.fromarray(annotated_image)
                            st.image(annotated_image_pil, caption="检测结果", use_container_width=True)
                        
                        # 分析结果
                        analysis = self.analyze_detection_results(result)
                        
                        # 显示统计信息
                        self.show_detection_statistics(analysis, inference_time)
                        
                        # 保存结果
                        if self.save_detection_result(user_id, uploaded_file.name, analysis, confidence_threshold, inference_time):
                            st.success("✅ 检测结果已保存")
                    
                    else:
                        st.error(f"检测失败: {result if isinstance(result, str) else '未知错误'}")
        else:
            st.info("请上传图像文件开始检测")
    
    def show_batch_image_detection(self, user_id, confidence_threshold, iou_threshold,
                                 show_confidence, show_labels, show_boxes, line_width):
        """显示批量图像检测界面"""
        st.markdown("### 📁 批量图像检测")
        
        # 批量上传
        uploaded_files = st.file_uploader(
            "选择多张图像进行批量检测",
            type=self.supported_formats,
            accept_multiple_files=True,
            help="可以同时选择多张图像进行批量检测"
        )
        
        if uploaded_files:
            st.success(f"✅ 已选择 {len(uploaded_files)} 张图像")
            
            # 批量检测按钮
            if st.button("🚀 开始批量检测", use_container_width=True):
                # 创建进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 存储所有结果
                all_results = []
                total_time = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # 更新进度
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"正在处理: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
                    
                    # 检查文件大小
                    if uploaded_file.size > self.max_file_size:
                        continue
                    
                    # 处理单张图像
                    image = Image.open(uploaded_file)
                    result, inference_time = self.detect_objects(image, confidence_threshold, iou_threshold)
                    
                    if result is not None and not isinstance(result, str):
                        total_time += inference_time
                        analysis = self.analyze_detection_results(result)
                        
                        all_results.append({
                            "文件名": uploaded_file.name,
                            "检测数量": analysis["total_count"],
                            "平均置信度": f"{analysis['avg_confidence']:.2%}",
                            "处理时间": f"{inference_time:.3f}s",
                            "图像": image,
                            "结果": result,
                            "分析": analysis
                        })
                        
                        # 保存结果
                        self.save_detection_result(user_id, uploaded_file.name, analysis, confidence_threshold, inference_time)
                
                # 完成处理
                progress_bar.progress(1.0)
                status_text.text("✅ 批量检测完成!")
                
                # 显示批量检测统计
                self.show_batch_statistics(all_results, total_time)
        else:
            st.info("请选择图像文件进行批量检测")
    
    def show_detection_statistics(self, analysis, inference_time):
        """显示检测统计信息"""
        st.markdown("## 📈 检测统计")
        
        if analysis["total_count"] > 0:
            # 基本统计
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 检测数量", analysis["total_count"])
            
            with col2:
                st.metric("📊 平均置信度", f"{analysis['avg_confidence']:.2%}")
            
            with col3:
                st.metric("🏆 最高置信度", f"{analysis['max_confidence']:.2%}")
            
            with col4:
                st.metric("⏱️ 检测时间", f"{inference_time:.3f}s")
            
            # 类别分布图表
            if analysis["category_counts"]:
                st.markdown("### 📊 类别分布")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # 饼图
                    fig_pie = px.pie(
                        values=list(analysis["category_counts"].values()),
                        names=list(analysis["category_counts"].keys()),
                        title="检测类别分布"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # 柱状图
                    fig_bar = px.bar(
                        x=list(analysis["category_counts"].keys()),
                        y=list(analysis["category_counts"].values()),
                        title="各类别检测数量"
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
            
            # 详细检测结果表格
            with st.expander("📋 查看详细检测结果"):
                detection_df = pd.DataFrame(analysis["detection_details"])
                st.dataframe(detection_df, use_container_width=True)
        
        else:
            st.info("未检测到任何目标，请尝试降低置信度阈值或使用更清晰的图像")
    
    def show_batch_statistics(self, all_results, total_time):
        """显示批量检测统计"""
        st.markdown("## 📊 批量检测统计")
        
        if all_results:
            col1, col2, col3, col4 = st.columns(4)
            
            total_detections = sum([r["检测数量"] for r in all_results])
            avg_time = total_time / len(all_results)
            successful_detections = len([r for r in all_results if r["检测数量"] > 0])
            
            with col1:
                st.metric("📁 处理图像", len(all_results))
            
            with col2:
                st.metric("🎯 总检测数", total_detections)
            
            with col3:
                st.metric("⏱️ 平均时间", f"{avg_time:.3f}s")
            
            with col4:
                success_rate = successful_detections / len(all_results) * 100
                st.metric("✅ 成功率", f"{success_rate:.1f}%")
            
            # 详细结果表格
            st.markdown("### 📋 详细结果")
            results_df = pd.DataFrame([
                {
                    "文件名": r["文件名"],
                    "检测数量": r["检测数量"],
                    "平均置信度": r["平均置信度"],
                    "处理时间": r["处理时间"]
                } for r in all_results
            ])
            st.dataframe(results_df, use_container_width=True)

# 全局柑橘检测器实例
citrus_detector = CitrusDetector()
