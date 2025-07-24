"""
智能果园检测系统 - 视频内容理解模块
集成Visual-Vision-RAG功能
"""

import streamlit as st
import subprocess
import sys
import os
from pathlib import Path

from config.settings import API_CONFIG, VIDEO_CONFIG
from config.database import db_manager

class VideoAnalyzer:
    def __init__(self):
        self.video_rag_path = "/home/hllqk/projects/Visual-Vision-RAG/app.py"

    def show_video_analysis_interface(self, user_id):
        """显示视频分析界面"""
        st.markdown("## 📹 视频内容理解")

        # 检查Visual-Vision-RAG是否存在
        if not os.path.exists(self.video_rag_path):
            st.error("❌ Visual-Vision-RAG应用未找到")
            st.info(f"期望路径: {self.video_rag_path}")
            return

        st.success("✅ 使用专业的Visual-Vision-RAG系统进行视频分析")

        # 显示功能介绍
        st.markdown("""
        ### 🎯 功能特点
        - **智能视频分析**: 自动提取关键帧并进行内容分析
        - **自然语言查询**: 支持中文问答式视频内容检索
        - **事件时间线**: 生成视频事件摘要和时间轴
        - **RAG技术**: 基于向量数据库的智能检索
        """)

        # 启动选项
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 启动视频分析系统", type="primary"):
                self.launch_video_rag_system()

        with col2:
            if st.button("📖 查看使用说明"):
                self.show_usage_instructions()

        # 显示最近的分析历史
        self.show_recent_analysis_history(user_id)

    def launch_video_rag_system(self):
        """启动Visual-Vision-RAG系统"""
        try:
            st.info("🚀 正在启动Visual-Vision-RAG系统...")

            # 在后台启动Visual-Vision-RAG系统
            import subprocess
            import threading

            def start_video_rag():
                try:
                    # 启动命令
                    cmd = [
                        "/home/hllqk/miniconda3/envs/deeplearn/bin/python",
                        "-m", "streamlit", "run", "app.py",
                        "--server.port", "8502",
                        "--server.headless", "true"
                    ]

                    # 在Visual-Vision-RAG目录中启动
                    subprocess.Popen(
                        cmd,
                        cwd="/home/hllqk/projects/Visual-Vision-RAG",
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                except Exception as e:
                    print(f"启动Visual-Vision-RAG失败: {str(e)}")

            # 在后台线程中启动
            thread = threading.Thread(target=start_video_rag)
            thread.daemon = True
            thread.start()

            st.success("✅ Visual-Vision-RAG系统正在启动...")
            st.info("📱 请稍等几秒钟，然后访问: http://localhost:8502")

            # 添加直接链接
            st.markdown("""
            ### 🔗 快速访问
            [点击这里打开视频分析系统](http://localhost:8502)

            如果链接无法打开，请手动访问: http://localhost:8502
            """)

        except Exception as e:
            st.error(f"❌ 启动失败: {str(e)}")
            st.markdown("""
            ### 🔧 手动启动方式
            请在终端中运行以下命令：
            ```bash
            cd /home/hllqk/projects/Visual-Vision-RAG
            /home/hllqk/miniconda3/envs/deeplearn/bin/python -m streamlit run app.py --server.port 8502
            ```
            """)

    def show_usage_instructions(self):
        """显示使用说明"""
        st.markdown("""
        ### 📖 使用说明

        #### 1. 视频上传
        - 支持MP4、AVI、MOV等常见视频格式
        - 建议视频大小不超过100MB
        - 系统会自动提取关键帧进行分析

        #### 2. 内容分析
        - 自动识别视频中的物体、场景、动作
        - 生成详细的内容描述和标签
        - 构建时间轴和事件序列

        #### 3. 智能问答
        - 使用自然语言询问视频内容
        - 例如："视频中出现了什么水果？"
        - 例如："第5分钟发生了什么？"

        #### 4. 结果导出
        - 支持导出分析结果为JSON格式
        - 可保存关键帧图像
        - 生成视频摘要报告
        """)

    def show_recent_analysis_history(self, user_id):
        """显示最近的分析历史"""
        st.markdown("### 📊 最近分析历史")

        try:
            # 获取用户的视频分析历史
            history = db_manager.get_user_history(user_id, "video_analysis", limit=5)

            if history:
                for record in history:
                    with st.expander(f"📹 {record.get('video_name', '未知视频')} - {record.get('created_at', '')}"):
                        st.write(f"**分析时间**: {record.get('created_at', '')}")
                        st.write(f"**视频名称**: {record.get('video_name', '未知')}")
                        if record.get('analysis_result'):
                            st.write("**分析结果**:")
                            st.json(record['analysis_result'])
            else:
                st.info("暂无分析历史")

        except Exception as e:
            st.warning(f"获取历史记录失败: {str(e)}")
