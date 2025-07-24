"""
智能果园检测系统 - 主应用程序
"""

import streamlit as st
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置
from config.settings import UI_CONFIG, DEBUG_CONFIG
from config.database import db_manager

# 导入模块
from modules.auth.auth_manager import auth_manager
from modules.auth.face_recognition_manager import face_manager
from modules.detection.citrus_detector import citrus_detector
from modules.prediction.agriculture_predictor import agriculture_predictor
from modules.utils.data_manager import data_manager
from modules.utils.system_monitor import system_monitor
from modules.training.model_trainer import model_trainer
from modules.video_analysis.video_analyzer import VideoAnalyzer

# 初始化模块实例
video_analyzer = VideoAnalyzer()

# 页面配置
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
def load_custom_css():
    """加载自定义CSS样式"""
    # 使用简化的样式
    st.markdown("""
    <style>
        /* 简化的样式 */
        .main-header {
            font-size: 3rem;
            text-align: center;
            margin-bottom: 2rem;
            color: #FF6B35;
        }

        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .metric-card h2 {
            font-size: 2rem;
            margin: 0.5rem 0;
            font-weight: bold;
        }

        .metric-card h3 {
            font-size: 1rem;
            margin: 0.2rem 0;
            opacity: 0.9;
        }

        .info-box {
            background: #e3f2fd;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #2196F3;
            margin: 1rem 0;
        }

        .success-box {
            background: #e8f5e8;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #4CAF50;
            margin: 1rem 0;
        }

        .warning-box {
            background: #fff8e1;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #FF9800;
            margin: 1rem 0;
        }

        .error-box {
            background: #ffebee;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 5px solid #F44336;
            margin: 1rem 0;
        }

        .feature-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
        }
    </style>
    """, unsafe_allow_html=True)


def show_welcome_page():
    """显示欢迎页面"""
    st.markdown('<h1 class="main-header">🍊 智能果园检测系统</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h2>🌟 欢迎使用智能果园管理平台</h2>
        <p>集成AI技术的现代化果园管理解决方案，为您提供全方位的智能化服务</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能介绍
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🍊 果园智能检测</h3>
            <p>• 基于YOLO-MECD的柑橘检测</p>
            <p>• 实时图像分析和统计</p>
            <p>• 批量处理和结果导出</p>
            <p>• 检测历史记录管理</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🌱 农业预测建议</h3>
            <p>• 智能作物推荐系统</p>
            <p>• 产量预测和分析</p>
            <p>• AI生成种植建议</p>
            <p>• 个性化优化方案</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🤖 模型训练中心</h3>
            <p>• 自定义数据集训练</p>
            <p>• 多种网络架构选择</p>
            <p>• LeNet、AlexNet、ResNet等</p>
            <p>• 实时训练监控</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📹 视频内容理解</h3>
            <p>• 智能视频分析和查询</p>
            <p>• 关键帧自动提取</p>
            <p>• 自然语言查询支持</p>
            <p>• 事件时间线生成</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🔐 安全认证系统</h3>
            <p>• 传统密码登录</p>
            <p>• 人脸识别登录</p>
            <p>• 用户数据安全保护</p>
            <p>• 会话管理和监控</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速开始
    # 技术亮点展示
    st.markdown("---")
    st.markdown("### 🔬 技术亮点")

    tech_col1, tech_col2 = st.columns(2)

    with tech_col1:
        st.markdown("""
        <div class="info-box">
            <h4>🧠 深度学习架构</h4>
            <p>• <strong>LeNet</strong>: 经典卷积神经网络</p>
            <p>• <strong>AlexNet</strong>: 深度CNN先驱</p>
            <p>• <strong>ResNet</strong>: 残差网络架构</p>
            <p>• <strong>VGG</strong>: 深层特征提取</p>
            <p>• <strong>EfficientNet</strong>: 高效网络设计</p>
            <p>• <strong>R-CNN</strong>: 目标检测网络</p>
        </div>
        """, unsafe_allow_html=True)

    with tech_col2:
        st.markdown("""
        <div class="success-box">
            <h4>🚀 AI能力</h4>
            <p>• <strong>YOLO检测</strong>: 实时目标检测</p>
            <p>• <strong>视频理解</strong>: 智能内容分析</p>
            <p>• <strong>人脸识别</strong>: 生物特征认证</p>
            <p>• <strong>自然语言</strong>: 智能问答系统</p>
            <p>• <strong>机器学习</strong>: 预测和推荐</p>
            <p>• <strong>自定义训练</strong>: 专属模型训练</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🚀 快速开始")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔐 用户登录", use_container_width=True):
            st.session_state.show_auth = True
            st.rerun()

    with col2:
        if st.button("📝 新用户注册", use_container_width=True):
            st.session_state.show_auth = True
            st.rerun()

    with col3:
        if st.button("🎯 在线演示", use_container_width=True):
            st.balloons()
            st.success("🎉 欢迎体验智能果园检测系统！")

    # 系统统计信息
    st.markdown("---")
    st.markdown("### 📊 系统概览")

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

    with stats_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI模型</h3>
            <h2>7+</h2>
        </div>
        """, unsafe_allow_html=True)

    with stats_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔧 功能模块</h3>
            <h2>9</h2>
        </div>
        """, unsafe_allow_html=True)

    with stats_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 检测精度</h3>
            <h2>95%+</h2>
        </div>
        """, unsafe_allow_html=True)

    with stats_col4:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ 响应速度</h3>
            <h2><1s</h2>
        </div>
        """, unsafe_allow_html=True)

def show_main_navigation():
    """显示主导航菜单"""
    st.sidebar.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    st.sidebar.markdown("## 🧭 功能导航")
    
    # 导航选项
    nav_options = {
        "🏠 首页": "home",
        "🍊 果园检测": "detection",
        "📹 视频分析": "video_analysis",
        "🌱 农业预测": "prediction",
        "🤖 模型训练": "model_training",
        "👤 个人中心": "profile",
        "📊 历史记录": "history",
        "💾 数据管理": "data_management",
        "🔧 系统监控": "system_monitor"
    }
    
    # 当前页面
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    # 显示导航按钮
    for option, page_key in nav_options.items():
        if st.sidebar.button(option, use_container_width=True, key=f"nav_{page_key}"):
            st.session_state.current_page = page_key
            st.rerun()
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # 显示用户信息
    auth_manager.show_user_info()

def show_home_page():
    """显示主页"""
    st.markdown('<h1 class="main-header">🏠 系统主页</h1>', unsafe_allow_html=True)
    
    user = auth_manager.get_current_user()
    if user:
        st.markdown(f"""
        <div class="success-box">
            <h3>👋 欢迎回来，{user['username']}！</h3>
            <p>您可以使用左侧导航菜单访问各项功能</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示系统状态
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🍊 检测模型</h3>
            <h2>已加载</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📹 视频分析</h3>
            <h2>可用</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🌱 预测模型</h3>
            <h2>就绪</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>👤 用户数据</h3>
            <h2>安全</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # 快速操作
    st.markdown("### ⚡ 快速操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🍊 开始检测", use_container_width=True):
            st.session_state.current_page = "detection"
            st.rerun()
    
    with col2:
        if st.button("📹 分析视频", use_container_width=True):
            st.session_state.current_page = "video_analysis"
            st.rerun()
    
    with col3:
        if st.button("🌱 预测分析", use_container_width=True):
            st.session_state.current_page = "prediction"
            st.rerun()

    # 添加第二行快速操作
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🤖 模型训练", use_container_width=True):
            st.session_state.current_page = "model_training"
            st.rerun()

    with col2:
        if st.button("💾 数据管理", use_container_width=True):
            st.session_state.current_page = "data_management"
            st.rerun()

    with col3:
        if st.button("🔧 系统监控", use_container_width=True):
            st.session_state.current_page = "system_monitor"
            st.rerun()

def show_profile_page():
    """显示个人中心页面"""
    st.markdown('<h1 class="main-header">👤 个人中心</h1>', unsafe_allow_html=True)
    
    user = auth_manager.get_current_user()
    if user:
        # 用户基本信息
        st.markdown("### 📋 基本信息")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <p><strong>用户名:</strong> {user['username']}</p>
                <p><strong>邮箱:</strong> {user['email'] or '未设置'}</p>
                <p><strong>注册时间:</strong> {user['created_at'][:10]}</p>
                <p><strong>最后登录:</strong> {user['last_login'][:19] if user['last_login'] else '首次登录'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 人脸识别设置
            st.markdown("### 👤 人脸识别设置")
            
            face_info = face_manager.get_user_face_info(user['id'])
            
            if face_info['has_face']:
                st.success("✅ 已设置人脸识别")
                if st.button("🗑️ 删除人脸信息"):
                    success, message = face_manager.delete_user_face(user['id'])
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            else:
                st.info("ℹ️ 未设置人脸识别")
                if st.button("📷 设置人脸识别"):
                    st.session_state.show_face_registration = True
                    st.rerun()
        
        # 人脸注册界面
        if st.session_state.get('show_face_registration', False):
            face_manager.show_face_registration(user['id'])
            if st.button("❌ 取消"):
                st.session_state.show_face_registration = False
                st.rerun()

def show_video_analysis_page(user_id):
    """显示视频分析页面"""
    # 直接调用VideoAnalyzer的界面
    video_analyzer.show_video_analysis_interface(user_id)

def show_video_analysis_results(analyzer):
    """显示视频分析结果"""
    if not analyzer.frame_results:
        return

    # 显示内容总结
    st.markdown("## 📝 内容总结")
    if analyzer.event_summary:
        st.markdown(f"""
        <div class="info-box">
            {analyzer.event_summary.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

    # 视频查询功能
    st.markdown("## 🔍 视频内容查询")
    st.markdown("输入问题，系统将基于视频内容为您提供答案")

    query = st.text_input(
        "输入您的问题",
        placeholder="例如：视频中有什么果树？果园的状况如何？"
    )

    if st.button("🔍 查询", use_container_width=True) and query:
        with st.spinner("正在查询视频内容..."):
            ai_response, results = analyzer.query_video(query, top_k=3)

            if ai_response:
                st.markdown("### 🤖 AI回答")
                st.markdown(f"""
                <div class="success-box">
                    {ai_response}
                </div>
                """, unsafe_allow_html=True)

                # 显示相关片段
                if results:
                    st.markdown("### 📋 相关视频片段")
                    for i, result in enumerate(results):
                        with st.expander(f"片段 {i+1} - {result['metadata']['time_str']}"):
                            st.write(result['content'])

    # 关键帧时间线
    with st.expander("🎬 查看关键帧时间线"):
        for i, result in enumerate(analyzer.frame_results):
            st.markdown(f"### 时间点 {result['time_str']}")

            col1, col2 = st.columns([1, 2])

            with col1:
                try:
                    if os.path.exists(result['frame_path']):
                        from PIL import Image
                        img = Image.open(result['frame_path'])
                        st.image(img, caption=f"关键帧 {i+1}")
                    else:
                        st.warning("图像文件不存在")
                except Exception as e:
                    st.error(f"加载图像时出错: {str(e)}")

            with col2:
                st.markdown("**分析结果:**")
                st.markdown(f"""
                <div class="info-box">
                    {result['analysis'].replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

def show_data_management_page(user_id):
    """显示数据管理页面"""
    st.markdown('<h1 class="main-header">💾 数据管理</h1>', unsafe_allow_html=True)

    # 存储使用情况
    st.markdown("## 📊 存储使用情况")

    storage_info = data_manager.get_storage_usage(user_id)
    if storage_info:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🖼️ 用户图像</h3>
                <h2>{data_manager.format_size(storage_info['user_images'])}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📹 视频帧</h3>
                <h2>{data_manager.format_size(storage_info['video_frames'])}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📁 导出文件</h3>
                <h2>{data_manager.format_size(storage_info['exports'])}</h2>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 总使用</h3>
                <h2>{data_manager.format_size(storage_info['total_size'])}</h2>
            </div>
            """, unsafe_allow_html=True)

    # 数据导出
    st.markdown("## 📤 数据导出")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🍊 检测结果导出")
        export_format = st.selectbox("选择导出格式", ["JSON", "CSV", "Excel"], key="detection_export")

        if st.button("导出检测结果", use_container_width=True):
            with st.spinner("正在导出..."):
                file_path, message = data_manager.export_detection_results(user_id, export_format.lower())
                if file_path:
                    st.success(message)
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label="下载文件",
                            data=f.read(),
                            file_name=os.path.basename(file_path),
                            mime="application/octet-stream"
                        )
                else:
                    st.error(message)

    with col2:
        st.markdown("### 📹 视频分析导出")

        if st.button("导出视频分析", use_container_width=True):
            with st.spinner("正在导出..."):
                file_path, message = data_manager.export_video_analysis(user_id, "json")
                if file_path:
                    st.success(message)
                    with open(file_path, 'rb') as f:
                        st.download_button(
                            label="下载文件",
                            data=f.read(),
                            file_name=os.path.basename(file_path),
                            mime="application/json"
                        )
                else:
                    st.error(message)

    # 数据备份与恢复
    st.markdown("## 🔄 数据备份与恢复")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 💾 数据备份")

        if st.button("创建备份", use_container_width=True):
            with st.spinner("正在创建备份..."):
                backup_file, message = data_manager.backup_user_data(user_id)
                if backup_file:
                    st.success(message)
                    with open(backup_file, 'rb') as f:
                        st.download_button(
                            label="下载备份文件",
                            data=f.read(),
                            file_name=os.path.basename(backup_file),
                            mime="application/zip"
                        )
                else:
                    st.error(message)

    with col2:
        st.markdown("### 📁 数据恢复")

        uploaded_backup = st.file_uploader(
            "选择备份文件",
            type=['zip'],
            help="上传之前创建的备份文件"
        )

        if uploaded_backup and st.button("恢复数据", use_container_width=True):
            with st.spinner("正在恢复数据..."):
                # 保存上传的备份文件到临时位置
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                    tmp_file.write(uploaded_backup.getvalue())
                    temp_backup_path = tmp_file.name

                try:
                    success, message = data_manager.restore_user_data(user_id, temp_backup_path)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                finally:
                    os.unlink(temp_backup_path)

    # 数据清理
    st.markdown("## 🧹 数据清理")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("清理临时文件", use_container_width=True):
            with st.spinner("正在清理..."):
                count, message = data_manager.clean_temp_files()
                st.success(f"{message}，清理了 {count} 个文件")

    with col2:
        if st.button("清理缓存", use_container_width=True):
            with st.spinner("正在清理..."):
                success, message = data_manager.clean_cache()
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with col3:
        if st.button("清理数据库", use_container_width=True):
            with st.spinner("正在清理..."):
                db_manager.cleanup_old_data(days=30)
                st.success("数据库清理完成")

def show_history_page():
    """显示历史记录页面"""
    st.markdown('<h1 class="main-header">📊 历史记录</h1>', unsafe_allow_html=True)
    
    user = auth_manager.get_current_user()
    if user:
        # 选择历史记录类型
        history_type = st.selectbox(
            "选择记录类型",
            ["检测历史", "视频分析历史", "预测历史"]
        )
        
        # 获取历史记录
        if history_type == "检测历史":
            records = db_manager.get_user_history(user['id'], "detection", limit=20)
        elif history_type == "视频分析历史":
            records = db_manager.get_user_history(user['id'], "video", limit=20)
        else:
            records = db_manager.get_user_history(user['id'], "prediction", limit=20)
        
        if records:
            st.success(f"找到 {len(records)} 条记录")
            
            # 显示记录
            for i, record in enumerate(records):
                with st.expander(f"记录 {i+1} - {record['created_at'][:19]}"):
                    if history_type == "检测历史":
                        st.write(f"**检测类型:** {record['detection_type']}")
                        st.write(f"**图像路径:** {record['image_path']}")
                        st.write(f"**检测数量:** {record['detection_count']}")
                        st.write(f"**置信度阈值:** {record['confidence_threshold']}")
                    elif history_type == "视频分析历史":
                        st.write(f"**视频名称:** {record['video_name']}")
                        st.write(f"**帧数量:** {record['frame_count']}")
                        st.write(f"**视频哈希:** {record['video_hash']}")
                    else:
                        st.write(f"**预测类型:** {record['prediction_type']}")
                        if record['ai_suggestion']:
                            st.write(f"**AI建议:** {record['ai_suggestion'][:100]}...")
        else:
            st.info("暂无历史记录")

def main():
    """主函数"""
    # 加载自定义CSS
    load_custom_css()
    
    # 初始化会话状态
    auth_manager.init_session_state()
    
    # 检查是否需要显示认证页面
    if not auth_manager.is_authenticated():
        if st.session_state.get('show_auth', False):
            auth_manager.show_auth_page()
        else:
            show_welcome_page()
        return
    
    # 显示主导航
    show_main_navigation()
    
    # 根据当前页面显示内容
    current_page = st.session_state.get('current_page', 'home')
    
    if current_page == "home":
        show_home_page()
    elif current_page == "detection":
        citrus_detector.show_detection_interface(st.session_state.user_id)
    elif current_page == "video_analysis":
        show_video_analysis_page(st.session_state.user_id)
    elif current_page == "prediction":
        agriculture_predictor.show_prediction_interface(st.session_state.user_id)
    elif current_page == "model_training":
        model_trainer.show_training_interface(st.session_state.user_id)
    elif current_page == "profile":
        show_profile_page()
    elif current_page == "history":
        show_history_page()
    elif current_page == "data_management":
        show_data_management_page(st.session_state.user_id)
    elif current_page == "system_monitor":
        system_monitor.show_system_dashboard()
    
    # 调试信息
    if DEBUG_CONFIG.get("debug_mode", False):
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🐛 调试信息")
            st.write(f"当前页面: {current_page}")
            st.write(f"用户ID: {st.session_state.get('user_id', 'None')}")
            st.write(f"认证状态: {st.session_state.get('authenticated', False)}")

if __name__ == "__main__":
    main()
