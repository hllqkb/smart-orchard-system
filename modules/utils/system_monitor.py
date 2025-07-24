"""
智能果园检测系统 - 系统监控工具
"""

import streamlit as st
import psutil
import torch
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from config.database import db_manager
from modules.utils.data_manager import data_manager

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.start_time = datetime.now()
    
    def get_system_info(self):
        """获取系统信息"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            
            # GPU信息
            gpu_info = self.get_gpu_info()
            
            # 网络信息
            network = psutil.net_io_counters()
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": cpu_count,
                    "frequency": cpu_freq.current if cpu_freq else 0
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "gpu": gpu_info,
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv
                },
                "uptime": datetime.now() - self.start_time
            }
        except Exception as e:
            st.error(f"获取系统信息失败: {str(e)}")
            return None
    
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    # 获取GPU使用情况
                    torch.cuda.empty_cache()
                    gpu_memory_allocated = torch.cuda.memory_allocated(i)
                    gpu_memory_cached = torch.cuda.memory_reserved(i)
                    
                    gpu_info.append({
                        "name": gpu_name,
                        "total_memory": gpu_memory,
                        "allocated_memory": gpu_memory_allocated,
                        "cached_memory": gpu_memory_cached,
                        "usage_percent": (gpu_memory_allocated / gpu_memory) * 100
                    })
                
                return gpu_info
            else:
                return [{"name": "No GPU Available", "total_memory": 0, "usage_percent": 0}]
        except Exception as e:
            return [{"name": "GPU Info Error", "total_memory": 0, "usage_percent": 0}]
    
    def get_application_stats(self):
        """获取应用统计信息"""
        try:
            # 获取数据库统计
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            # 用户统计
            cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
            active_users = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM users")
            total_users = cursor.fetchone()[0]
            
            # 检测统计
            cursor.execute("SELECT COUNT(*) FROM detection_history")
            total_detections = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM detection_history WHERE created_at >= datetime('now', '-24 hours')")
            daily_detections = cursor.fetchone()[0]
            
            # 视频分析统计
            cursor.execute("SELECT COUNT(*) FROM video_analysis_history")
            total_videos = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM video_analysis_history WHERE created_at >= datetime('now', '-24 hours')")
            daily_videos = cursor.fetchone()[0]
            
            # 预测统计
            cursor.execute("SELECT COUNT(*) FROM prediction_history")
            total_predictions = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM prediction_history WHERE created_at >= datetime('now', '-24 hours')")
            daily_predictions = cursor.fetchone()[0]
            
            conn.close()
            
            # 存储使用情况
            storage_info = data_manager.get_storage_usage()
            
            return {
                "users": {
                    "total": total_users,
                    "active": active_users
                },
                "detections": {
                    "total": total_detections,
                    "daily": daily_detections
                },
                "videos": {
                    "total": total_videos,
                    "daily": daily_videos
                },
                "predictions": {
                    "total": total_predictions,
                    "daily": daily_predictions
                },
                "storage": storage_info
            }
        except Exception as e:
            st.error(f"获取应用统计失败: {str(e)}")
            return None
    
    def show_system_dashboard(self):
        """显示系统监控面板"""
        st.markdown('<h1 class="main-header">📊 系统监控</h1>', unsafe_allow_html=True)
        
        # 获取系统信息
        system_info = self.get_system_info()
        app_stats = self.get_application_stats()
        
        if not system_info or not app_stats:
            st.error("无法获取系统信息")
            return
        
        # 系统资源使用情况
        st.markdown("## 💻 系统资源")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🖥️ CPU使用率</h3>
                <h2>{system_info['cpu']['usage_percent']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💾 内存使用率</h3>
                <h2>{system_info['memory']['percent']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>💿 磁盘使用率</h3>
                <h2>{system_info['disk']['percent']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if system_info['gpu'] and system_info['gpu'][0]['name'] != "No GPU Available":
                gpu_usage = system_info['gpu'][0]['usage_percent']
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎮 GPU使用率</h3>
                    <h2>{gpu_usage:.1f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <h3>🎮 GPU</h3>
                    <h2>不可用</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # 应用统计
        st.markdown("## 📈 应用统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>👥 活跃用户</h3>
                <h2>{app_stats['users']['active']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🍊 今日检测</h3>
                <h2>{app_stats['detections']['daily']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📹 今日视频</h3>
                <h2>{app_stats['videos']['daily']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🌱 今日预测</h3>
                <h2>{app_stats['predictions']['daily']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # 详细信息
        col1, col2 = st.columns(2)
        
        with col1:
            # 系统详细信息
            st.markdown("### 🔧 系统详细信息")
            
            system_details = {
                "CPU核心数": system_info['cpu']['count'],
                "CPU频率": f"{system_info['cpu']['frequency']:.0f} MHz",
                "总内存": data_manager.format_size(system_info['memory']['total']),
                "可用内存": data_manager.format_size(system_info['memory']['available']),
                "总磁盘": data_manager.format_size(system_info['disk']['total']),
                "可用磁盘": data_manager.format_size(system_info['disk']['free']),
                "运行时间": str(system_info['uptime']).split('.')[0]
            }
            
            for key, value in system_details.items():
                st.write(f"**{key}:** {value}")
        
        with col2:
            # 存储使用情况
            st.markdown("### 💾 存储使用情况")
            
            if app_stats['storage']:
                storage_data = []
                for key, size in app_stats['storage'].items():
                    if key != 'total_size' and size > 0:
                        storage_data.append({
                            "类型": key.replace('_', ' ').title(),
                            "大小": data_manager.format_size(size),
                            "字节": size
                        })
                
                if storage_data:
                    # 存储饼图
                    fig = px.pie(
                        values=[item['字节'] for item in storage_data],
                        names=[item['类型'] for item in storage_data],
                        title="存储分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 存储详情
                    for item in storage_data:
                        st.write(f"**{item['类型']}:** {item['大小']}")
                    
                    total_size = data_manager.format_size(app_stats['storage']['total_size'])
                    st.write(f"**总计:** {total_size}")
        
        # 实时监控图表
        st.markdown("## 📊 实时监控")
        
        # 创建实时更新的图表
        if st.button("🔄 刷新监控数据"):
            st.rerun()
        
        # GPU详细信息（如果可用）
        if system_info['gpu'] and system_info['gpu'][0]['name'] != "No GPU Available":
            st.markdown("### 🎮 GPU详细信息")
            
            for i, gpu in enumerate(system_info['gpu']):
                st.write(f"**GPU {i}:** {gpu['name']}")
                st.write(f"**总显存:** {data_manager.format_size(gpu['total_memory'])}")
                st.write(f"**已用显存:** {data_manager.format_size(gpu['allocated_memory'])}")
                st.write(f"**缓存显存:** {data_manager.format_size(gpu['cached_memory'])}")
                
                # GPU使用率进度条
                st.progress(gpu['usage_percent'] / 100)
    
    def show_performance_metrics(self):
        """显示性能指标"""
        st.markdown("### ⚡ 性能指标")
        
        # 模拟性能数据（实际应用中应该从日志或监控系统获取）
        performance_data = {
            "检测平均时间": "0.85秒",
            "视频分析平均时间": "45秒",
            "预测平均时间": "0.12秒",
            "系统响应时间": "0.05秒",
            "错误率": "0.02%",
            "成功率": "99.98%"
        }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (metric, value) in enumerate(performance_data.items()):
            col = [col1, col2, col3][i % 3]
            with col:
                st.metric(metric, value)
    
    def show_system_logs(self):
        """显示系统日志"""
        st.markdown("### 📋 系统日志")
        
        try:
            # 获取最近的系统日志
            conn = db_manager.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sl.*, u.username 
                FROM system_logs sl 
                LEFT JOIN users u ON sl.user_id = u.id 
                ORDER BY sl.created_at DESC 
                LIMIT 50
            """)
            
            logs = cursor.fetchall()
            conn.close()
            
            if logs:
                log_data = []
                for log in logs:
                    log_data.append({
                        "时间": log['created_at'][:19],
                        "用户": log['username'] or '系统',
                        "操作": log['action'],
                        "详情": log['details'] or '-'
                    })
                
                df = pd.DataFrame(log_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("暂无系统日志")
                
        except Exception as e:
            st.error(f"获取系统日志失败: {str(e)}")

# 全局系统监控器实例
system_monitor = SystemMonitor()
