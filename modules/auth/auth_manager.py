"""
智能果园检测系统 - 用户认证管理器
"""

import streamlit as st
import hashlib
import secrets
from datetime import datetime, timedelta
import re

from config.database import db_manager
from config.settings import AUTH_CONFIG

class AuthManager:
    """用户认证管理器"""
    
    def __init__(self):
        self.session_timeout = AUTH_CONFIG["session_timeout"]
        self.max_login_attempts = AUTH_CONFIG["max_login_attempts"]
        self.password_min_length = AUTH_CONFIG["password_min_length"]
    
    def init_session_state(self):
        """初始化会话状态"""
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'last_activity' not in st.session_state:
            st.session_state.last_activity = datetime.now()
    
    def is_authenticated(self):
        """检查用户是否已认证"""
        self.init_session_state()
        
        # 检查会话是否过期
        if st.session_state.authenticated:
            time_since_activity = datetime.now() - st.session_state.last_activity
            if time_since_activity.total_seconds() > self.session_timeout:
                self.logout()
                return False
            
            # 更新最后活动时间
            st.session_state.last_activity = datetime.now()
        
        return st.session_state.authenticated
    
    def login(self, username, password):
        """用户登录"""
        self.init_session_state()
        
        # 检查登录尝试次数
        if st.session_state.login_attempts >= self.max_login_attempts:
            return False, "登录尝试次数过多，请稍后再试"
        
        # 验证用户
        user_id = db_manager.verify_user(username, password)
        
        if user_id:
            st.session_state.authenticated = True
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.session_state.last_activity = datetime.now()
            st.session_state.login_attempts = 0
            
            # 记录登录日志
            db_manager.log_action(user_id, "LOGIN_SUCCESS", f"User {username} logged in successfully")
            
            return True, "登录成功"
        else:
            st.session_state.login_attempts += 1
            
            # 记录失败日志
            db_manager.log_action(None, "LOGIN_FAILED", f"Failed login attempt for username: {username}")
            
            return False, "用户名或密码错误"
    
    def register(self, username, password, confirm_password, email=None):
        """用户注册"""
        # 验证输入
        validation_result = self.validate_registration_input(username, password, confirm_password, email)
        if not validation_result[0]:
            return validation_result
        
        # 创建用户
        user_id = db_manager.create_user(username, password, email)
        
        if user_id:
            # 记录注册日志
            db_manager.log_action(user_id, "USER_REGISTERED", f"New user {username} registered")
            
            return True, "注册成功，请登录"
        else:
            return False, "用户名已存在或注册失败"
    
    def validate_registration_input(self, username, password, confirm_password, email=None):
        """验证注册输入"""
        # 检查用户名
        if not username or len(username) < 3:
            return False, "用户名至少需要3个字符"
        
        if not re.match("^[a-zA-Z0-9_]+$", username):
            return False, "用户名只能包含字母、数字和下划线"
        
        # 检查密码
        if not password or len(password) < self.password_min_length:
            return False, f"密码至少需要{self.password_min_length}个字符"
        
        if password != confirm_password:
            return False, "两次输入的密码不一致"
        
        # 检查邮箱（如果提供）
        if email and not self.validate_email(email):
            return False, "邮箱格式不正确"
        
        return True, "验证通过"
    
    def validate_email(self, email):
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def logout(self):
        """用户登出"""
        if st.session_state.get('user_id'):
            # 记录登出日志
            db_manager.log_action(st.session_state.user_id, "LOGOUT", f"User {st.session_state.username} logged out")
        
        # 清除会话状态
        st.session_state.authenticated = False
        st.session_state.user_id = None
        st.session_state.username = None
        st.session_state.login_attempts = 0
        st.session_state.last_activity = datetime.now()
    
    def get_current_user(self):
        """获取当前用户信息"""
        if self.is_authenticated():
            return db_manager.get_user_by_id(st.session_state.user_id)
        return None
    
    def require_auth(self):
        """装饰器：要求用户认证"""
        if not self.is_authenticated():
            st.warning("请先登录")
            st.stop()
    
    def show_login_form(self):
        """显示登录表单"""
        st.markdown("### 🔐 用户登录")
        
        with st.form("login_form"):
            username = st.text_input("用户名", placeholder="请输入用户名")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🚀 登录", use_container_width=True)
            with col2:
                face_login_button = st.form_submit_button("👤 人脸登录", use_container_width=True)
            
            if login_button:
                if username and password:
                    success, message = self.login(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("请输入用户名和密码")
            
            if face_login_button:
                # 调用人脸识别登录
                from modules.auth.face_recognition_manager import face_manager
                face_manager.show_face_login()
    
    def show_register_form(self):
        """显示注册表单"""
        st.markdown("### 📝 用户注册")
        
        with st.form("register_form"):
            username = st.text_input("用户名", placeholder="请输入用户名（3-20个字符）")
            email = st.text_input("邮箱", placeholder="请输入邮箱（可选）")
            password = st.text_input("密码", type="password", placeholder=f"请输入密码（至少{self.password_min_length}个字符）")
            confirm_password = st.text_input("确认密码", type="password", placeholder="请再次输入密码")
            
            register_button = st.form_submit_button("📋 注册", use_container_width=True)
            
            if register_button:
                if username and password and confirm_password:
                    success, message = self.register(username, password, confirm_password, email)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
                else:
                    st.error("请填写所有必填字段")
    
    def show_user_info(self):
        """显示用户信息"""
        user = self.get_current_user()
        if user:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 用户信息")
            st.sidebar.markdown(f"**用户名:** {user['username']}")
            if user['email']:
                st.sidebar.markdown(f"**邮箱:** {user['email']}")
            st.sidebar.markdown(f"**注册时间:** {user['created_at'][:10]}")
            
            if st.sidebar.button("🚪 退出登录"):
                self.logout()
                st.rerun()
    
    def show_auth_page(self):
        """显示认证页面"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #FF6B35;">🍊 智能果园检测系统</h1>
            <p style="color: #666; font-size: 1.2rem;">欢迎使用智能果园管理平台</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 选择登录或注册
        tab1, tab2 = st.tabs(["🔐 登录", "📝 注册"])
        
        with tab1:
            self.show_login_form()
        
        with tab2:
            self.show_register_form()
        
        # 系统介绍
        st.markdown("---")
        st.markdown("""
        ### 🌟 系统功能
        
        - **🍊 果园智能检测**: 基于AI的柑橘检测和分析
        - **📹 视频内容理解**: 智能视频分析和查询
        - **🌱 农业预测建议**: 作物推荐和产量预测
        - **👤 人脸识别登录**: 便捷的生物识别认证
        - **📊 数据管理**: 完整的历史记录和统计
        """)

# 全局认证管理器实例
auth_manager = AuthManager()
