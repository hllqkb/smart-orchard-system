"""
智能果园检测系统 - 人脸识别管理器
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition
import pickle
import os
from datetime import datetime

from config.database import db_manager
from config.settings import FACE_CONFIG, DATA_DIR

class FaceRecognitionManager:
    """人脸识别管理器"""
    
    def __init__(self):
        self.tolerance = FACE_CONFIG["tolerance"]
        self.model = FACE_CONFIG["model"]
        self.max_faces_per_user = FACE_CONFIG["max_faces_per_user"]
        self.face_image_size = FACE_CONFIG["face_image_size"]
        self.face_encodings_dir = DATA_DIR / "face_encodings"
        self.face_encodings_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_faces_in_image(self, image):
        """检测图像中的人脸"""
        try:
            # 转换图像格式
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # 如果是RGBA，转换为RGB
            if len(image_array.shape) == 3 and image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # BGR to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            # 检测人脸位置
            face_locations = face_recognition.face_locations(image_array, model=self.model)
            
            return face_locations, image_array
        except Exception as e:
            st.error(f"人脸检测失败: {str(e)}")
            return [], None
    
    def encode_faces(self, image_array, face_locations):
        """编码人脸特征"""
        try:
            face_encodings = face_recognition.face_encodings(image_array, face_locations)
            return face_encodings
        except Exception as e:
            st.error(f"人脸编码失败: {str(e)}")
            return []
    
    def register_face(self, user_id, image):
        """注册用户人脸"""
        try:
            # 检测人脸
            face_locations, image_array = self.detect_faces_in_image(image)
            
            if not face_locations:
                return False, "未检测到人脸，请确保图像中有清晰的人脸"
            
            if len(face_locations) > 1:
                return False, "检测到多张人脸，请确保图像中只有一张人脸"
            
            # 编码人脸
            face_encodings = self.encode_faces(image_array, face_locations)
            
            if not face_encodings:
                return False, "人脸编码失败，请重试"
            
            face_encoding = face_encodings[0]
            
            # 保存人脸编码到数据库
            db_manager.update_face_encoding(user_id, face_encoding)
            
            # 保存人脸图像（可选）
            self.save_face_image(user_id, image_array, face_locations[0])
            
            # 记录日志
            db_manager.log_action(user_id, "FACE_REGISTERED", "Face encoding registered successfully")
            
            return True, "人脸注册成功"
            
        except Exception as e:
            return False, f"人脸注册失败: {str(e)}"
    
    def save_face_image(self, user_id, image_array, face_location):
        """保存人脸图像"""
        try:
            # 裁剪人脸区域
            top, right, bottom, left = face_location
            face_image = image_array[top:bottom, left:right]
            
            # 调整大小
            face_image = cv2.resize(face_image, self.face_image_size)
            
            # 保存图像
            face_image_path = self.face_encodings_dir / f"user_{user_id}_face.jpg"
            face_image_pil = Image.fromarray(face_image)
            face_image_pil.save(face_image_path)
            
            return str(face_image_path)
        except Exception as e:
            print(f"保存人脸图像失败: {str(e)}")
            return None
    
    def verify_face(self, image, username=None):
        """验证人脸"""
        try:
            # 检测人脸
            face_locations, image_array = self.detect_faces_in_image(image)
            
            if not face_locations:
                return False, None, "未检测到人脸"
            
            # 编码人脸
            face_encodings = self.encode_faces(image_array, face_locations)
            
            if not face_encodings:
                return False, None, "人脸编码失败"
            
            unknown_face_encoding = face_encodings[0]
            
            # 如果指定了用户名，只验证该用户
            if username:
                user = db_manager.get_user_by_username(username)
                if user and user['face_encoding'] is not None:
                    known_face_encoding = user['face_encoding']
                    
                    # 比较人脸
                    matches = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding, tolerance=self.tolerance)
                    
                    if matches[0]:
                        # 记录登录日志
                        db_manager.log_action(user['id'], "FACE_LOGIN_SUCCESS", f"Face login successful for user {username}")
                        return True, user['id'], "人脸验证成功"
                    else:
                        # 记录失败日志
                        db_manager.log_action(user['id'], "FACE_LOGIN_FAILED", f"Face login failed for user {username}")
                        return False, None, "人脸验证失败"
                else:
                    return False, None, "用户未注册人脸信息"
            
            # 如果没有指定用户名，遍历所有用户进行匹配
            else:
                # 这里需要实现获取所有有人脸编码的用户的方法
                # 为了简化，我们暂时返回需要用户名的提示
                return False, None, "请先输入用户名进行人脸验证"
                
        except Exception as e:
            return False, None, f"人脸验证失败: {str(e)}"
    
    def show_face_registration(self, user_id):
        """显示人脸注册界面"""
        st.markdown("### 👤 人脸注册")
        st.info("请上传一张清晰的正面照片进行人脸注册")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择人脸图像",
            type=['jpg', 'jpeg', 'png'],
            help="请上传清晰的正面照片，确保图像中只有一张人脸"
        )
        
        if uploaded_file is not None:
            # 显示上传的图像
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="上传的图像", use_container_width=True)
            
            with col2:
                # 检测人脸预览
                face_locations, image_array = self.detect_faces_in_image(image)
                
                if face_locations:
                    st.success(f"✅ 检测到 {len(face_locations)} 张人脸")
                    
                    if len(face_locations) == 1:
                        # 显示检测到的人脸区域
                        top, right, bottom, left = face_locations[0]
                        face_image = image_array[top:bottom, left:right]
                        st.image(face_image, caption="检测到的人脸", use_container_width=True)
                        
                        # 注册按钮
                        if st.button("🔐 注册人脸", use_container_width=True):
                            success, message = self.register_face(user_id, image)
                            if success:
                                st.success(message)
                                st.balloons()
                            else:
                                st.error(message)
                    else:
                        st.warning("⚠️ 检测到多张人脸，请确保图像中只有一张人脸")
                else:
                    st.error("❌ 未检测到人脸，请重新选择图像")
    
    def show_face_login(self):
        """显示人脸登录界面"""
        st.markdown("### 👤 人脸登录")
        
        # 用户名输入
        username = st.text_input("用户名", placeholder="请输入用户名")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择人脸图像进行登录",
            type=['jpg', 'jpeg', 'png'],
            help="请上传包含您人脸的图像进行登录验证"
        )
        
        if uploaded_file is not None and username:
            # 显示上传的图像
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="验证图像", use_container_width=True)
            
            with col2:
                # 验证按钮
                if st.button("🔍 验证人脸", use_container_width=True):
                    with st.spinner("正在验证人脸..."):
                        success, user_id, message = self.verify_face(image, username)
                        
                        if success:
                            # 设置登录状态
                            st.session_state.authenticated = True
                            st.session_state.user_id = user_id
                            st.session_state.username = username
                            st.session_state.last_activity = datetime.now()
                            
                            st.success(message)
                            st.balloons()
                            st.rerun()
                        else:
                            st.error(message)
        
        elif uploaded_file and not username:
            st.warning("请先输入用户名")
        elif username and not uploaded_file:
            st.info("请上传人脸图像进行验证")
    
    def show_camera_capture(self):
        """显示摄像头捕获界面（未来功能）"""
        st.markdown("### 📷 摄像头捕获")
        st.info("摄像头实时捕获功能将在后续版本中实现")
        
        # 这里可以集成streamlit-webrtc或其他实时视频处理库
        # 实现实时人脸检测和登录功能
    
    def get_user_face_info(self, user_id):
        """获取用户人脸信息"""
        user = db_manager.get_user_by_id(user_id)
        if user and user['face_encoding'] is not None:
            return {
                "has_face": True,
                "face_image_path": self.face_encodings_dir / f"user_{user_id}_face.jpg"
            }
        return {"has_face": False, "face_image_path": None}
    
    def delete_user_face(self, user_id):
        """删除用户人脸信息"""
        try:
            # 删除数据库中的人脸编码
            db_manager.update_face_encoding(user_id, None)
            
            # 删除人脸图像文件
            face_image_path = self.face_encodings_dir / f"user_{user_id}_face.jpg"
            if face_image_path.exists():
                face_image_path.unlink()
            
            # 记录日志
            db_manager.log_action(user_id, "FACE_DELETED", "Face encoding deleted")
            
            return True, "人脸信息删除成功"
        except Exception as e:
            return False, f"删除人脸信息失败: {str(e)}"

# 全局人脸识别管理器实例
face_manager = FaceRecognitionManager()
