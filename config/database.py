"""
智能果园检测系统 - 数据库配置和管理
"""

import sqlite3
import json
import hashlib
import bcrypt
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import os

from .settings import DATABASE_CONFIG, DATA_DIR

def json_serializer(obj):
    """JSON序列化辅助函数，处理numpy类型"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.db_path = DATABASE_CONFIG["path"]
        self.backup_path = DATABASE_CONFIG["backup_path"]
        Path(self.backup_path).mkdir(parents=True, exist_ok=True)
        self.init_database()
    
    def get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 使结果可以像字典一样访问
        return conn
    
    def init_database(self):
        """初始化数据库表"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 用户表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE,
                password_hash TEXT NOT NULL,
                face_encoding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                profile_data TEXT
            )
        ''')
        
        # 检测历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                detection_type TEXT NOT NULL,
                image_path TEXT,
                results TEXT,
                confidence_threshold REAL,
                detection_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 视频分析历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS video_analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                video_name TEXT NOT NULL,
                video_hash TEXT,
                analysis_results TEXT,
                frame_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 农业预测历史表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_type TEXT NOT NULL,
                input_parameters TEXT,
                prediction_result TEXT,
                ai_suggestion TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 用户会话表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # 系统日志表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_user(self, username, password, email=None, face_encoding=None):
        """创建新用户"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # 密码哈希
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # 人脸编码序列化
            face_data = pickle.dumps(face_encoding) if face_encoding is not None else None
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash, face_encoding)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, face_data))
            
            user_id = cursor.lastrowid
            conn.commit()
            
            # 记录日志
            self.log_action(user_id, "USER_CREATED", f"User {username} created")
            
            return user_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def verify_user(self, username, password):
        """验证用户密码"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, password_hash FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
            # 更新最后登录时间
            cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
            conn.commit()
            conn.close()
            
            # 记录日志
            self.log_action(user['id'], "USER_LOGIN", f"User {username} logged in")
            
            return user['id']
        
        conn.close()
        return None
    
    def get_user_by_id(self, user_id):
        """根据ID获取用户信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE id = ? AND is_active = 1', (user_id,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            user_dict = dict(user)
            # 反序列化人脸编码
            if user_dict['face_encoding']:
                user_dict['face_encoding'] = pickle.loads(user_dict['face_encoding'])
            return user_dict
        return None
    
    def get_user_by_username(self, username):
        """根据用户名获取用户信息"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ? AND is_active = 1', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            user_dict = dict(user)
            # 反序列化人脸编码
            if user_dict['face_encoding']:
                user_dict['face_encoding'] = pickle.loads(user_dict['face_encoding'])
            return user_dict
        return None
    
    def update_face_encoding(self, user_id, face_encoding):
        """更新用户人脸编码"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        face_data = pickle.dumps(face_encoding) if face_encoding is not None else None
        
        cursor.execute('UPDATE users SET face_encoding = ? WHERE id = ?', (face_data, user_id))
        conn.commit()
        conn.close()
        
        # 记录日志
        self.log_action(user_id, "FACE_ENCODING_UPDATED", "Face encoding updated")
    
    def save_detection_result(self, user_id, detection_type, image_path, results, confidence_threshold, detection_count):
        """保存检测结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        results_json = json.dumps(results, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO detection_history (user_id, detection_type, image_path, results, confidence_threshold, detection_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, detection_type, image_path, results_json, confidence_threshold, detection_count))
        
        conn.commit()
        conn.close()
        
        # 记录日志
        self.log_action(user_id, "DETECTION_SAVED", f"Detection result saved: {detection_type}")
    
    def save_video_analysis(self, user_id, video_name, video_hash, analysis_results, frame_count):
        """保存视频分析结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        results_json = json.dumps(analysis_results, ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO video_analysis_history (user_id, video_name, video_hash, analysis_results, frame_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, video_name, video_hash, results_json, frame_count))
        
        conn.commit()
        conn.close()
        
        # 记录日志
        self.log_action(user_id, "VIDEO_ANALYSIS_SAVED", f"Video analysis saved: {video_name}")
    
    def save_prediction_result(self, user_id, prediction_type, input_parameters, prediction_result, ai_suggestion=None):
        """保存预测结果"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        input_json = json.dumps(input_parameters, ensure_ascii=False, default=json_serializer)
        result_json = json.dumps(prediction_result, ensure_ascii=False, default=json_serializer)
        
        cursor.execute('''
            INSERT INTO prediction_history (user_id, prediction_type, input_parameters, prediction_result, ai_suggestion)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, prediction_type, input_json, result_json, ai_suggestion))
        
        conn.commit()
        conn.close()
        
        # 记录日志
        self.log_action(user_id, "PREDICTION_SAVED", f"Prediction saved: {prediction_type}")
    
    def get_user_history(self, user_id, history_type, limit=10):
        """获取用户历史记录"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        if history_type == "detection":
            cursor.execute('''
                SELECT * FROM detection_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
        elif history_type == "video":
            cursor.execute('''
                SELECT * FROM video_analysis_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
        elif history_type == "prediction":
            cursor.execute('''
                SELECT * FROM prediction_history 
                WHERE user_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (user_id, limit))
        else:
            conn.close()
            return []
        
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def log_action(self, user_id, action, details=None, ip_address=None):
        """记录系统日志"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_logs (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        ''', (user_id, action, details, ip_address))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_data(self, days=30):
        """清理旧数据"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # 清理旧的会话
        cursor.execute('DELETE FROM user_sessions WHERE expires_at < ?', (cutoff_date,))
        
        # 清理旧的日志
        cursor.execute('DELETE FROM system_logs WHERE created_at < ?', (cutoff_date,))
        
        conn.commit()
        conn.close()

# 全局数据库管理器实例
db_manager = DatabaseManager()
