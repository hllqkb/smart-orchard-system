"""
智能果园检测系统 - 配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"
TEMP_DIR = PROJECT_ROOT / "temp"

# 确保目录存在
for dir_path in [DATA_DIR, MODELS_DIR, STATIC_DIR, TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 数据库配置
DATABASE_CONFIG = {
    "path": DATA_DIR / "smart_orchard.db",
    "backup_path": DATA_DIR / "backups"
}

# 模型路径配置
MODEL_PATHS = {
    # YOLO柑橘检测模型 - 使用训练好的YOLOv11s模型
    "yolo_citrus": "/home/hllqk/projects/smart-orchard-system/model/best-baseed-yolov11s.pt",
    
    # 农作物推荐模型
    "crop_recommendation": "/home/hllqk/projects/smart-orchard-system/model/best_xgb_model.pkl",
    "crop_scaler": "/home/hllqk/projects/smart-orchard-system/model/scaler.pkl",

    # 产量预测模型
    "yield_prediction": "/home/hllqk/projects/smart-orchard-system/model/AutogluonModels/ag-20250703_165505",
    
    # 人脸识别模型目录
    "face_encodings": DATA_DIR / "face_encodings"
}

# API配置
API_CONFIG = {
    # 文心一言API
    "ernie_api_key": "1bc3aca311f155f00ad7a33d2eb5b86c472e558b",
    "ernie_base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    
    # OpenRouter API (用于农业建议)
    "openrouter_api_key": "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",
    "openrouter_base_url": "https://openrouter.ai/api/v1"
}

# 检测参数配置
DETECTION_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 100,
    "supported_formats": ["jpg", "jpeg", "png", "bmp"],
    "max_file_size": 10 * 1024 * 1024,  # 10MB
}

# 视频分析配置
VIDEO_CONFIG = {
    "supported_formats": ["mp4", "avi", "mov", "mkv", "wmv"],
    "max_file_size": 100 * 1024 * 1024,  # 100MB
    "frame_extraction_interval": 5,  # 秒
    "scene_change_threshold": 30,
    "max_frames": 20
}

# 人脸识别配置
FACE_CONFIG = {
    "tolerance": 0.6,
    "model": "hog",  # 或 "cnn"
    "max_faces_per_user": 5,
    "face_image_size": (150, 150)
}

# 用户认证配置
AUTH_CONFIG = {
    "session_timeout": 3600,  # 1小时
    "max_login_attempts": 5,
    "password_min_length": 6,
    "enable_face_login": True
}

# 界面配置
UI_CONFIG = {
    "page_title": "🍊 智能果园检测系统",
    "page_icon": "🍊",
    "layout": "wide",
    "theme": {
        "primary_color": "#FF6B35",
        "secondary_color": "#4CAF50",
        "background_color": "#FFFFFF",
        "text_color": "#333333"
    }
}

# 农作物类别配置
CROP_CATEGORIES = [
    "苹果", "香蕉", "黑豆", "鹰嘴豆", "椰子", "咖啡", "棉花", "葡萄",
    "黄麻", "芸豆", "扁豆", "玉米", "芒果", "豆荚", "绿豆", "香瓜",
    "橙子", "木瓜", "鸽豆", "石榴", "水稻", "西瓜"
]

# 柑橘检测类别配置 (基于YOLOv11s训练的专用模型)
CITRUS_CATEGORIES = {
    0: {"name": "Fruit-Citrus-0GcP", "color": (255, 0, 0), "display": "Citrus Fruit"},
    1: {"name": "Fruit_on_Ground", "color": (0, 255, 0), "display": "Ground Fruit"},
    2: {"name": "Fruit_on_Tree", "color": (0, 0, 255), "display": "Tree Fruit"}
}

# 模型详细信息
MODEL_INFO = {
    "yolo_citrus": {
        "name": "YOLOv11s柑橘检测模型",
        "architecture": "YOLOv11s",
        "purpose": "专用柑橘果实检测",
        "training_data": "柑橘果园数据集",
        "classes": list(CITRUS_CATEGORIES.keys()),
        "input_size": (640, 640),
        "description": "基于YOLOv11s架构训练的专用柑橘检测模型，针对果园环境中的柑橘果实进行了优化"
    }
}

# 缓存配置
CACHE_CONFIG = {
    "enable_cache": True,
    "cache_dir": DATA_DIR / "cache",
    "max_cache_size": 1024 * 1024 * 1024,  # 1GB
    "cache_expiry": 7 * 24 * 3600  # 7天
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "file": DATA_DIR / "logs" / "app.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# 安全配置
SECURITY_CONFIG = {
    "secret_key": "smart_orchard_secret_key_2025",
    "encrypt_user_data": True,
    "enable_csrf_protection": True
}

# 性能配置
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "max_concurrent_detections": 3,
    "image_resize_threshold": 2048,
    "video_processing_timeout": 300  # 5分钟
}

# 开发配置
DEBUG_CONFIG = {
    "debug_mode": False,
    "show_performance_metrics": True,
    "enable_profiling": False
}

# 获取配置的辅助函数
def get_model_path(model_name):
    """获取模型路径"""
    path = MODEL_PATHS.get(model_name)
    if path and os.path.exists(str(path)):
        return str(path)
    return None

def get_api_config(api_name):
    """获取API配置"""
    return API_CONFIG.get(api_name)

def is_debug_mode():
    """检查是否为调试模式"""
    return DEBUG_CONFIG.get("debug_mode", False)

def get_supported_formats(file_type):
    """获取支持的文件格式"""
    if file_type == "image":
        return DETECTION_CONFIG["supported_formats"]
    elif file_type == "video":
        return VIDEO_CONFIG["supported_formats"]
    return []
