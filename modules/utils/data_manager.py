"""
智能果园检测系统 - 数据管理工具
"""

import os
import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from config.settings import DATA_DIR, TEMP_DIR, CACHE_CONFIG
from config.database import db_manager

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.temp_dir = TEMP_DIR
        self.cache_dir = CACHE_CONFIG["cache_dir"]
        self.max_cache_size = CACHE_CONFIG["max_cache_size"]
        self.cache_expiry = CACHE_CONFIG["cache_expiry"]
        
        # 确保目录存在
        for dir_path in [self.data_dir, self.temp_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_detection_image(self, user_id, image, filename=None):
        """保存检测图像"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{user_id}_{timestamp}.jpg"
            
            # 创建用户目录
            user_dir = self.data_dir / "user_images" / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存图像
            image_path = user_dir / filename
            image.save(image_path)
            
            return str(image_path)
        except Exception as e:
            print(f"保存检测图像失败: {str(e)}")
            return None
    
    def save_video_frame(self, user_id, frame, timestamp, video_hash):
        """保存视频帧"""
        try:
            # 创建视频帧目录
            frame_dir = self.data_dir / "video_frames" / str(user_id) / video_hash
            frame_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存帧
            frame_filename = f"frame_{int(timestamp)}.jpg"
            frame_path = frame_dir / frame_filename
            
            import cv2
            cv2.imwrite(str(frame_path), frame)
            
            return str(frame_path)
        except Exception as e:
            print(f"保存视频帧失败: {str(e)}")
            return None
    
    def export_detection_results(self, user_id, format="json"):
        """导出检测结果"""
        try:
            # 获取用户的检测历史
            detection_history = db_manager.get_user_history(user_id, "detection", limit=100)
            
            if not detection_history:
                return None, "没有检测历史记录"
            
            # 准备导出数据
            export_data = []
            for record in detection_history:
                try:
                    results = json.loads(record['results'])
                    export_data.append({
                        "时间": record['created_at'],
                        "图像路径": record['image_path'],
                        "检测类型": record['detection_type'],
                        "检测数量": record['detection_count'],
                        "置信度阈值": record['confidence_threshold'],
                        "分析结果": results.get('analysis', {})
                    })
                except:
                    continue
            
            if not export_data:
                return None, "没有有效的检测数据"
            
            # 创建导出目录
            export_dir = self.data_dir / "exports" / str(user_id)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                # 导出为JSON
                export_file = export_dir / f"detection_results_{timestamp}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == "csv":
                # 导出为CSV
                export_file = export_dir / f"detection_results_{timestamp}.csv"
                df = pd.DataFrame(export_data)
                df.to_csv(export_file, index=False, encoding='utf-8-sig')
            
            elif format.lower() == "excel":
                # 导出为Excel
                export_file = export_dir / f"detection_results_{timestamp}.xlsx"
                df = pd.DataFrame(export_data)
                df.to_excel(export_file, index=False)
            
            else:
                return None, "不支持的导出格式"
            
            return str(export_file), "导出成功"
            
        except Exception as e:
            return None, f"导出失败: {str(e)}"
    
    def export_video_analysis(self, user_id, format="json"):
        """导出视频分析结果"""
        try:
            # 获取用户的视频分析历史
            video_history = db_manager.get_user_history(user_id, "video", limit=50)
            
            if not video_history:
                return None, "没有视频分析历史记录"
            
            # 准备导出数据
            export_data = []
            for record in video_history:
                try:
                    results = json.loads(record['analysis_results'])
                    export_data.append({
                        "时间": record['created_at'],
                        "视频名称": record['video_name'],
                        "视频哈希": record['video_hash'],
                        "帧数量": record['frame_count'],
                        "事件总结": results.get('event_summary', ''),
                        "关键帧分析": results.get('frame_results', [])
                    })
                except:
                    continue
            
            if not export_data:
                return None, "没有有效的视频分析数据"
            
            # 创建导出目录
            export_dir = self.data_dir / "exports" / str(user_id)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                # 导出为JSON
                export_file = export_dir / f"video_analysis_{timestamp}.json"
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            else:
                return None, "视频分析结果仅支持JSON格式导出"
            
            return str(export_file), "导出成功"
            
        except Exception as e:
            return None, f"导出失败: {str(e)}"
    
    def clean_temp_files(self, max_age_hours=24):
        """清理临时文件"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except:
                            continue
            
            return cleaned_count, "清理完成"
            
        except Exception as e:
            return 0, f"清理失败: {str(e)}"
    
    def clean_cache(self):
        """清理缓存"""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                return True, "缓存清理完成"
            return True, "缓存目录不存在"
        except Exception as e:
            return False, f"缓存清理失败: {str(e)}"
    
    def get_storage_usage(self, user_id=None):
        """获取存储使用情况"""
        try:
            usage_info = {
                "total_size": 0,
                "user_images": 0,
                "video_frames": 0,
                "exports": 0,
                "cache": 0,
                "temp": 0
            }
            
            # 计算各目录大小
            directories = {
                "user_images": self.data_dir / "user_images",
                "video_frames": self.data_dir / "video_frames",
                "exports": self.data_dir / "exports",
                "cache": self.cache_dir,
                "temp": self.temp_dir
            }
            
            for key, dir_path in directories.items():
                if dir_path.exists():
                    if user_id and key in ["user_images", "video_frames", "exports"]:
                        # 只计算特定用户的数据
                        user_dir = dir_path / str(user_id)
                        if user_dir.exists():
                            usage_info[key] = self._get_directory_size(user_dir)
                    else:
                        usage_info[key] = self._get_directory_size(dir_path)
            
            usage_info["total_size"] = sum(usage_info.values())
            
            return usage_info
            
        except Exception as e:
            print(f"获取存储使用情况失败: {str(e)}")
            return None
    
    def _get_directory_size(self, directory):
        """计算目录大小"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except:
            pass
        return total_size
    
    def format_size(self, size_bytes):
        """格式化文件大小"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"
    
    def backup_user_data(self, user_id):
        """备份用户数据"""
        try:
            # 创建备份目录
            backup_dir = self.data_dir / "backups" / str(user_id)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = backup_dir / f"backup_{timestamp}.zip"
            
            import zipfile
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 备份用户图像
                user_images_dir = self.data_dir / "user_images" / str(user_id)
                if user_images_dir.exists():
                    for file_path in user_images_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = f"user_images/{file_path.relative_to(user_images_dir)}"
                            zipf.write(file_path, arcname)
                
                # 备份视频帧
                video_frames_dir = self.data_dir / "video_frames" / str(user_id)
                if video_frames_dir.exists():
                    for file_path in video_frames_dir.rglob("*"):
                        if file_path.is_file():
                            arcname = f"video_frames/{file_path.relative_to(video_frames_dir)}"
                            zipf.write(file_path, arcname)
                
                # 备份数据库记录
                user_data = {
                    "detection_history": db_manager.get_user_history(user_id, "detection", limit=1000),
                    "video_history": db_manager.get_user_history(user_id, "video", limit=1000),
                    "prediction_history": db_manager.get_user_history(user_id, "prediction", limit=1000)
                }
                
                # 将数据库记录保存为JSON
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                    json.dump(user_data, tmp_file, ensure_ascii=False, indent=2, default=str)
                    tmp_file_path = tmp_file.name
                
                zipf.write(tmp_file_path, "database_records.json")
                os.unlink(tmp_file_path)
            
            return str(backup_file), "备份完成"
            
        except Exception as e:
            return None, f"备份失败: {str(e)}"
    
    def restore_user_data(self, user_id, backup_file_path):
        """恢复用户数据"""
        try:
            import zipfile
            
            if not os.path.exists(backup_file_path):
                return False, "备份文件不存在"
            
            with zipfile.ZipFile(backup_file_path, 'r') as zipf:
                # 恢复到临时目录
                temp_restore_dir = self.temp_dir / f"restore_{user_id}_{int(datetime.now().timestamp())}"
                zipf.extractall(temp_restore_dir)
                
                # 恢复用户图像
                user_images_backup = temp_restore_dir / "user_images"
                if user_images_backup.exists():
                    user_images_dir = self.data_dir / "user_images" / str(user_id)
                    user_images_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(user_images_backup, user_images_dir, dirs_exist_ok=True)
                
                # 恢复视频帧
                video_frames_backup = temp_restore_dir / "video_frames"
                if video_frames_backup.exists():
                    video_frames_dir = self.data_dir / "video_frames" / str(user_id)
                    video_frames_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(video_frames_backup, video_frames_dir, dirs_exist_ok=True)
                
                # 清理临时目录
                shutil.rmtree(temp_restore_dir)
            
            return True, "恢复完成"
            
        except Exception as e:
            return False, f"恢复失败: {str(e)}"

# 全局数据管理器实例
data_manager = DataManager()
