"""
智能果园检测系统 - 模型训练模块
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet50, vgg16, efficientnet_b0
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
import zipfile
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.settings import DATA_DIR, MODELS_DIR
from config.database import db_manager

# 导入深度学习网络架构
class LeNet(nn.Module):
    """LeNet-5 网络架构"""
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlexNet(nn.Module):
    """AlexNet 网络架构"""
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SimpleRCNN(nn.Module):
    """简化的R-CNN架构用于目标检测"""
    def __init__(self, num_classes=2, backbone='resnet18'):
        super(SimpleRCNN, self).__init__()
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
            self.backbone.fc = nn.Identity()
            feature_dim = 2048
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # 回归头（边界框）
        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # x, y, w, h
        )
    
    def forward(self, x):
        features = self.backbone(x)
        cls_scores = self.classifier(features)
        bbox_pred = self.bbox_regressor(features)
        return cls_scores, bbox_pred

class CustomDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, data_dir, transform=None, task_type='classification'):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.task_type = task_type
        self.samples = []
        self.classes = []
        
        if task_type == 'classification':
            self._load_classification_data()
        elif task_type == 'detection':
            self._load_detection_data()
    
    def _load_classification_data(self):
        """加载分类数据"""
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.classes:
                    self.classes.append(class_name)
                
                class_idx = self.classes.index(class_name)
                
                for img_file in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_file), class_idx))
    
    def _load_detection_data(self):
        """加载检测数据"""
        # 假设有annotations.json文件包含标注信息
        annotations_file = self.data_dir / 'annotations.json'
        if annotations_file.exists():
            with open(annotations_file, 'r') as f:
                annotations = json.load(f)
            
            for ann in annotations:
                img_path = self.data_dir / ann['image']
                if img_path.exists():
                    self.samples.append((str(img_path), ann))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if self.task_type == 'classification':
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        elif self.task_type == 'detection':
            img_path, annotation = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            # 提取边界框和类别
            bbox = annotation.get('bbox', [0, 0, 1, 1])
            class_id = annotation.get('class_id', 0)
            
            return image, torch.tensor(bbox, dtype=torch.float32), torch.tensor(class_id, dtype=torch.long)

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = MODELS_DIR / "trained_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # 可用的网络架构
        self.available_architectures = {
            'LeNet': LeNet,
            'AlexNet': AlexNet,
            'ResNet18': lambda num_classes: self._get_pretrained_resnet(18, num_classes),
            'ResNet50': lambda num_classes: self._get_pretrained_resnet(50, num_classes),
            'VGG16': lambda num_classes: self._get_pretrained_vgg(num_classes),
            'EfficientNet-B0': lambda num_classes: self._get_pretrained_efficientnet(num_classes),
            'Simple R-CNN': SimpleRCNN
        }
    
    def _get_pretrained_resnet(self, layers, num_classes):
        """获取预训练的ResNet模型"""
        if layers == 18:
            model = resnet18(pretrained=True)
        elif layers == 50:
            model = resnet50(pretrained=True)
        
        # 修改最后一层
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    def _get_pretrained_vgg(self, num_classes):
        """获取预训练的VGG模型"""
        model = vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    
    def _get_pretrained_efficientnet(self, num_classes):
        """获取预训练的EfficientNet模型"""
        model = efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    def prepare_data(self, data_path, task_type='classification', batch_size=32, val_split=0.2):
        """准备训练数据"""
        # 数据增强
        if task_type == 'classification':
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_transform = train_transform
        
        # 创建数据集
        full_dataset = CustomDataset(data_path, transform=train_transform, task_type=task_type)
        
        # 分割训练和验证集
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transform = val_transform
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, full_dataset.classes
    
    def create_model(self, architecture, num_classes, task_type='classification'):
        """创建模型"""
        if architecture in self.available_architectures:
            if architecture == 'Simple R-CNN' and task_type == 'detection':
                model = self.available_architectures[architecture](num_classes)
            else:
                model = self.available_architectures[architecture](num_classes)
            
            model = model.to(self.device)
            return model
        else:
            raise ValueError(f"不支持的架构: {architecture}")
    
    def train_model(self, model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, task_type='classification'):
        """训练模型"""
        # 定义损失函数和优化器
        if task_type == 'classification':
            criterion = nn.CrossEntropyLoss()
        elif task_type == 'detection':
            cls_criterion = nn.CrossEntropyLoss()
            bbox_criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 训练历史
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, batch in enumerate(train_loader):
                if task_type == 'classification':
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                elif task_type == 'detection':
                    inputs, bboxes, labels = batch
                    inputs = inputs.to(self.device)
                    bboxes = bboxes.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    cls_outputs, bbox_outputs = model(inputs)
                    
                    cls_loss = cls_criterion(cls_outputs, labels)
                    bbox_loss = bbox_criterion(bbox_outputs, bboxes)
                    loss = cls_loss + bbox_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(cls_outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                
                # 更新进度
                progress = (i + 1) / len(train_loader)
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch+1}/{num_epochs} - Batch {i+1}/{len(train_loader)}")
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if task_type == 'classification':
                        inputs, labels = batch
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                    
                    elif task_type == 'detection':
                        inputs, bboxes, labels = batch
                        inputs = inputs.to(self.device)
                        bboxes = bboxes.to(self.device)
                        labels = labels.to(self.device)
                        
                        cls_outputs, bbox_outputs = model(inputs)
                        cls_loss = cls_criterion(cls_outputs, labels)
                        bbox_loss = bbox_criterion(bbox_outputs, bboxes)
                        loss = cls_loss + bbox_loss
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(cls_outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # 记录历史
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # 更新学习率
            scheduler.step()
            
            # 显示进度
            st.write(f"Epoch {epoch+1}/{num_epochs}")
            st.write(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            st.write(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            st.write("---")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
    
    def save_model(self, model, model_name, classes, training_history, task_type='classification'):
        """保存训练好的模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        model_path = model_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # 保存模型信息
        model_info = {
            'model_name': model_name,
            'num_classes': len(classes),
            'classes': classes,
            'task_type': task_type,
            'training_history': training_history,
            'timestamp': timestamp,
            'device': str(self.device)
        }
        
        info_path = model_dir / "model_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        return str(model_dir)
    
    def plot_training_history(self, history):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_accs'], label='Train Acc')
        ax2.plot(history['val_accs'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        return fig

    def show_training_interface(self, user_id):
        """显示训练界面"""
        st.markdown('<h1 class="main-header">🤖 模型训练中心</h1>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            <h3>🎯 智能模型训练</h3>
            <p>上传您的数据集，选择网络架构，训练专属的AI模型</p>
        </div>
        """, unsafe_allow_html=True)

        # 训练配置
        col1, col2 = st.columns([2, 1])

        with col1:
            # 任务类型选择
            task_type = st.selectbox(
                "选择任务类型",
                ["classification", "detection"],
                format_func=lambda x: "图像分类" if x == "classification" else "目标检测"
            )

            # 网络架构选择
            if task_type == "classification":
                available_archs = [k for k in self.available_architectures.keys() if k != 'Simple R-CNN']
            else:
                available_archs = ['Simple R-CNN', 'ResNet18', 'ResNet50']

            architecture = st.selectbox("选择网络架构", available_archs)

            # 显示架构信息
            arch_info = {
                'LeNet': "经典的卷积神经网络，适合简单的图像分类任务",
                'AlexNet': "深度卷积神经网络的先驱，适合复杂图像分类",
                'ResNet18': "18层残差网络，平衡性能和速度",
                'ResNet50': "50层残差网络，更强的特征提取能力",
                'VGG16': "16层VGG网络，特征提取能力强",
                'EfficientNet-B0': "高效网络，在准确率和效率间取得平衡",
                'Simple R-CNN': "简化的R-CNN架构，适合目标检测任务"
            }

            st.info(f"📋 {arch_info.get(architecture, '自定义网络架构')}")

        with col2:
            # 训练参数
            st.markdown("### ⚙️ 训练参数")
            num_epochs = st.slider("训练轮数", 1, 100, 10)
            batch_size = st.selectbox("批次大小", [8, 16, 32, 64], index=2)
            learning_rate = st.selectbox("学习率", [0.0001, 0.001, 0.01, 0.1], index=1)
            val_split = st.slider("验证集比例", 0.1, 0.5, 0.2)

        # 数据上传
        st.markdown("## 📁 数据集上传")

        if task_type == "classification":
            st.markdown("""
            **分类数据集格式要求：**
            - 上传ZIP文件
            - 文件夹结构：`类别名/图像文件.jpg`
            - 支持格式：JPG, PNG
            """)
        else:
            st.markdown("""
            **检测数据集格式要求：**
            - 上传ZIP文件
            - 包含图像文件和annotations.json标注文件
            - 标注格式：`{"image": "文件名", "bbox": [x, y, w, h], "class_id": 0}`
            """)

        uploaded_file = st.file_uploader(
            "选择数据集ZIP文件",
            type=['zip'],
            help="上传包含训练数据的ZIP文件"
        )

        if uploaded_file is not None:
            # 显示数据集信息
            st.success(f"✅ 已上传数据集: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.2f} MB)")

            # 开始训练按钮
            if st.button("🚀 开始训练", use_container_width=True):
                self._start_training(
                    user_id, uploaded_file, task_type, architecture,
                    num_epochs, batch_size, learning_rate, val_split
                )

        # 显示已训练的模型
        self._show_trained_models(user_id)

    def _start_training(self, user_id, uploaded_file, task_type, architecture,
                       num_epochs, batch_size, learning_rate, val_split):
        """开始训练过程"""
        try:
            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # 解压数据集
                st.info("📦 正在解压数据集...")
                with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)

                # 查找数据目录
                data_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if not data_dirs:
                    st.error("❌ 数据集格式错误：未找到数据目录")
                    return

                data_path = data_dirs[0]

                # 准备数据
                st.info("🔄 正在准备训练数据...")
                train_loader, val_loader, classes = self.prepare_data(
                    data_path, task_type, batch_size, val_split
                )

                st.success(f"✅ 数据准备完成：{len(classes)} 个类别，{len(train_loader.dataset)} 训练样本，{len(val_loader.dataset)} 验证样本")

                # 创建模型
                st.info(f"🏗️ 正在创建 {architecture} 模型...")
                model = self.create_model(architecture, len(classes), task_type)

                # 显示模型信息
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总参数量", f"{total_params:,}")
                with col2:
                    st.metric("可训练参数", f"{trainable_params:,}")
                with col3:
                    st.metric("设备", str(self.device))

                # 开始训练
                st.info("🎯 开始模型训练...")
                history = self.train_model(
                    model, train_loader, val_loader,
                    num_epochs, learning_rate, task_type
                )

                # 显示训练结果
                st.success("🎉 训练完成！")

                # 绘制训练曲线
                fig = self.plot_training_history(history)
                st.pyplot(fig)

                # 保存模型
                model_name = f"{architecture}_{task_type}"
                model_path = self.save_model(model, model_name, classes, history, task_type)

                st.success(f"💾 模型已保存到: {model_path}")

                # 记录训练日志
                db_manager.log_action(
                    user_id,
                    "MODEL_TRAINED",
                    f"Trained {architecture} for {task_type} with {len(classes)} classes"
                )

        except Exception as e:
            st.error(f"❌ 训练过程中发生错误: {str(e)}")

    def _show_trained_models(self, user_id):
        """显示已训练的模型"""
        st.markdown("## 📚 已训练的模型")

        if not self.models_dir.exists():
            st.info("暂无已训练的模型")
            return

        model_dirs = [d for d in self.models_dir.iterdir() if d.is_dir()]

        if not model_dirs:
            st.info("暂无已训练的模型")
            return

        # 按时间排序
        model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        for model_dir in model_dirs[:10]:  # 显示最近10个模型
            info_file = model_dir / "model_info.json"
            if info_file.exists():
                with open(info_file, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)

                with st.expander(f"🤖 {model_info['model_name']} - {model_info['timestamp']}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.write(f"**任务类型:** {model_info['task_type']}")
                        st.write(f"**类别数量:** {model_info['num_classes']}")
                        st.write(f"**训练设备:** {model_info['device']}")

                    with col2:
                        st.write(f"**类别列表:**")
                        for i, cls in enumerate(model_info['classes']):
                            st.write(f"  {i}: {cls}")

                    with col3:
                        # 显示最终性能
                        history = model_info['training_history']
                        if history['val_accs']:
                            final_acc = history['val_accs'][-1]
                            st.metric("最终验证准确率", f"{final_acc:.2f}%")

                        # 下载按钮
                        model_file = model_dir / "model.pth"
                        if model_file.exists():
                            with open(model_file, 'rb') as f:
                                st.download_button(
                                    "📥 下载模型",
                                    data=f.read(),
                                    file_name=f"{model_info['model_name']}.pth",
                                    mime="application/octet-stream"
                                )

# 全局模型训练器实例
model_trainer = ModelTrainer()
