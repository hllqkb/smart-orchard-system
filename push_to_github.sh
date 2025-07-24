#!/bin/bash

# 推送到GitHub的脚本
# 使用方法: ./push_to_github.sh

echo "🚀 准备推送智能果园检测系统到GitHub..."

# 检查是否已经设置了远程仓库
if git remote get-url origin 2>/dev/null; then
    echo "✅ 远程仓库已设置"
else
    echo "📝 请先在GitHub上创建仓库，然后运行以下命令："
    echo ""
    echo "git remote add origin https://github.com/hllqkb/smart-orchard-system.git"
    echo ""
    echo "或者如果您使用SSH："
    echo "git remote add origin git@github.com:hllqkb/smart-orchard-system.git"
    echo ""
    read -p "是否已经创建了GitHub仓库并想要继续？(y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ 取消推送"
        exit 1
    fi
    
    read -p "请输入您的GitHub仓库URL: " repo_url
    git remote add origin "$repo_url"
fi

# 推送到GitHub
echo "📤 推送代码到GitHub..."
git branch -M main
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ 代码推送成功！"
    echo ""
    echo "🎉 您的智能果园检测系统已成功推送到GitHub！"
    echo ""
    echo "📋 接下来的步骤："
    echo "1. 📦 创建Release并上传模型文件"
    echo "2. 🔧 在GitHub仓库设置中配置项目"
    echo "3. 📚 查看README文件确保信息正确"
    echo ""
    echo "🔗 仓库地址: $(git remote get-url origin)"
else
    echo "❌ 推送失败，请检查网络连接和仓库权限"
    exit 1
fi
