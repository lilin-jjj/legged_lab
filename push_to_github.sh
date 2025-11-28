#!/bin/bash

# Git推送辅助脚本
# 用于简化日常代码推送流程



echo "开始推送代码到GitHub..."

# 检查是否有未暂存的更改
if ! git diff --quiet; then
    echo "发现未暂存的更改"
    echo "1. 添加所有更改到暂存区"
    git add .
    
    echo "2. 请输入提交信息:"
    read commit_message
    
    if [ -z "$commit_message" ]; then
        commit_message="Update code"
    fi
    
    echo "3. 提交更改"
    git commit -m "$commit_message"
else
    echo "没有未暂存的更改"
fi

echo "4. 拉取远程更改 (使用rebase模式)"
git pull --rebase

if [ $? -eq 0 ]; then
    echo "5. 推送更改到远程仓库"
    git push
    
    if [ $? -eq 0 ]; then
        echo "✅ 代码成功推送至GitHub!"
    else
        echo "❌ 推送失败，请检查网络连接或权限设置"
    fi
else
    echo "❌ 拉取远程更改失败，请手动解决冲突后再推送"
fi

echo "推送流程结束"