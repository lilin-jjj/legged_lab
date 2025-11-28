# Git 推送操作指南

本文档介绍了如何使用项目中的自动化脚本来推送代码到 GitHub。

## 使用自动化脚本推送

项目中包含了一个自动化脚本 [push_to_github.sh](file:///home/ymzz-tec/code/ros2_serial_control/push_to_github.sh)，可以帮助你完成常用的 Git 推送操作。

### 使用步骤

1. 确保你在项目根目录下
2. 给脚本添加执行权限（如果尚未添加）：
   ```bash
   chmod +x push_to_github.sh
   ```
(base) ymzz-tec@ymzztec:~/code/legged_lab$ cd /home/ymzz-tec/code/legged_lab && git remote -v
origin  https://github.com/zitongbai/legged_lab (fetch)
origin  https://github.com/zitongbai/legged_lab (push)
(base) ymzz-tec@ymzztec:~/code/legged_lab$ cd /home/ymzz-tec/code/legged_lab && git remote set-url origin https://github.com/lilin-jjj/legged_lab
(base) ymzz-tec@ymzztec:~/code/legged_lab$ cd /home/ymzz-tec/code/legged_lab && git remote -v
origin  https://github.com/lilin-jjj/legged_lab (fetch)
origin  https://github.com/lilin-jjj/legged_lab (push)

cd /home/ymzz-tec/code/legged_lab && git status


3. 运行脚本：
   ```bash
   ./push_to_github.sh
   ```

### 脚本功能

脚本会自动执行以下操作：

1. 检查是否有未暂存的更改
2. 如果有更改，会自动将所有更改添加到暂存区
3. 提示你输入提交信息
4. 自动拉取远程更改（使用 rebase 模式以保持提交历史整洁）
5. 将本地更改推送到 GitHub

## 手动推送流程

如果你希望手动执行推送操作，可以按照以下步骤：

1. 检查当前状态：
   ```bash
   git status
   ```

2. 添加更改到暂存区：
   ```bash
   git add .
   # 或者只添加特定文件
   git add <文件名>
   ```

3. 提交更改：
   ```bash
   git commit -m "提交说明"
   ```

4. 拉取远程更改（推荐使用 rebase）：
   ```bash
   git pull --rebase
   ```

5. 推送更改到远程仓库：
   ```bash
   git push
   ```

## 常见问题处理

### 1. 推送时出现冲突
如果在 `git pull --rebase` 步骤出现冲突，需要手动解决冲突后再继续：
```bash
# 解决冲突后
git add .
git rebase --continue
git push
```

### 2. 网络连接问题
如果推送时出现网络连接问题，请检查网络连接后重试。

### 3. 权限问题
确保你有向目标仓库推送的权限。如果是 SSH 方式连接，确保已正确配置 SSH 密钥。

## 最佳实践

1. **频繁提交**：小的、逻辑相关的更改作为一个提交，便于追踪和回滚
2. **有意义的提交信息**：写清楚本次提交的主要更改内容
3. **推送前先拉取**：避免出现不必要的合并冲突
4. **定期推送**：避免积累大量本地更改后再推送