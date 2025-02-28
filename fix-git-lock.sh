#!/bin/bash

# 定义锁文件路径
LOCK_FILE=".git/index.lock"

# 检查锁文件是否存在
if [ -f "$LOCK_FILE" ]; then
  echo "发现 $LOCK_FILE 文件，正在删除..."
  rm -f "$LOCK_FILE"
  echo "锁文件已删除，现在可以尝试重新运行 Git 命令。"
else
  echo "没有找到 $LOCK_FILE 文件，Git 仓库可能没有锁定问题。"
fi
