#!/bin/bash
set -e

# ==========================================
# 训练脚本 - Combined + Molecular Similarity Soft Labels (配置2)
# ==========================================

echo "Job running on: $(hostname)"
echo "Working dir: $(pwd)"

# Extract --save_path from command line arguments to create the correct output directory
SAVE_PATH=""
args_array=("$@")
for i in "${!args_array[@]}"; do
    if [ "${args_array[$i]}" = "--save_path" ] && [ $((i+1)) -lt ${#args_array[@]} ]; then
        SAVE_PATH="${args_array[$((i+1))]}"
        break
    fi
done

# Create output directory immediately to ensure it exists for Condor transfer
# Remove trailing slash and leading ./ for consistency
SAVE_PATH_CLEAN=$(echo "$SAVE_PATH" | sed 's|^\./||' | sed 's|/$||')
if [ -n "$SAVE_PATH_CLEAN" ]; then
    echo "Creating output directory: $SAVE_PATH_CLEAN"
    mkdir -p "$SAVE_PATH_CLEAN"
else
    echo "WARNING: --save_path not found in arguments, creating default output_MolBridge_combined"
    mkdir -p output_MolBridge_combined
    SAVE_PATH_CLEAN="output_MolBridge_combined"
fi

# 1. 环境配置
# ------------------------------------------
if [ ! -d "molbridge_env" ]; then
    echo "Unpacking environment..."
    mkdir -p molbridge_env
    tar -xzf molbridge_env.tar.gz -C molbridge_env
else
    echo "Environment directory exists, skipping unpack."
fi

export ENV_DIR=$(pwd)/molbridge_env

# 2. 如果存在 conda-unpack，运行它修复硬编码路径
if [ -f "$ENV_DIR/bin/conda-unpack" ]; then
    echo "Running conda-unpack..."
    $ENV_DIR/bin/conda-unpack
fi

# 激活环境
echo "Configuring environment..."
if [ -f "$ENV_DIR/bin/activate" ]; then
    echo "Sourcing activate script..."
    source $ENV_DIR/bin/activate
else
    echo "Activate script not found, manually updating PATH..."
    export PATH=$ENV_DIR/bin:$PATH
fi

# 3. 验证 Python 路径 (调试用)
echo "Current PATH: $PATH"
echo "Checking python..."
which python || echo "WARNING: 'which python' failed to find python in PATH"
python --version || echo "WARNING: 'python --version' failed"

# 4. 运行训练 (Exit-Driven Checkpointing)
# ------------------------------------------
# 使用timeout实现exit-driven checkpointing
# 6天 = 144小时，在72小时限制前主动退出并保存checkpoint

CHECKPOINT_TIMEOUT="144h"  # 6天

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_ID=${_CONDOR_JOB_AD+$(grep -oP '(?<=ClusterId = )[0-9]+' $_CONDOR_JOB_AD 2>/dev/null)}
JOB_ID=${JOB_ID:-"local"}

echo "Starting training with timestamp: $TIMESTAMP"
echo "Job ID: $JOB_ID"
echo "Checkpoint timeout: $CHECKPOINT_TIMEOUT"
echo "Arguments: $@"

# Use the explicit path to python
PYTHON_EXEC="$ENV_DIR/bin/python"
if [ ! -x "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python"
fi

echo "Using python executable: $PYTHON_EXEC"

# 使用timeout运行训练，6天后自动超时
timeout $CHECKPOINT_TIMEOUT $PYTHON_EXEC train_RISEN.py "$@"

# 获取退出状态
timeout_exit_status=$?

echo "Training exited with status: $timeout_exit_status"

# Exit-Driven Checkpointing逻辑
if [ $timeout_exit_status -eq 124 ]; then
    echo "Timeout reached ($CHECKPOINT_TIMEOUT). Triggering checkpoint save and requeue..."
    echo "Exiting with code 85 to signal HTCondor to save checkpoint and requeue job."
    exit 85
fi

# 正常完成或其他错误
if [ $timeout_exit_status -eq 0 ]; then
    echo "Training completed successfully!"
fi

exit $timeout_exit_status
