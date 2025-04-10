#!/bin/bash

# 设置工作目录
cd /root/task3

# 输出分隔线函数
print_separator() {
    echo "========================================="
    echo "$1"
    echo "========================================="
}

# 创建日志目录
LOG_DIR="/root/task3/mmpose/logs"
mkdir -p $LOG_DIR

# 记录开始时间
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "批量推理开始于: $start_time"
echo

# 依次运行每个模型的推理脚本
models=(
    "simcc"
    "hrnetv2"
    "hrnet"
    "lite"
    "vipnas_r50"
    "vipnas_mbv3"
    "hrformer"
)

for model in "${models[@]}"; do
    print_separator "运行 ${model} 模型推理"
    python mmpose/demo/batch_model_inference/${model}_batch_inference.py 2>&1 | tee "$LOG_DIR/${model}_inference.log"
    echo
done

# 记录结束时间
end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo
print_separator "批量推理完成"
echo "开始时间: $start_time"
echo "结束时间: $end_time"

# 输出结果目录信息
echo
print_separator "结果保存在:"
echo "- Cobb角度: /root/task3/mmpose/cobb/[model_name]/"
echo "- 可视化结果: /root/task3/mmpose/vis_results/[model_name]/"
echo "- 日志: $LOG_DIR/[model_name]_inference.log"
