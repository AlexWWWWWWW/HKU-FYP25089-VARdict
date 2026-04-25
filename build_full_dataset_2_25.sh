#!/bin/bash


#SBATCH --job-name=vicuna_finetune       # 任务名字
#SBATCH -p q-3090-batch                  # 官方规定的 3090 批处理分区
#SBATCH --nodes=1
#SBATCH --gres=gpu:1                     # 1张卡 (系统会自动给你配 8个CPU 和 112G内存)
#SBATCH -t 08:00:00                      # 运行 8 小时
#SBATCH --mail-type=ALL                  # 官方推荐：任务开始和结束会给你发邮件
#SBATCH --output=vicuna_log_%j.out
#SBATCH --error=vicuna_error_%j.out



# 1. 激活你的 Python/Conda 环境
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate xvars                     # 激活你的环境

# 2. 运行你的 Python 训练代码
python build_full_dataset_2_25_train.py
# python build_full_dataset_2_25_test.py