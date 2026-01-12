import os
import shutil
import random
import glob
import numpy as np
import torch
import pickle
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer

# ================= 配置区域 =================
SOURCE_ROOT = "/userhome/cs/u3598820/soccernet/mvfouls"  # 原始数据集根目录
TARGET_ROOT = "/userhome/cs/u3598820/HKU-FYP25089-VARdict/mini_dataset"  # 你的目标小数据集路径
NUM_SAMPLES = 100               # 采样数量
SPLIT = "Train"                 # 从哪个集采样
DEVICE = 'cuda'

# 假设 create_features.py 生成的 CLIP 特征就在原始视频文件夹里
# 如果你的 CLIP 特征在别的地方，请修改这里
CLIP_FEATURE_PATTERN = "PRE_CLIP_feature_clip_*.pkl" 
VIDEO_PATTERN = "clip_*.mp4"

# ===========================================

def extract_and_save_pose(video_path, save_path, inferencer):
    """
    基于你提供的 extract_pose_mmpose.py 修改的函数
    """
    try:
        # return_vis=False 加速
        result_generator = inferencer(video_path, return_vis=False, batch_size=4)
        
        all_frames_data = []
        MAX_PEOPLE = 2
        NUM_KEYPOINTS = 17 

        for result in result_generator:
            predictions = result['predictions'][0]
            frame_keypoints = []

            for person in predictions:
                # 这里的 keypoints 已经是列表了
                kpts = np.array(person['keypoints']) 
                # 这里假设我们只取 x,y，如果你需要 score，请自行修改逻辑
                # 为了简单起见，这里只取坐标，不做 score 拼接，因为你的 extract 代码最后也没存 score
                frame_keypoints.append(kpts) 

            # 对齐数据 Tensor [MAX_PEOPLE, 17, 2]
            tensor_frame = np.zeros((MAX_PEOPLE, NUM_KEYPOINTS, 2))

            if len(frame_keypoints) > 0:
                # 简单排序逻辑：按检测到的人的第一个点坐标大概排序，或者置信度(如果有)
                # 由于 mmpose 返回结果通常按置信度排序，直接取前2个即可
                actual_people = min(len(frame_keypoints), MAX_PEOPLE)
                for i in range(actual_people):
                    tensor_frame[i] = frame_keypoints[i]
            
            all_frames_data.append(tensor_frame)

        # 保存
        full_array = np.array(all_frames_data) # [Frames, 2, 17, 2]
        np.save(save_path, full_array)
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return False

def main():
    # 1. 初始化 Pose 模型 (只加载一次)
    print("正在加载 MMPose 模型...")
    inferencer = MMPoseInferencer('human', device=DEVICE)

    # 2. 确定源文件夹列表
    source_split_dir = os.path.join(SOURCE_ROOT, SPLIT)
    # 假设结构是 .../Train/action_0, .../Train/action_1
    all_action_dirs = sorted(glob.glob(os.path.join(source_split_dir, "action_*")))
    
    if len(all_action_dirs) == 0:
        print(f"错误：在 {source_split_dir} 下没有找到 action_X 文件夹")
        return

    print(f"找到 {len(all_action_dirs)} 个动作文件夹，准备采样 {NUM_SAMPLES} 个...")
    
    # 3. 随机采样
    sampled_dirs = random.sample(all_action_dirs, min(NUM_SAMPLES, len(all_action_dirs)))

    # 4. 处理循环
    success_count = 0
    
    # 创建目标根目录
    target_split_dir = os.path.join(TARGET_ROOT, SPLIT)
    os.makedirs(target_split_dir, exist_ok=True)

    for action_dir in tqdm(sampled_dirs, desc="Building Dataset"):
        action_name = os.path.basename(action_dir) # e.g., "action_123"
        target_action_dir = os.path.join(target_split_dir, action_name)
        
        os.makedirs(target_action_dir, exist_ok=True)

        # A. 查找该文件夹下所有的视频 (clip_1.mp4, clip_2.mp4 ...)
        video_files = glob.glob(os.path.join(action_dir, VIDEO_PATTERN))

        for video_file in video_files:
            clip_name = os.path.basename(video_file) # clip_1.mp4
            clip_id = clip_name.split('.')[0]        # clip_1
            
            # --- 任务 1: 复制 CLIP 特征 (.pkl) ---
            # 假设 create_features.py 生成的文件名为 PRE_CLIP_feature_clip_1.pkl
            pkl_name = f"PRE_CLIP_feature_{clip_id}.pkl"
            src_pkl_path = os.path.join(action_dir, pkl_name)
            dst_pkl_path = os.path.join(target_action_dir, pkl_name)

            if os.path.exists(src_pkl_path):
                shutil.copy2(src_pkl_path, dst_pkl_path)
            else:
                print(f"警告: 找不到 CLIP 特征 {src_pkl_path}，跳过此 Clip")
                continue # 如果没有 CLIP 特征，通常这个样本也没法用，直接跳过

            # --- 任务 2: 提取并保存 Pose 特征 (.npy) ---
            npy_name = f"{clip_id}_pose.npy"
            dst_npy_path = os.path.join(target_action_dir, npy_name)

            # 运行 Pose 推理
            if not os.path.exists(dst_npy_path): # 避免重复跑
                extract_success = extract_and_save_pose(video_file, dst_npy_path, inferencer)
                if not extract_success:
                    continue
            
            success_count += 1
            
    print(f"完成！成功处理并转移了 {success_count} 个 Clip 到 {TARGET_ROOT}")

if __name__ == "__main__":
    main()