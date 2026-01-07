import matplotlib
matplotlib.use("Agg")

import cv2
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer

# 1. 初始化模型
# 'human' 代表我们使用通用的高精度人体模型 (RTMPose-l)
# device='cuda' 确保使用 GPU
print("正在加载 MMPose 模型...")
inferencer = MMPoseInferencer('human', device='cuda')

# 2. 设置视频路径
video_path = "/userhome/cs/u3598820/soccernet/mvfouls/Test/action_0/clip_1.mp4" # 替换成你的文件路径
# video_path = input()

# 3. 运行推理
# return_vis=False: 不返回画了图的图片，只返回数据（速度更快）
print(f"开始处理视频: {video_path}")
result_generator = inferencer(video_path, return_vis=False, batch_size=4)


# 4. 收集数据
all_frames_data = []

# 这里我们定义一下最大人数，方便对齐 Tensor
MAX_PEOPLE = 2 
NUM_KEYPOINTS = 17 # RTMPose 默认是 COCO 17点，如果需要26点或133点可以换模型

for frame_idx, result in enumerate(result_generator):
    # result 是一个字典，里面包含了 'predictions'
    predictions = result['predictions']
    
    print(predictions)

    # 这一帧所有的关键点列表
    frame_keypoints = []
    
    for person in predictions:
        # keypoints 是一个列表 [x1, y1, x2, y2, ...] 或者是 [[x,y], [x,y]...]
        # MMPose 新版通常返回字典包含 'keypoints' 和 'keypoint_scores'
        kpts = np.array(person['keypoints']) # shape (17, 2)
        scores = np.array(person['keypoint_scores']) # shape (17,)
        
        # 把置信度拼回去变成 (x, y, score) -> shape (17, 3)
        # 这一步是为了和你之后的处理逻辑兼容
        kpts_with_score = np.column_stack((kpts, scores))
        frame_keypoints.append(kpts_with_score)
    
    # --- 数据对齐逻辑 (Padding/Truncating) ---
    # 我们需要每一帧的数据形状都是 [MAX_PEOPLE, 17, 3]
    tensor_frame = np.zeros((MAX_PEOPLE, NUM_KEYPOINTS, 3))
    
    # 按照置信度排序，取最可信的前2个人（通常是画面中心的球员）
    if len(frame_keypoints) > 0:
        # 简单的根据第一个点的置信度排序
        frame_keypoints.sort(key=lambda x: np.mean(x[:, 2]), reverse=True)
        
        # 填入数据
        actual_people = min(len(frame_keypoints), MAX_PEOPLE)
        for i in range(actual_people):
            tensor_frame[i] = frame_keypoints[i]
            
    all_frames_data.append(tensor_frame)

    if frame_idx % 10 == 0:
        print(f"已处理帧: {frame_idx}")

# 5. 保存为 .npy
full_array = np.array(all_frames_data) # shape: [Frames, 2, 17, 3]

# 只要 x,y 不要 score?
final_xy = full_array[:, :, :, :2] # shape: [Frames, 2, 17, 2]

output_path = video_path.replace(".mp4", "_pose.npy")
np.save(output_path, final_xy)

print("-" * 50)
print(f"处理完成！")
print(f"保存路径: {output_path}")
print(f"数据形状: {final_xy.shape}") # 应该显示 (帧数, 2, 17, 2)
