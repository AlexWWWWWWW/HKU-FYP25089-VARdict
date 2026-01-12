import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import glob
from video_chatgpt.video_conversation import conv_templates, SeparatorStyle

# ================= 常量定义 (来自 VARS_Explain) =================
IGNORE_INDEX = -100
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

class VARdictDataset(Dataset):
    """
    VARdict 多模态数据集
    - 结构: 基于文件系统扫描 (确保 pkl 和 npy 均存在)
    - 逻辑: 集成 VARS_Explain 的 Prompt 构建与 Tokenization
    """
    def __init__(self, 
                 data_root, 
                 split, 
                 json_path_qa,           # annotations_train.json 的路径
                 json_path_predictions,  # CLIP_prediction_train.json 的路径
                 tokenizer, 
                 video_token_len=300):
        
        self.data_root = os.path.join(data_root, split)
        self.split = split
        self.tokenizer = tokenizer
        self.video_token_len = video_token_len
        self.conv_mode = "video-chatgpt_v1"
        
        # 1. 加载 QA 标注并建立查找表 (Lookup Table)
        # 原始 list 结构 -> Dict 结构: {"action_0": {question:..., answer:...}, ...}
        # 这样我们可以通过文件夹名字快速找到对应的 Question 和 Answer
        print(f"Loading annotations from {json_path_qa}...")
        with open(json_path_qa, 'r') as f:
            raw_data = json.load(f)
            # 假设 raw_data 是一个 list，每个元素有 "path": "action_0"
            self.qa_lookup = {item['path']: item for item in raw_data}

        # 2. 加载分类器预测 (用于 Prompt 增强)
        print(f"Loading predictions from {json_path_predictions}...")
        with open(json_path_predictions, 'r') as f2:
            self.pred = json.load(f2)
            if "Actions" in self.pred:
                self.pred = self.pred["Actions"]

        # 3. 扫描所有可用的样本 (File System Scanning)
        self.samples = []
        
        print(f"Scanning files in {self.data_root}...")
        action_dirs = glob.glob(os.path.join(self.data_root, "action_*"))
        
        for action_dir in action_dirs:
            action_id = os.path.basename(action_dir) # e.g. "action_0"
            
            # 如果这个 action 不在我们的标注文件里，跳过
            if action_id not in self.qa_lookup:
                continue

            # 找到里面所有的 CLIP 特征 (.pkl)
            pkl_files = glob.glob(os.path.join(action_dir, "PRE_CLIP_feature_clip_*.pkl"))
            
            for pkl_path in pkl_files:
                # 推断对应的 Pose 路径
                # pkl: .../action_0/PRE_CLIP_feature_clip_1.pkl
                dirname = os.path.dirname(pkl_path)
                filename = os.path.basename(pkl_path)
                
                # 提取 clip_id (e.g. "clip_1")
                # filename 是 PRE_CLIP_feature_clip_1.pkl
                clip_id = filename.replace("PRE_CLIP_feature_", "").replace(".pkl", "")
                
                npy_path = os.path.join(dirname, f"{clip_id}_pose.npy")
                
                # 只有当 CLIP 和 Pose 文件都存在时，才视为有效样本
                if os.path.exists(npy_path):
                    self.samples.append({
                        'clip_path': pkl_path,
                        'pose_path': npy_path,
                        'action_id': action_id,  # 用来查 QA
                        'clip_id': clip_id,      
                        'rel_path': os.path.join(action_id, f"{clip_id}.mp4") # 用来查 Prediction
                    })
        
        print(f"[{split}] Loaded {len(self.samples)} valid samples (CLIP+Pose+Annotation).")

    def preprocess_text(self, action_id, clip_path_key):
        """
        移植自 VARS_Explain 的 preprocess 逻辑
        负责构建 Prompt, Tokenize, 和 Masking
        """
        sep2 = "</s>"

        # 获取 QA
        qa_data = self.qa_lookup[action_id]
        question = qa_data["question"]
        answer = qa_data["answer"]

        # 获取预测信息 (用来构建 System Prompt)
        # 注意：这里需要构建预测文件中的 Key。
        # VARS_Explain 中的 key 通常是 "path/to/dataset/Train/action_0/clip_1.mp4" 
        # 或者有时是相对路径。这里我们尝试用 self.pred 的 key 进行匹配。
        # 为了稳健，我们先尝试构建完整 Key，如果找不到就用默认值。
        
        # 尝试构建 key：这里假设 pred 的 key 包含 action_id 和 clip_id
        # 你可能需要根据实际 predictions.json 的 key 格式微调这里
        # 简单策略：遍历 pred keys 找到包含 action_id 和 clip_id 的那个
        # (由于遍历太慢，我们假设 key 格式为 ".../{action_id}/{clip_id}.mp4" 或者类似)
        
        # 这里使用一个 trick: 你的 create_features.py 生成的 key 是完整绝对路径
        # 但我们这里只有相对路径。
        # 临时方案：我们用默认值填充，或者你需要确保 json_path_predictions 里的 key 是相对路径
        
        # 这里的 key 构造是最大的坑，需要根据你的 prediction.json 实际内容调整
        # 假设我们能找到：
        pred_action = "action"
        pred_off = "offence"
        pred_card = ""

        # 尝试从 self.pred 查找 (这是一个模糊查找的简易实现)
        # 实际使用建议统一路径格式
        full_key_guess = None
        for k in self.pred.keys():
            if f"{action_id}/{clip_id}.mp4" in k or f"{action_id}\\{clip_id}.mp4" in k: # Windows/Linux
                 # 修正：clip_path_key 是传入参数，我们用上面的 sample 信息
                 pass 
        # 为了不卡住，我们先用传入的 key 尝试查找
        if clip_path_key in self.pred:
            pred_entry = self.pred[clip_path_key]
        else:
            # Fallback: 尝试在 pred keys 里搜
            pred_entry = {"Action class": "unknown", "Offence": "unknown", "Severity": "unknown"}
            for k, v in self.pred.items():
                if f"{action_id}" in k and f"clip" in k: # 这是一个很宽泛的匹配，仅作示例
                     pass

        if clip_path_key in self.pred:
             pred_entry = self.pred[clip_path_key]
             pred_action = pred_entry["Action class"]
             pred_off = pred_entry["Offence"]
             pred_card = pred_entry["Severity"]

        # --- 以下是 VARS_Explain 的硬编码逻辑 ---
        if pred_off == "Offence":
            pred_off = ", foul and "
        if pred_off == "No offence":
            pred_off = "and no foul."
        if pred_card == "1.0":
            pred_off += "no card."
        if pred_card == "3.0":
            pred_off += "a yellow card."
        if pred_card == "5.0":
            pred_off += "a red card."

        action_map = {
            "Tackling": "a tackle ",
            "Standing tackling": "a foot duel ",
            "Elbowing": "using his elbows or arms ",
            "Holding": "holding ",
            "High leg": "a high leg ",
            "Pushing": "pushing ",
            "Challenge": "a shoulder challenge ",
            "Dive": "a simulation "
        }
        pred_action = action_map.get(pred_action, pred_action + " ")

        # 构建 Prompt
        qs = question + " The prediction for this video is " + pred_action + pred_off + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids

        targets = input_ids.clone()

        # Masking (只计算 Assistant 回答的 Loss)
        sep = "ASSISTANT:"
        total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
        
        parts = prompt.split(sep)
        if len(parts) >= 2:
            # parts[0] 是 "USER: ... \nASSISTANT:"
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1 # -1 去掉最后的 space 或 token
            # 简单粗暴的 masking：把 Instruction 部分设为 IGNORE
            # 注意：这里需要根据具体 tokenizer 的行为微调，VARS 原版逻辑比较复杂
            # 这里使用一个简化但有效的版本：
            targets[0, :instruction_len] = IGNORE_INDEX
        
        # Mask padding
        cur_len = total_len
        targets[0, cur_len:] = IGNORE_INDEX

        return dict(
            input_ids=input_ids.squeeze(),
            labels=targets.squeeze(),
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id).squeeze(),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 加载 CLIP 特征
        try:
            with open(sample['clip_path'], 'rb') as f:
                clip_features = pickle.load(f) # [T, 1024]
            clip_tensor = torch.from_numpy(clip_features).float()
        except Exception as e:
            print(f"Error loading CLIP {sample['clip_path']}: {e}")
            clip_tensor = torch.zeros(self.video_token_len, 1024).float()
        
        # 2. 加载 Pose 特征
        try:
            pose_data = np.load(sample['pose_path']) # [Frames, 2, 17, 2]
            
            # 维度展平: [Frames, 2, 17, 2] -> [Frames, 68]
            if pose_data.shape[0] > 0:
                pose_flat = pose_data.reshape(pose_data.shape[0], -1)
            else:
                pose_flat = np.zeros((1, 68))
                
            pose_tensor = torch.from_numpy(pose_flat).float()
            
        except Exception as e:
            print(f"Error loading Pose {sample['pose_path']}: {e}")
            pose_tensor = torch.zeros(1, 68).float() # Dummy

        # 3. 处理文本 (Prompt & Tokenization)
        # 关键：我们需要构造一个 key 来在 self.pred 中查找预测结果
        # 你的 create_features.py 保存 key 时使用了绝对路径。
        # 我们这里尝试构造一个尽可能匹配的 key。
        # 在真实训练中，你需要确保 create_features.py 和这里使用的路径一致。
        # 这里我们传入完整路径作为 Key 的一部分尝试查找。
        
        # 构造一个模拟的 key，或者需要在 init 里把 pred 的 key 清洗成相对路径
        # 这里为了演示，我们假设 pred json 里的 key 包含了 "action_X/clip_Y.mp4"
        lookup_key = f"{sample['action_id']}/{sample['clip_id']}.mp4" 
        
        # 在 pred 字典中寻找匹配的 Key (因为绝对路径可能不同)
        # 这是一个性能较低的查找，建议在 init 中做 key mapping
        matched_key = "unknown"
        for k in self.pred.keys():
            if lookup_key in k:
                matched_key = k
                break
        
        text_data = self.preprocess_text(sample['action_id'], matched_key)

        return {
            "input_ids": text_data["input_ids"],
            "labels": text_data["labels"],
            "attention_mask": text_data["attention_mask"],
            "video_spatio_temporal_features": clip_tensor,
            "pose_spatio_temporal_features": pose_tensor
        }







# import os
# import pickle
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# import json
# import glob

# class VARdictDataset(Dataset):
#     """
#     VARdict 多模态数据集
#     读取结构:
#         root_dir/Train/action_X/PRE_CLIP_feature_clip_Y.pkl
#         root_dir/Train/action_X/clip_Y_pose.npy
#     """
#     def __init__(self, 
#                  data_root, 
#                  split, 
#                  tokenizer, 
#                  annotations_json_path=None, 
#                  video_token_len=300):
        
#         self.data_root = os.path.join(data_root, split)
#         self.split = split
#         self.tokenizer = tokenizer
#         self.video_token_len = video_token_len
        
#         # 1. 扫描所有可用的样本
#         # 我们以 .pkl 文件为锚点，因为只有同时有 pkl 和 npy 的样本才有效
#         self.samples = []
        
#         # 遍历所有 action_X 文件夹
#         action_dirs = glob.glob(os.path.join(self.data_root, "action_*"))
        
#         for action_dir in action_dirs:
#             # 找到里面所有的 pkl
#             pkl_files = glob.glob(os.path.join(action_dir, "PRE_CLIP_feature_clip_*.pkl"))
            
#             for pkl_path in pkl_files:
#                 # 根据 pkl 路径推断 npy 路径
#                 # pkl: .../action_0/PRE_CLIP_feature_clip_1.pkl
#                 # npy: .../action_0/clip_1_pose.npy
#                 dirname = os.path.dirname(pkl_path)
#                 filename = os.path.basename(pkl_path)
                
#                 # 提取 clip_X
#                 # filename 是 PRE_CLIP_feature_clip_1.pkl -> clip_id = clip_1
#                 clip_id = filename.replace("PRE_CLIP_feature_", "").replace(".pkl", "")
                
#                 npy_path = os.path.join(dirname, f"{clip_id}_pose.npy")
                
#                 # 只有当两个文件都存在时，才加入数据集
#                 if os.path.exists(npy_path):
#                     self.samples.append({
#                         'clip_path': pkl_path,
#                         'pose_path': npy_path,
#                         'action_id': os.path.basename(dirname), # action_0
#                         'clip_id': clip_id
#                     })
        
#         print(f"[{split}] Loaded {len(self.samples)} valid samples (CLIP+Pose).")

#         # 2. 加载标注文本 (Ground Truth)
#         # 如果你有 annotations.json，这里需要加载。
#         # 这里我写一个 Placeholder，你需要根据实际 json 结构修改
#         self.annotations = {}
#         if annotations_json_path and os.path.exists(annotations_json_path):
#             with open(annotations_json_path, 'r') as f:
#                 self.annotations = json.load(f)
#         else:
#             print("Warning: No annotation JSON provided. Using dummy text.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
        
#         # --- 1. 加载 CLIP 特征 ---
#         with open(sample['clip_path'], 'rb') as f:
#             clip_features = pickle.load(f) # shape: [T, 1024]
        
#         # --- 2. 加载 Pose 特征 ---
#         pose_data = np.load(sample['pose_path']) # shape: [Frames, 2, 17, 2]
        
#         # 展平 Pose 特征以适应 Linear Layer
#         # [Frames, 2, 17, 2] -> [Frames, 2*17*2] = [Frames, 68]
#         # 注意：这里需要处理 Frames 长度，使 Pose 和 CLIP 的时间维度对齐？
#         # 通常 CLIP 特征已经下采样过了（比如 300 个 token）。
#         # 而 Pose 是每帧提取的。
#         # 简单的做法：先转成 tensor，后续在模型 forward 里做对齐 (Min-pooling)
#         # 也可以在这里采样。为了保留完整信息，我们这里只展平维度。
        
#         # 确保数据不是空的
#         if pose_data.shape[0] == 0:
#              pose_features = np.zeros((1, 68)) # Dummy
#         else:
#             # Flatten last 3 dims: people, joints, coords
#             pose_features = pose_data.reshape(pose_data.shape[0], -1) # [Frames, 68]
            
#         # 转换为 Float Tensor
#         clip_tensor = torch.from_numpy(clip_features).float()
#         pose_tensor = torch.from_numpy(pose_features).float()

#         # --- 3. 准备文本 Input (Prompt) ---
#         # 这里的逻辑需要根据你的 annotations.json 结构来定
#         # X-VARS 论文中的 Prompt 格式：
#         # USER: Question > <Pfoul> <Psev> <w> Assistant: Answer
        
#         # 这是一个示例，你需要替换成真实的 question/answer
#         key = f"{sample['action_id']}/{sample['clip_id']}"
#         qa_data = self.annotations.get(key, {"question": "Describe the foul.", "answer": "It is a foul."})
        
#         question = qa_data['question']
#         answer = qa_data['answer']
        
#         # 构建 Prompt (这里简化了，实际需要加上特殊的 Video Token)
#         # <w> 代表视频特征的位置
#         source_text = f"USER: {question} <video>\nAssistant:"
#         target_text = f"{answer} </s>"
        
#         # Tokenization (简略版，具体参考 Video-ChatGPT 的 preprocess)
#         input_ids = self.tokenizer(source_text, return_tensors='pt').input_ids[0]
#         labels = self.tokenizer(target_text, return_tensors='pt').input_ids[0]

#         return {
#             "input_ids": input_ids,
#             "labels": labels,
#             "video_spatio_temporal_features": clip_tensor,
#             "pose_spatio_temporal_features": pose_tensor
#         }