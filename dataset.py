import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import glob
import sys

# 请确保这个路径包含 video_chatgpt 文件夹
# sys.path.append("/userhome/cs/u3598820/X-VARS/X-VARS")

from video_chatgpt.video_conversation import conv_templates, SeparatorStyle

# ================= 常量定义 =================
IGNORE_INDEX = -100
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

class VARdictDataset(Dataset):
    """
    VARdict 多模态数据集 (Optimized Version)
    - 核心优化: Prediction Key 直接匹配，移除 O(N) 循环
    """
    def __init__(self, 
                 data_root, 
                 split, 
                 json_path_qa,           # annotations_train.json
                 json_path_predictions,  # clean 后的 predictionsTrain_clip.json
                 tokenizer, 
                 video_token_len=300,
                 mode="train",
                 json_path_ground_truth=None,):
        
        self.mode = mode                 # mode
        if self.mode == "eval":
            if json_path_ground_truth is None: raise Exception("no ground truth for evaluation.")
            print(f"Loading Ground Truth from {json_path_ground_truth}...")
            with open(json_path_ground_truth, 'r') as f:
                self.gt_data = json.load(f)
        self.data_root = os.path.join(data_root, split)
        self.split = split
        self.tokenizer = tokenizer
        self.video_token_len = video_token_len
        self.conv_mode = "video-chatgpt_v1"
        
        # ---------------------------------------------------------
        # 1. 加载 QA 标注
        # ---------------------------------------------------------
        print(f"Loading annotations from {json_path_qa}...")
        with open(json_path_qa, 'r') as f:
            raw_data = json.load(f)
            # 建立查找表: "action_0" -> {question, answer}
            # 注意：这里假设 annotations 里的 path 就是 "action_0" 这种格式
            self.qa_lookup = {item['path']: item for item in raw_data}

        # ---------------------------------------------------------
        # 2. 加载 Prediction (你的 Clean JSON)
        # ---------------------------------------------------------
        print(f"Loading predictions from {json_path_predictions}...")
        with open(json_path_predictions, 'r') as f2:
            pred_data = json.load(f2)
            # 处理你的 JSON 结构: {"Set": "Train", "Actions": {...}}
            if "Actions" in pred_data:
                self.pred = pred_data["Actions"]
            else:
                self.pred = pred_data
        
        # 打印一个 Key 样例用于调试，确保加载正确
        if len(self.pred) > 0:
            example_key = list(self.pred.keys())[0]
            print(f"Prediction Key Example: '{example_key}'")

        # ---------------------------------------------------------
        # 3. 扫描文件系统并构建 Sample List
        # ---------------------------------------------------------
        self.samples = []
        print(f"Scanning files in {self.data_root}...")
        
        action_dirs = glob.glob(os.path.join(self.data_root, "action_*"))
        
        for action_dir in action_dirs:
            # action_dir_name: "action_0"
            action_dir_name = os.path.basename(action_dir)
            
            # 检查 QA 中是否有此 action
            # 兼容性处理：有些 QA json 里的 path 可能是 "Train/action_0"
            # 这里优先匹配 "action_0"
            qa_key = action_dir_name
            if qa_key not in self.qa_lookup:
                # 尝试加上 split 再次查找 (防止 QA key 是 Train/action_0)
                alt_key = f"{self.split}/{action_dir_name}"
                if alt_key in self.qa_lookup:
                    qa_key = alt_key
                else:
                    # 确实找不到，跳过
                    continue

            # 扫描 .pkl 文件
            pkl_files = glob.glob(os.path.join(action_dir, "PRE_CLIP_feature_clip_*.pkl"))
            
            for pkl_path in pkl_files:
                filename = os.path.basename(pkl_path) 
                # filename e.g.: "PRE_CLIP_feature_clip_1.pkl"
                
                # 提取 clip_id e.g.: "clip_1"
                clip_id = filename.replace("PRE_CLIP_feature_", "").replace(".pkl", "")
                
                # 推断 Pose 路径
                dirname = os.path.dirname(pkl_path)
                npy_path = os.path.join(dirname, f"{clip_id}_pose.npy")
                
                # 只有 CLIP 和 Pose 都存在才算有效数据
                if os.path.exists(npy_path):
                    
                    # 🔥 关键点：构造 Prediction Lookup Key 🔥
                    # 根据你提供的 JSON，Key 是 "action_0/PRE_CLIP_feature_clip_1.pkl"
                    # 也就是: action_dir_name / filename
                    pred_key = f"{action_dir_name}/{filename}"

                    self.samples.append({
                        'clip_path': pkl_path,
                        'pose_path': npy_path,
                        'action_key': qa_key,      # 用于查 QA
                        'pred_key': pred_key,      # 用于查 Prediction (直接匹配，无需循环)
                        'debug_id': f"{action_dir_name}/{clip_id}"
                    })
        
        print(f"[{split}] Loaded {len(self.samples)} valid samples.")

    def preprocess_text(self, qa_key, pred_key):
        """
        构建 Prompt
        - qa_key: 用于 self.qa_lookup
        - pred_key: 用于 self.pred (精准匹配)
        """
        # 1. 获取 QA
        qa_data = self.qa_lookup[qa_key]
        question = qa_data["question"]
        answer = qa_data["answer"]

        # 2. 获取 Prediction (O(1) 查找)
        # 默认值
        pred_action = "unknown"
        pred_off = "unknown"
        pred_card = "unknown"

        if pred_key in self.pred:
            pred_entry = self.pred[pred_key]
            pred_action = pred_entry.get("Action class", "unknown")
            pred_off = pred_entry.get("Offence", "unknown")
            pred_card = str(pred_entry.get("Severity", "unknown")) # 转字符串防止 float 报错
        else:
            # 这种情况理论上极少发生（除非 JSON 和文件系统不一致）
            # print(f"Warning: Key {pred_key} not found in predictions.")
            pass

        # 3. 格式化 Prediction 文本 (VARS 逻辑)
        if pred_off == "Offence":
            pred_off = ", foul and "
        elif pred_off == "No offence":
            pred_off = "and no foul."
        
        if pred_card == "1.0" or pred_card == "1":
            pred_off += "no card."
        elif pred_card == "3.0" or pred_card == "3":
            pred_off += "a yellow card."
        elif pred_card == "5.0" or pred_card == "5":
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

        # 4. 组装 Prompt
        qs = question + " The prediction for this video is " + pred_action + pred_off + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)


        if self.mode == "train":
            conv.append_message(conv.roles[1], answer)
            prompt = conv.get_prompt()

            # 5. Tokenize
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids

            targets = input_ids.clone()

            # 6. Masking (只训练 Assistant 回答部分)
            sep = "ASSISTANT:"
            total_len = int(targets.ne(self.tokenizer.pad_token_id).sum())
            
            parts = prompt.split(sep)
            if len(parts) >= 2:
                # Mask 掉 "USER: ... \nASSISTANT:"
                instruction_len = len(self.tokenizer(parts[0]).input_ids) - 1 
                targets[0, :instruction_len] = IGNORE_INDEX
            
            # Mask 掉 Padding
            cur_len = total_len
            targets[0, cur_len:] = IGNORE_INDEX

            return dict(
                input_ids=input_ids.squeeze(),
                labels=targets.squeeze(),
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id).squeeze(),
            )
        elif self.mode == "eval":
            conv.append_message(conv.roles[1], None) # 评估时：空出答案位置
            prompt = conv.get_prompt()
            
            # 评估时：不需要 padding，直接返回紧凑的张量
            tokenized = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.tokenizer.model_max_length
            )
            return dict(
                input_ids=tokenized.input_ids.squeeze(),
                attention_mask=tokenized.attention_mask.squeeze()
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. Load CLIP
        try:
            with open(sample['clip_path'], 'rb') as f:
                clip_features = pickle.load(f)
            clip_tensor = torch.from_numpy(clip_features).float()
        except Exception as e:
            print(f"Error loading CLIP {sample['clip_path']}: {e}")
            clip_tensor = torch.zeros(self.video_token_len, 1024).float()
        
        # print(clip_tensor.shape)

        # --- CLIP 归一化测试：开始 ---
        # print(f"DEBUG [Idx:{idx}] | CLIP Before Norm: Min={clip_tensor.min():.2e}, Max={clip_tensor.max():.2e}")
        
        # 使用 L2 归一化：将特征向量映射到单位球面上，彻底干掉 122 这种离群值
        if clip_tensor.numel() > 0:
            clip_tensor = torch.nn.functional.normalize(clip_tensor, p=2, dim=-1)
        
        # print(f"DEBUG [Idx:{idx}] | CLIP After Norm:  Min={clip_tensor.min():.2e}, Max={clip_tensor.max():.2e}")
        # --- CLIP 归一化测试：结束 ---
        

        # 2. Load Pose
        try:
            pose_data = np.load(sample['pose_path'])
            if pose_data.shape[0] > 0:
                pose_flat = pose_data.reshape(pose_data.shape[0], -1)
            else:
                pose_flat = np.zeros((1, 68))
            pose_tensor = torch.from_numpy(pose_flat).float()
        except Exception as e:
            print(f"Error loading Pose {sample['pose_path']}: {e}")
            pose_tensor = torch.zeros(1, 68).float()
        # print(pose_tensor.shape)

        # --- 打印测试：归一化前 ---
        # print(f"DEBUG [Idx:{idx}] | Pose Before Norm: Min={pose_tensor.min():.2f}, Max={pose_tensor.max():.2f}")
        
        # normalization
        if pose_tensor.numel() > 0: pose_tensor = pose_tensor / (pose_tensor.abs().max() + 1e-6)

        # --- 打印测试：归一化后 ---
        # print(f"DEBUG [Idx:{idx}] | Pose After Norm:  Min={pose_tensor.min():.2f}, Max={pose_tensor.max():.2f}")


        # ================= 🔥 核心修改：强制插值对齐到 300 🔥 =================
        # 1. 调整维度适应 interpolate: [T, 68] -> [1, 68, T]
        pose_tensor = pose_tensor.permute(1, 0).unsqueeze(0)
        
        # 2. 插值: 无论原来多长，统统变成 300 (与 CLIP 长度一致)
        pose_tensor = torch.nn.functional.interpolate(
            pose_tensor, 
            size=clip_tensor.shape[-2],
            mode='linear', 
            align_corners=False
        )
        
        # 3. 还原维度: [1, 68, 300] -> [300, 68]
        pose_tensor = pose_tensor.squeeze(0).permute(1, 0)
        # ===================================================================

        # print(pose_tensor.shape) # 此时永远是 [300, 68]



        # 3. Process Text (直接传入准备好的 key)
        text_data = self.preprocess_text(sample['action_key'], sample['pred_key'])

        if self.mode == "train":
            # 训练模式：保持原样，吐出带有 labels 的张量字典
            return {
                "input_ids": text_data["input_ids"],
                "labels": text_data["labels"],
                "attention_mask": text_data["attention_mask"],
                "video_spatio_temporal_features": clip_tensor,
                "pose_spatio_temporal_features": pose_tensor
            }
        elif self.mode == "eval":
            qa_data = self.qa_lookup[sample['action_key']]
            
            # 1. 提取 Action ID (从 "action_2915" 或 "Train/action_2915" 中提取 "2915")
            action_id = sample['action_key'].split('_')[-1]
            
            # 2. 从终极 Ground Truth JSON 里捞出人类裁判标注
            gt_entry = self.gt_data.get(action_id, {})
            
            raw_offence = gt_entry.get("Offence", "Unknown")
            raw_severity = str(gt_entry.get("Severity", ""))
            
            # 3. 严重程度映射 (对齐你传给 ChatGPT 的提取要求)
            if raw_severity == "": # 像你发的例子，没犯规就是空字符串
                gt_mapped_severity = "No Card"
            else:
                severity_map = {
                    "1.0": "No Card", "1": "No Card",
                    "2.0": "No Card", "2": "No Card",
                    "3.0": "Yellow Card", "3": "Yellow Card",
                    "4.0": "Red Card", "4": "Red Card",
                    "5.0": "Red Card", "5": "Red Card"
                }
                gt_mapped_severity = severity_map.get(raw_severity, "Unknown")
            
            return {
                "input_ids": text_data["input_ids"],
                "attention_mask": text_data["attention_mask"],
                "video_spatio_temporal_features": clip_tensor,
                "pose_spatio_temporal_features": pose_tensor,
                
                "video_id": sample['debug_id'],
                "raw_question": qa_data["question"],
                "gt_explanation": qa_data["answer"],
                # ⬇️ 算分必备的无污染 Ground Truth ⬇️
                "gt_offence": raw_offence,         # 例如: "No offence"
                "gt_severity": gt_mapped_severity  # 例如: "No Card"
            }


            # 1. 从 QA json 里捞出原始问题和参考解释
            qa_data = self.qa_lookup[sample['action_key']]
            
            # 2. 从 Predictions (Truth) json 里捞出量化指标
            pred_entry = self.pred.get(sample['pred_key'], {})
            raw_severity = str(pred_entry.get("Severity", "Unknown"))
            
            # 把 1.0, 3.0 转换成文本，对齐 ChatGPT 的提取结果
            severity_map = {
                "1.0": "No Card", "1": "No Card",
                "3.0": "Yellow Card", "3": "Yellow Card",
                "5.0": "Red Card", "5": "Red Card",
            }
            
            return {
                "input_ids": text_data["input_ids"],
                "attention_mask": text_data["attention_mask"],
                "video_spatio_temporal_features": clip_tensor,
                "pose_spatio_temporal_features": pose_tensor,
                
                # ⬇️ 这里就是 Eval Pipeline 需要的所有明文信息 ⬇️
                "video_id": sample['debug_id'],
                "raw_question": qa_data["question"],
                "gt_explanation": qa_data["answer"],
                "gt_offence": pred_entry.get("Offence", "Unknown"), 
                "gt_severity": severity_map.get(raw_severity, "Unknown")
            }


# ==============================================================================
#  测试模块 (If Name == Main)
# ==============================================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # --- 配置 ---
    TEST_DATA_ROOT = "/userhome/cs/u3598820/HKU-FYP25089-VARdict/mini_dataset"
    TEST_JSON_QA = "/userhome/cs/u3598820/annotations/annotations_train.json"
    TEST_JSON_PRED = "/userhome/cs/u3598820/HKU-FYP25089-VARdict/predictionsTrain_clip.json"
    MODEL_PATH = "lmsys/vicuna-7b-v1.5" # 或者本地路径

    print("=== Starting Dataset Verification ===")

    # 1. Load Tokenizer
    try:
        print(f"Loading tokenizer from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
        tokenizer.add_special_tokens({'additional_special_tokens': ['<vid_start>', '<vid_end>', '<vid_patch>']})
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
    except Exception as e:
        print(f"Tokenizer error: {e}. Using CLIP tokenizer as fallback.")
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        tokenizer.add_special_tokens({'additional_special_tokens': ['<vid_start>', '<vid_end>', '<vid_patch>']})
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Init Dataset
    try:
        dataset = VARdictDataset(
            data_root=TEST_DATA_ROOT,
            split="Train",
            json_path_qa=TEST_JSON_QA,
            json_path_predictions=TEST_JSON_PRED,
            tokenizer=tokenizer
        )
    except Exception as e:
        print(f"Dataset Init Failed: {e}")
        exit()

    # 3. Test __getitem__ and Content
    if len(dataset) > 0:
        print(f"\nFetching sample [0] (Key: {dataset.samples[0]['pred_key']})...")
        sample = dataset[0]

        # 检查 Input IDs 解码
        input_ids = sample['input_ids']
        valid_ids = input_ids.clone()
        valid_ids[valid_ids == -100] = tokenizer.pad_token_id
        decoded = tokenizer.decode(valid_ids, skip_special_tokens=False)

        print("-" * 40)
        print("DECODED PROMPT SNIPPET:")
        print(decoded[:800] + " ...")
        # print(decoded + " ...")
        print("-" * 40)

        if "The prediction for this video is" in decoded and "unknown" not in decoded:
            print("✅ SUCCESS: Prediction injected correctly!")
        elif "unknown" in decoded:
            print("⚠️ WARNING: Prediction injected but values are 'unknown'. Check JSON key matching.")
        else:
            print("❌ FAILURE: Prediction template missing.")
    else:
        print("❌ Dataset empty. Check paths.")