import json
import os
import argparse
import sys

def clean_path_key(original_key):
    """
    核心清理逻辑：
    输入: D:\\Datasets\\Project\\Train\\action_10\\clip_05.mp4
    输出: action_10/clip_05.mp4
    
    输入: /home/user/data/action_10/PRE_CLIP_feature_clip_05.pkl
    输出: action_10/PRE_CLIP_feature_clip_05.pkl
    """
    # 1. 统一分隔符为 "/"
    norm_key = original_key.replace("\\", "/")
    
    # 2. 分割路径
    parts = norm_key.split("/")
    
    # 3. 寻找包含 "action_" 的层级
    action_index = -1
    for i, part in enumerate(parts):
        if part.startswith("action_"):
            action_index = i
            break
            
    # 4. 组装新 Key
    if action_index != -1 and action_index < len(parts) - 1:
        # 找到了 action_X 且它不是最后一部分
        # 取 action_X 和 文件名
        # 例如: parts[action_index] = "action_0"
        #      parts[-1]           = "clip_1.mp4"
        # 这种方式最稳健，即使中间有多层文件夹也能处理
        
        # 这里的逻辑是：取 action_X 这一层，和最后的文件名，拼接起来
        # 这样能变成标准格式: action_X/filename
        clean_key = f"{parts[action_index]}/{parts[-1]}"
        return clean_key
    else:
        # 如果找不到 action_ 或者路径结构很怪，尝试只取最后两层
        if len(parts) >= 2:
            return f"{parts[-2]}/{parts[-1]}"
        else:
            return original_key # 实在没办法，原样返回

def main():
    parser = argparse.ArgumentParser(description="Clean JSON keys to format: action_X/filename")
    parser.add_argument("--input", type=str, required=True, help="Path to original predictions.json")
    parser.add_argument("--output", type=str, required=True, help="Path to save cleaned json")
    
    args = parser.parse_args()
    
    input_path = args.input
    output_path = args.output
    
    print(f"Reading from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # 检查数据结构，VARS/Video-ChatGPT 的预测通常包裹在 "Actions" 里
    has_actions_wrapper = False
    if "Actions" in data:
        print("Detected 'Actions' wrapper in JSON.")
        raw_preds = data["Actions"]
        has_actions_wrapper = True
    else:
        print("No 'Actions' wrapper detected, processing root keys.")
        raw_preds = data

    cleaned_preds = {}
    count = 0
    
    for key, value in raw_preds.items():
        new_key = clean_path_key(key)
        cleaned_preds[new_key] = value
        count += 1
        
        if count <= 3:
            print(f"Example convert: \n  '{key}' \n  -> '{new_key}'")

    # 保持原有的 JSON 结构
    if has_actions_wrapper:
        data["Actions"] = cleaned_preds
    else:
        data = cleaned_preds

    print(f"\nProcessed {count} keys.")
    print(f"Saving to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()