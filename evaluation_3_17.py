import os
import json
import torch
from tqdm import tqdm
from peft import PeftModel
# from sklearn.metrics import accuracy_score, balanced_accuracy_score
# from openai import OpenAI

# 导入你现有的模块
from load_model import safe_load_model
from dataset import VARdictDataset 

# 初始化 API (请在环境变量中设置 OPENAI_API_KEY)
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# def extract_with_chatgpt(explanation_text):
#     """
#     使用 ChatGPT 强行从长文本中抽取标准化标签
#     """
#     system_prompt = """
#     You are an expert soccer referee analyst. Read the following explanation of a soccer incident.
#     Extract the 'offence' and 'severity' based strictly on the text.
#     Return ONLY a valid JSON object with exactly two keys:
#     1. "offence": Must be either "Offence" or "No offence".
#     2. "severity": Must be either "No Card", "Yellow Card", or "Red Card".
#     If the text is ambiguous, make your best inferred guess.
#     """
#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo", # 如果有条件可以换成 gpt-4o-mini，更便宜更准
#             response_format={ "type": "json_object" },
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": explanation_text}
#             ],
#             temperature=0.0
#         )
#         result = json.loads(response.choices[0].message.content)
#         return result.get("offence", "Unknown"), result.get("severity", "Unknown")
#     except Exception as e:
#         print(f"API Error: {e}")
#         return "Unknown", "Unknown"

def main():
    # ================= 1. 配置区域 =================
    X_VARS_WEIGHTS = "/userhome/cs/u3598820/X-VARS_weight/X-VARS_weights" 
    PHASE1_PATH = "./checkpoints_3_17/phase1/final"
    LORA_PATH = "./checkpoints_3_17/phase2/final_lora" # 你的 LoRA 训练成果
    
    DATA_ROOT = "./full_dataset_3_17"
    JSON_QA = "./annotations/annotations_test.json"         # 测试集文本
    JSON_PRED = "./predictionsTest_clip.json"               # 测试集先验小抄
    JSON_GT = "/userhome/cs/u3598820/check_size/mvfouls/Test/annotations.json"        # 👈 你刚找到的终极 1-5 级硬标签
    OUTPUT_DIR = "./vardict_generated_results_3_17.json"
    
    # ================= 2. 加载模型与权重 =================
    print(">>> Loading base model...")
    model, tokenizer = safe_load_model(PHASE1_PATH, pose_feature_dim=3*17*2, device='cuda')
    
    # 初始化视觉模块 (复用你的训练逻辑)
    model.get_model().initialize_vision_modules(pretrain_mm_mlp_adapter=None)
    model.config.mm_use_vid_start_end = True
    model.get_model().vision_config.use_vid_start_end = True
    model.initialize_vision_tokenizer(
        mm_use_vid_start_end=True, tokenizer=tokenizer, device='cuda', 
        tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # 🔥 核心：加载 LoRA 并合并
    print(f">>> Loading LoRA weights from {LORA_PATH}...")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval() # 开启评估模式
    
    # ================= 3. 加载数据集 =================
    print(">>> Initializing Dataset in GENERATE mode...")
    # ⚠️ 确保你的 dataset.py 已经加上了我们上一轮讨论的 mode="eval" 开关
    eval_dataset = VARdictDataset(
        data_root=DATA_ROOT,
        split="Test",
        json_path_qa=JSON_QA,
        json_path_predictions=JSON_PRED,
        json_path_ground_truth=JSON_GT, # 传入终极标签
        tokenizer=tokenizer,
        mode="eval", # 激活评估模式，不塞入答案，返回明文 GT
        dim=3*17*2
        )
    
    results = []
    y_true_off, y_pred_off = [], []
    y_true_sev, y_pred_sev = [], []
    
    # ================= 4. 推理与评估循环 =================
    print(f">>> Starting evaluation on {len(eval_dataset)} samples...")
    # results = []
    with torch.inference_mode():
        for i in tqdm(range(len(eval_dataset))):
            item = eval_dataset[i]
            
            # 准备输入张量并升维 (添加 batch=1 维度)，注意特征要转为 FP16
            input_ids_foul = item["input_ids_foul"].unsqueeze(0).cuda()
            attention_mask_foul = item["attention_mask_foul"].unsqueeze(0).cuda()
            
            input_ids_card = item["input_ids_card"].unsqueeze(0).cuda()
            attention_mask_card = item["attention_mask_card"].unsqueeze(0).cuda()
            clip_feat = item["video_spatio_temporal_features"].unsqueeze(0).to(torch.float16).cuda()
            pose_feat = item["pose_spatio_temporal_features"].unsqueeze(0).to(torch.float16).cuda()
            
            # 1. 大模型生成裁判解释
            out_ids_foul = model.generate(
                input_ids=input_ids_foul,
                attention_mask=attention_mask_foul,
                video_spatio_temporal_features=clip_feat,
                pose_spatio_temporal_features=pose_feat,
                max_new_tokens=128, temperature=0.0
            )
            ans_foul = tokenizer.decode(out_ids_foul[0][input_ids_foul.shape[1]:], skip_special_tokens=True).strip()
            
            # --- 第二次：测给牌 ---
            out_ids_card = model.generate(
                input_ids=input_ids_card,
                attention_mask=attention_mask_card,
                video_spatio_temporal_features=clip_feat,
                pose_spatio_temporal_features=pose_feat,
                max_new_tokens=128, temperature=0.0
            )
            ans_card = tokenizer.decode(out_ids_card[0][input_ids_card.shape[1]:], skip_special_tokens=True).strip()            
            
            
            # results.append(generated_text)
            # 2. ChatGPT 抽取客观标签
            # pred_offence, pred_severity = extract_with_chatgpt(generated_text)
            
            # # 3. 从 Dataset 里拿回降维后的 3 级 Ground Truth
            gt_offence = item["gt_offence"]
            gt_severity = item["gt_severity"]
            
            # y_true_off.append(gt_offence)
            # y_pred_off.append(pred_offence)
            # y_true_sev.append(gt_severity)
            # y_pred_sev.append(pred_severity)
            
            results.append({
                "video_id": item["video_id"],
                "generated_explanation": f"Foul Assessment: {ans_foul} | Card Assessment: {ans_card}",
                "gt_offence": gt_offence, # "pred_offence": pred_offence,
                "gt_severity": gt_severity, # "pred_severity": pred_severity
            })
    with open(OUTPUT_DIR, "w") as f:
        json.dump(results, f, indent=4)


    # # ================= 5. 计算并打印指标 =================
    # # 过滤掉抽取失败的异常值，防止算分报错
    # valid_idx = [i for i, p in enumerate(y_pred_off) if p != "Unknown" and y_true_off[i] != "Unknown"]
    
    # if len(valid_idx) == 0:
    #     print("❌ CRITICAL: No valid predictions extracted!")
    #     return

    # bacc_off = balanced_accuracy_score([y_true_off[i] for i in valid_idx], [y_pred_off[i] for i in valid_idx])
    # bacc_sev = balanced_accuracy_score([y_true_sev[i] for i in valid_idx], [y_pred_sev[i] for i in valid_idx])
    
    # print("\n" + "="*40)
    # print("🏆 Evaluation Results 🏆")
    # print(f"Valid Samples: {len(valid_idx)} / {len(eval_dataset)}")
    # print(f"Offence Balanced Acc:  {bacc_off * 100:.2f}%")
    # print(f"Severity Balanced Acc: {bacc_sev * 100:.2f}%")
    # print("="*40)
    
    # # 存下来慢慢复盘
    # with open("./eval_results_log.json", "w") as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
