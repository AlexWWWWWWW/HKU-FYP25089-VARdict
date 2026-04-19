import json
import os
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import concurrent.futures  # 👈 1. 引入多线程库

# ================= 配置区 =================
INPUT_JSON = "./vardict_generated_results_3_17.json"  # 你刚才生成的那个文件
OUTPUT_JSON = "./vardict_evaluated_results_3_17.json" # 提取后的结果保存位置
MODEL_NAME = "gpt-4o-mini"                       # 强烈推荐 4o-mini，比 3.5 聪明且便宜极多

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://chatapi.littlewheat.com/v1"
)

def extract_labels(explanation_text):
    """
    让 ChatGPT 从胡言乱语中强行榨取标准化标签
    """
    # system_prompt = """
    # You are an expert soccer referee logic parser. 
    # Read the following generated text from a video model and extract the intended 'offence' and 'severity'.
    
    # Rules for 'offence': Must be exactly "Offence", "No offence", or "Unknown".
    # Rules for 'severity': Must be exactly "No Card", "Yellow Card", "Red Card", or "Unknown".
    
    # Note: The text might be contradictory (e.g., "No, it is a foul"). Make your best judgment on the core intent. If it explicitly mentions a card, extract it.
    
    # Return ONLY a valid JSON format: {"offence": "...", "severity": "..."}
    # """
    system_prompt = """
    You are an expert soccer referee analyst. Read the following explanation of a soccer incident.
    Extract the 'offence' and 'severity' based strictly on the text.
    Return ONLY a valid JSON object with exactly two keys:
    1. "offence": Must be either "Offence" or "No offence".
    2. "severity": Must be either "No Card", "Yellow Card", or "Red Card".
    If the text is ambiguous, make your best inferred guess.
    """



    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": explanation_text}
            ],
            temperature=0.0 # 保持 0，我们需要确定的结果
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("offence", "Unknown"), result.get("severity", "Unknown")
    except Exception as e:
        print(f"API Error: {e}")
        return "Unknown", "Unknown"


def process_single_item(item):
    """把单个任务封装成一个函数，供多线程调用"""
    gen_text = item.get("generated_explanation", "")
    pred_off, pred_sev = extract_labels(gen_text)
    item["extracted_offence"] = pred_off
    item["extracted_severity"] = pred_sev
    return item, pred_off, pred_sev


def main():
    print(f"Loading generated results from {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    y_true_off, y_pred_off = [], []
    y_true_sev, y_pred_sev = [], []
    
    print("Starting ChatGPT extraction...")

    # 👈 2. 开启多线程池 (max_workers 就是并发数，设为 20 是比较安全的高速档位)
    MAX_WORKERS = 20 
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到线程池
        future_to_item = {executor.submit(process_single_item, item): item for item in data}
        
        # 使用 tqdm 监控多线程进度
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(data)):
            item = future_to_item[future]
            try:
                # 获取线程返回的结果
                updated_item, pred_off, pred_sev = future.result()
                
                y_true_off.append(updated_item["gt_offence"])
                y_pred_off.append(pred_off)
                
                y_true_sev.append(updated_item["gt_severity"])
                y_pred_sev.append(pred_sev)
            except Exception as exc:
                print(f"Item generated an exception: {exc}")



    # for item in tqdm(data):
    #     gen_text = item.get("generated_explanation", "")
        
    #     # 调用 API 提取
    #     pred_off, pred_sev = extract_labels(gen_text)
        
    #     # 把提取结果塞回字典里，方便后续人工复盘
    #     item["extracted_offence"] = pred_off
    #     item["extracted_severity"] = pred_sev
        
    #     y_true_off.append(item["gt_offence"])
    #     y_pred_off.append(pred_off)
        
    #     y_true_sev.append(item["gt_severity"])
    #     y_pred_sev.append(pred_sev)

    # 保存提取后的完整记录
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)

    # ================= 算分环节 =================
    # 过滤掉无法提取的 "Unknown" 数据，防止 sklearn 报错
    valid_idx_off = [i for i, p in enumerate(y_pred_off) if p != "Unknown" and y_true_off[i] != "Unknown"]
    valid_idx_sev = [i for i, p in enumerate(y_pred_sev) if p != "Unknown" and y_true_sev[i] != "Unknown"]

    print("\n" + "="*40)
    print("🏆 FINAL EVALUATION METRICS 🏆")
    print("="*40)
    
    if valid_idx_off:
        acc_off = accuracy_score([y_true_off[i] for i in valid_idx_off], [y_pred_off[i] for i in valid_idx_off])
        bacc_off = balanced_accuracy_score([y_true_off[i] for i in valid_idx_off], [y_pred_off[i] for i in valid_idx_off])
        print(f"Offence - Valid Samples: {len(valid_idx_off)}/{len(data)}")
        print(f"Offence - Accuracy:      {acc_off * 100:.2f}%")
        print(f"Offence - Balanced Acc:  {bacc_off * 100:.2f}%")
    else:
        print("Offence - No valid predictions extracted.")

    print("-" * 40)
    
    if valid_idx_sev:
        acc_sev = accuracy_score([y_true_sev[i] for i in valid_idx_sev], [y_pred_sev[i] for i in valid_idx_sev])
        bacc_sev = balanced_accuracy_score([y_true_sev[i] for i in valid_idx_sev], [y_pred_sev[i] for i in valid_idx_sev])
        print(f"Severity - Valid Samples: {len(valid_idx_sev)}/{len(data)}")
        print(f"Severity - Accuracy:      {acc_sev * 100:.2f}%")
        print(f"Severity - Balanced Acc:  {bacc_sev * 100:.2f}%")
    else:
        print("Severity - No valid predictions extracted.")
    print("="*40)

if __name__ == "__main__":
    main()
