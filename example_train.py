import os
import torch
import glob
import transformers
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import TrainingArguments, Trainer, AutoConfig, AutoTokenizer
from load_model import safe_load_model

# 请确保这两个 import 路径正确，对应你之前保存的文件
from dataset import VARdictDataset 
from video_chatgpt_multimodal import MultimodalVideoChatGPTLlamaForCausalLM 

def get_latest_checkpoint(output_dir):
    """
    检查输出目录，寻找最新的 checkpoint
    """
    if not os.path.exists(output_dir):
        return None
    # 查找 checkpoint-* 文件夹
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # 按修改时间排序，找最新的
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def main():
    # ================= 配置区域 =================
    # X-VARS 原始权重路径 (Stage 1 Model)
    X_VARS_WEIGHTS = "/userhome/cs/u3598820/X-VARS_weight/X-VARS_weights" 
    
    # 你的数据路径
    DATA_ROOT = "./mini_dataset"
    JSON_QA = "./annotations/annotations_train.json"
    
    # 这里的 JSON_PRED 是 Stage 1 分类器生成的预测结果
    # 必须存在，否则 dataset 构建 Prompt 时会报错
    JSON_PRED = "./predictionsTrain_clip.json" 
    
    OUTPUT_DIR = "./checkpoints_debug"
    
    # ================= 1. 确定加载路径 =================
    # 优先检查是否有已经训练过的 checkpoint
    latest_ckpt = get_latest_checkpoint(os.path.join(OUTPUT_DIR, "phase1")) # 先看 Phase 1 跑完没
    if not latest_ckpt:
        latest_ckpt = get_latest_checkpoint(OUTPUT_DIR) # 或者根目录
        
    if latest_ckpt:
        print(f">>> Found checkpoint: {latest_ckpt}. Resuming training...")
        load_path = latest_ckpt
    else:
        print(f">>> No local checkpoint found. Loading base X-VARS weights...")
        load_path = X_VARS_WEIGHTS

    # ================= 2. 加载模型 =================
    model, tokenizer = safe_load_model(
        load_path, 
        pose_feature_dim=68, # 你的 Pose 维度
        device='cuda'
    )


    # ================= 必须插入的作者逻辑 =================
    # 这一步不是为了训练参数，而是为了把 Token ID (32003) 填入 model.config
    # 从而修复 forward 里的报错
    model.get_model().initialize_vision_modules(pretrain_mm_mlp_adapter=None)
    vision_config = model.get_model().vision_config
    
    # 这些 Flag 决定了数据流向
    model.config.tune_mm_mlp_adapter = True 
    model.config.freeze_mm_mlp_adapter = False
    model.config.mm_use_vid_start_end = True
    vision_config.use_vid_start_end = True
    model.config.sep_video_conv_front = False

    # 🔥 核心：把 Tokenizer 里的 ID 同步给 Config 🔥
    model.initialize_vision_tokenizer(
        mm_use_vid_start_end=True, 
        tokenizer=tokenizer, 
        device='cuda', 
        tune_mm_mlp_adapter=False, 
        pretrain_mm_mlp_adapter=None
    )
    
    model.resize_token_embeddings(len(tokenizer))
    # ===================================================




    
    # 准备数据集
    train_dataset = VARdictDataset(
        data_root=DATA_ROOT,
        split="Train",
        json_path_qa=JSON_QA,
        json_path_predictions=JSON_PRED,
        tokenizer=tokenizer
    )
    
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ================= Phase 1: Train Projector Only =================
    # 只有当从 X-VARS 加载时，或者我们明确想再跑一次 Phase 1 时才执行
    # 这里简单处理：总是跑一次 Phase 1，如果想跳过可以手动注释
    
    print("\n" + "="*20 + " Phase 1: Train Projector Only " + "="*20)
    


    # 设置梯度：冻结所有，解冻 Projector
    for name, param in model.named_parameters():
        if "mm_projector" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    print_trainable_parameters(model)
    
    args_phase1 = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "phase1"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,

        gradient_checkpointing=True,

        # gradient_checkpointing=False, # 暂时关掉，排除掉它和硬件/驱动的兼容性问题
        # predict_with_generate=False,

        fp16=False,
        bf16=True,

        # max_grad_norm=1.0,  # 🔥 核心防御：梯度裁剪
        # 正义必胜
        # victory for justice

        # learning_rate=1e-3, 
        learning_rate=1e-4, 
        num_train_epochs=1,
        save_strategy="epoch", # 跑完 Phase 1 存一下，防止崩了重来
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none"
    )
    
    trainer1 = Trainer(
        model=model,
        args=args_phase1,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    trainer1.train()
    trainer1.save_model(os.path.join(OUTPUT_DIR, "phase1", "final"))

    # ================= Phase 2: Train LLM (LoRA) Only =================
    print("\n" + "="*20 + " Phase 2: Train LLM (LoRA) Only " + "="*20)
    
    # 此时模型已经经过了 Phase 1 的训练（内存中已经是新权重）
    # 添加 LoRA Config
    
    model.config.tune_mm_mlp_adapter = False 
    model.config.freeze_mm_mlp_adapter = True


    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 注意：get_peft_model 会把原模型包一层。
    # 如果加载的模型已经是 PeftModel (比如 Resume 且 Phase 2 跑了一半)，这里需要特殊处理
    # 为了 MVP 简单起见，我们假设 Phase 1 产出的是 Base Model + Updated Projector
    model = get_peft_model(model, peft_config)
    
    # 设置梯度：冻结 Projector，解冻 LoRA
    for name, param in model.named_parameters():
        if "mm_projector" in name:
            param.requires_grad = False 
        elif "lora_" in name:           
            param.requires_grad = True
        else:
            param.requires_grad = False 
            
    print_trainable_parameters(model)
    
    args_phase2 = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "phase2"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4, 
        num_train_epochs=1,
        save_strategy="no", # 最后不存，保持硬盘清洁 (按你要求)
        logging_steps=1,
        remove_unused_columns=False,
        report_to="none",


        gradient_checkpointing=True,
        fp16=False,
        bf16=True,

    )
    
    trainer2 = Trainer(
        model=model,
        args=args_phase2,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    trainer2.train()
    
    trainer2.save_model(os.path.join(OUTPUT_DIR, "phase2", "final_lora"))

    print("\n>>> All training finished!")

if __name__ == "__main__":
    main()