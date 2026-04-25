import torch
import torch.nn as nn
import os
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel


# 确保这里的 import 路径和你项目里的文件结构对应
# 假设你的模型定义在 video_chatgpt_multimodal.py 里
from video_chatgpt_multimodal import MultimodalVideoChatGPTLlamaForCausalLM 

def safe_load_model(model_path, tokenizer_path="/userhome/cs/u3598820/base/base_model_videoChatGPT", pose_feature_dim=68, device='cuda'):
    """
    智能加载模型 (Ultimate Version)
    
    功能:
    1. 支持分片模型 (Sharded Checkpoints): 自动读取 model.safetensors.index.json。
    2. 自动维度检查 (Dimension Check):
       - 如果 Checkpoint 里的 Projector 维度匹配 (例如是从 Phase 1 恢复) -> 完美加载。
       - 如果 Checkpoint 里的 Projector 维度不匹配 (例如是原版 X-VARS) -> 自动跳过加载该层，使用全新的 nn.Linear 随机初始化。
    
    Args:
        model_path: 模型路径
        pose_feature_dim: 你的 Pose 维度 (68)
        device: 'cuda' or 'cpu' or 'auto'
    """
    print(f"\n[SafeLoad] >>> Loading model from: {model_path}")
    print(f"[SafeLoad] >>> Target Pose Dim: {pose_feature_dim}")

    # 1. 加载 Config
    # 这一步是为了获取模型的元数据，确保配置正确
    try:
        config = AutoConfig.from_pretrained(model_path)
    except Exception as e:
        raise ValueError(f"无法从 {model_path} 加载 config.json. 请检查路径。错误: {e}")

    # 2. 核心加载逻辑
    # 我们使用 ignore_mismatched_sizes=True，这一个参数就实现了你所有的需求：
    # - 也就是: "能匹配就加载，不能匹配就保持随机初始化(New nn.Linear)"
    print("[SafeLoad] >>> Instantiating model...")
    
    try:
        model, loading_info = MultimodalVideoChatGPTLlamaForCausalLM.from_pretrained(
            model_path,
            config=config,
            pose_feature_dim=pose_feature_dim, # 传入新维度，模型内部会创建 [4096, 1024+68] 的层
            # device_map=device,                 # 自动处理显存分配
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float16,         # 节省显存

            torch_dtype=torch.bfloat16,    # 🔥 改成 bfloat16

            ignore_mismatched_sizes=True,       # 🔥 核心：如果维度对不上，自动丢弃旧权重，使用新初始化的层
            output_loading_info=True  # 🔥 need this to determine if reload
        )
    except OSError:
        # 如果不是分片模型，或者是旧版权重的特殊情况，这里兜底提示
        print("❌ Load failed. ensure your model path contains .safetensors or .bin files.")
        raise


    # 2. 检查 mm_projector 是否在“失败名单”里
    # mismatched_keys 包含了因为维度不匹配（比如原版 X-VARS 的 1024->4096）而被跳过的层
    # missing_keys 包含了权重文件中根本不存在的层
    failed_to_load = loading_info['mismatched_keys'] + loading_info['missing_keys']
    # [('model.mm_projector.weight', torch.Size([4096, 1024]), torch.Size([4096, 1092]))]
    # print(loading_info['mismatched_keys'][0])
    # print("===========",failed_to_load, "===============")
    needs_initialization = any("mm_projector" in key[0] for key in failed_to_load)

    if needs_initialization:
        print("[SafeLoad] >>> ⚠️ Projector weights are MISSING or MISMATCHED. Re-initializing...")

        # =================================================================
        # 3. 🔥 手动修复 Projector (Manual Fix for Meta Tensor) 🔥
        # =================================================================
        print(f"[SafeLoad] >>> Manually re-initializing mm_projector...")

        # A. 获取维度参数
        # 1. LLM Hidden Size (通常是 4096)
        llm_hidden_size = config.hidden_size
        
        # 2. Vision Hidden Size (通常是 1024)
        # 你的 MultimodalVideoChatGPTLlamaModel 类里默认是 1024，或者从 config 读取
        mm_hidden_size = getattr(config, "mm_hidden_size", 1024)
        
        # 3. 计算融合后的输入维度
        # 根据你的类定义: fused_input_dim = self.mm_hidden_size + self.pose_feature_dim
        fused_input_dim = mm_hidden_size + pose_feature_dim

        print(f"[SafeLoad] >>> Geometry: [CLIP({mm_hidden_size}) + Pose({pose_feature_dim})] -> {fused_input_dim} => LLM({llm_hidden_size})")

        # B. Create new Linear layer
        # Initialize directly to get into CPU memory, resolving the "No data" error
        # must change to **b**float16 to prevent fp16 precision overflow
        new_projector = nn.Linear(fused_input_dim, llm_hidden_size, bias=True)
        new_projector = new_projector.to(dtype=torch.bfloat16)
        
        # C. 强制替换模型中的 Meta Layer
        # 根据 VideoChatGPTLlamaForCausalLM 的结构，projector 位于 model.model.mm_projector
        if hasattr(model.model, 'mm_projector'):
            model.model.mm_projector = new_projector
            torch.nn.init.normal_(model.model.mm_projector.weight, std=0.01) # 缩小初始权重的标准差
            torch.nn.init.zeros_(model.model.mm_projector.bias)             # 偏置清零
        else:
            # 防御性代码：万一结构层级不同
            raise AttributeError(f"Critical: Could not find 'mm_projector' in model.model. Keys: {dir(model.model)}")

        print(f"[SafeLoad] >>> Projector successfully replaced with: {new_projector}")
        # =================================================================




    # 3. 验证 Projector 状态 (User Verification)
    # 为了让你放心，我们打印一下 Projector 的权重信息
    # 我们可以通过查看权重是否全为 0 或者是否符合特定分布来猜测（但通常没必要）
    # 这里我们主要确认形状是对的
    current_shape = model.model.mm_projector.weight.shape
    print(f"[SafeLoad] >>> Model loaded. Current Projector Shape: {current_shape}")
    print(f"[SafeLoad] >>> (Expected: [Output_Dim, 1024 + {pose_feature_dim}])")

    # 手动移动到 GPU
    print(f"[SafeLoad] >>> Moving model to {device}...")
    model.to(device)

    # 4. 加载 Tokenizer
    print(f"[SafeLoad] >>> Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    print("[SafeLoad] >>> ✅ Success.\n")
    return model, tokenizer



def merge_lora_to_base(base_model, lora_path, save_merged_path=None):
    """
    将 LoRA 权重合并到基础模型中
    
    Args:
        base_model: 通过 safe_load_model 加载后的模型对象
        lora_path: Phase 2 的 LoRA 文件夹路径 (包含 adapter_config.json)
        save_merged_path: 如果提供路径，则将合并后的完整 13G 模型保存到此
    """
    print(f"\n[LoRAMerge] >>> Attaching LoRA from: {lora_path}")
    
    # 1. 挂载 LoRA
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 2. 合并
    print("[LoRAMerge] >>> Merging weights... (This may take a minute)")
    model = model.merge_and_unload()
    
    print("[LoRAMerge] >>> ✅ Merge complete.")
    
    # 3. 保存 (可选)
    if save_merged_path:
        print(f"[LoRAMerge] >>> Saving merged model to {save_merged_path}...")
        model.save_pretrained(save_merged_path)
        print("[LoRAMerge] >>> ✅ Save complete.")
        
    return model