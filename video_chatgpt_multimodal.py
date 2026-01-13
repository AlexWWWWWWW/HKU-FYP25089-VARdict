from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# 导入原有的配置类和 Token 定义
from video_chatgpt.model.video_chatgpt import (
    VideoChatGPTLlamaModel,
    VideoChatGPTLlamaForCausalLM,
    VideoChatGPTConfig,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    DEFAULT_VIDEO_PATCH_TOKEN
)

class MultimodalVideoChatGPTLlamaModel(VideoChatGPTLlamaModel):
    """
    扩展的 VideoChatGPT 模型核心，支持 CLIP + Pose (Concatenate 融合)
    """
    def __init__(
        self,
        config: LlamaConfig,
        mm_vision_tower=None,
        mm_hidden_size=None,
        pose_feature_dim: int = 68,  # 默认为 68 (2人 x 17点 x 2坐标)
    ):
        super(MultimodalVideoChatGPTLlamaModel, self).__init__(config, mm_vision_tower, mm_hidden_size)
        
        self.pose_feature_dim = pose_feature_dim
        
        # 1. 获取基础配置
        if hasattr(config, "mm_hidden_size"):
            self.mm_hidden_size = config.mm_hidden_size
        else:
            self.mm_hidden_size = 1024 # 默认 CLIP Large 维度

        # 2. 计算融合后的输入维度 (CLIP + Pose)
        # 例如: 1024 + 68 = 1092
        fused_input_dim = self.mm_hidden_size + self.pose_feature_dim
        
        # 3. 重写投影层 (Projector)
        # 这一层将负责把 (视频+动作) 的联合特征映射到 LLM 的 4096 维空间
        # 注意: 加载预训练权重时，这一层会因为形状不匹配而被跳过(随机初始化)，这是符合预期的 (Stage 2 需要重训)
        if hasattr(config, "use_mm_proj") and config.use_mm_proj:
            self.mm_projector = nn.Linear(fused_input_dim, config.hidden_size)

    def fuse_features(self, clip_features: torch.Tensor, pose_features: torch.Tensor) -> torch.Tensor:
        """
        执行特征拼接融合
        """
        # A. 安全检查：将 Pose 移动到与 CLIP 相同的设备和数据类型
        # CLIP 通常是 fp16/bf16 且在 GPU 上，Pose 刚加载可能是 fp32/CPU
        if pose_features.device != clip_features.device:
            pose_features = pose_features.to(clip_features.device)
        if pose_features.dtype != clip_features.dtype:
            pose_features = pose_features.to(clip_features.dtype)

        # B. Align input length (Min-Pooling)
        # align according to the minimum length
        # min_len = min(clip_features.shape[1], pose_features.shape[1])
        # clip_features = clip_features[:, :min_len, :]
        # pose_features = pose_features[:, :min_len, :]
        # print(clip_features.shape, pose_features.shape)
        # C. Concatenation
        # [Batch, T, 1024] + [Batch, T, 68] -> [Batch, T, 1092]

        # 检查是否有 NaN 或 Inf
        if torch.isnan(clip_features).any() or torch.isinf(clip_features).any():
            print("❌ CLIP Features contain NaN/Inf!")

        if torch.isnan(pose_features).any() or torch.isinf(pose_features).any():
            print("❌ Pose Features contain NaN/Inf!")


        # 检查是否全是 0 (最重要！)
        if clip_features.sum() == 0 or pose_features.sum() == 0:
            print("❌ CLIP/Pose Features are ALL ZEROS! (Data loading failed)")

        fused = torch.cat([clip_features, pose_features], dim=-1)
        

        # print(f"DEBUG | Fused Features - Min: {fused.min().item():.4e}, Max: {fused.max().item():.4e}, Mean: {fused.mean().item():.4e}")
        if torch.isnan(fused).any():
            print("❌ Fused features contain NaN before projector!")
        return fused

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
        pose_spatio_temporal_features: Optional[torch.FloatTensor] = None, # 新增输入
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        

        # weight_sum = self.mm_projector.weight.sum().item()
        # print(f"DEBUG | Projector Weight Sum: {weight_sum}")
        
        if torch.isnan(self.mm_projector.weight).any():
             print("❌ FATAL: Projector weight is NaN at the VERY START of forward!")

        

        if (input_ids < 0).any() or (input_ids >= self.config.vocab_size).any():
            print(f"❌ CRITICAL: Illegal input_ids detected! Range: {input_ids.min()} to {input_ids.max()}")
        
        # 复用父类的 embed_tokens
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # --- 多模态处理逻辑 ---
        # 只有当存在视频特征且 input_ids 不是单步生成(shape!=1)或者是训练模式时才执行
        if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
            

            # 🔥🔥🔥【新增修复代码】开始 🔥🔥🔥
            # 获取 Projector 的目标精度 (通常是 float16)
            # target_dtype = self.mm_projector.weight.dtype
            # target_dtype = torch.float16
            target_dtype = torch.bfloat16
            
            # 强制将输入特征转为目标精度
            if video_spatio_temporal_features.dtype != target_dtype:
                video_spatio_temporal_features = video_spatio_temporal_features.to(target_dtype)
                
            if pose_spatio_temporal_features is not None:
                if pose_spatio_temporal_features.dtype != target_dtype:
                    pose_spatio_temporal_features = pose_spatio_temporal_features.to(target_dtype)
            # 🔥🔥🔥【新增修复代码】结束 🔥🔥🔥


            # 1. 特征融合
            if pose_spatio_temporal_features is not None:
                # 正常情况：两个特征都有
                fused_features = self.fuse_features(video_spatio_temporal_features, pose_spatio_temporal_features)
            else:
                # Fallback: 如果只有视频没有 Pose (例如推理时数据缺失)
                # 创建全 0 的 Dummy Pose 进行拼接，保证维度能通过 Projector
                B, T, _ = video_spatio_temporal_features.shape
                dummy_pose = torch.zeros(
                    B, T, self.pose_feature_dim,
                    device=video_spatio_temporal_features.device,
                    dtype=video_spatio_temporal_features.dtype
                )
                fused_features = torch.cat([video_spatio_temporal_features, dummy_pose], dim=-1)

            # 2. 投影到 LLM 空间 [Batch, T, 1092] -> [Batch, T, 4096]
            video_features = self.mm_projector(fused_features)

            # 3. 创建 Dummy 特征 (用于填充非视频 Token 位置)
            # 这里的维度必须匹配融合后的维度 (1092)
            dummy_video_features = torch.zeros(
                video_features.shape[1],
                self.mm_hidden_size + self.pose_feature_dim,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            dummy_video_features = self.mm_projector(dummy_video_features)

            # 4. 将视频特征插入到 inputs_embeds 中
            new_input_embeds = []
            cur_video_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                # 情况 A: 这一条数据里没有视频 token (纯文本)
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # 加入 dummy 梯度以防报错
                    cur_input_embeds = cur_input_embeds + (0. * dummy_video_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_video_idx += 1
                    continue

                # 情况 B: 使用 <vid_start> 和 <vid_end> 包裹视频
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != \
                       (cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    
                    video_start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in video_start_tokens:
                        cur_video_features = video_features[cur_video_idx].to(device=cur_input_embeds.device)
                        
                        # 融合后的特征长度 (min_len)
                        num_patches = cur_video_features.shape[0] 
                        
                        # 拼接: [Start Token] + [Video Features] + [End Token]
                        # 这里的切片逻辑假设 Dataset 中的占位符长度足以容纳 num_patches
                        # 实际上 VideoChatGPT 通常在 Dataset 预处理时就对齐了长度
                        cur_new_input_embeds = torch.cat((
                            cur_input_embeds[:video_start_token_pos + 1],
                            cur_video_features,
                            cur_input_embeds[video_start_token_pos + num_patches + 1:]
                        ), dim=0)
                        
                        cur_video_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                
                # 情况 C: 不使用 Start/End Token (直接替换 Patch Token)
                else:
                    cur_video_features = video_features[cur_video_idx]
                    num_patches = cur_video_features.shape[0]
                    
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                         # 这里做个兼容：如果 Token 数量不匹配，尝试截断或报错
                         # 为了稳健，我们以特征长度为准进行替换
                         pass

                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    
                    cur_new_input_embeds = torch.cat((
                        cur_input_embeds[:mask_index_start],
                        cur_video_features,
                        cur_input_embeds[mask_index_start + num_patches:]
                    ), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_video_idx += 1

            inputs_embeds = torch.stack(new_input_embeds, dim=0)




        # 检查每一个环节
        # print(f"1. Inputs Embeds NaN: {torch.isnan(inputs_embeds).any()}")

        # if video_spatio_temporal_features is not None:
        #     print(f"2. Video Feats NaN: {torch.isnan(video_spatio_temporal_features).any()}")
        #     # 检查 Projector 后的输出
        #     video_features_projected = self.mm_projector(fused_features)
        #     print(f"3. Projected Feats NaN: {torch.isnan(video_features_projected).any()}")

        # # 检查模型权重是否有 NaN
        # for name, param in self.named_parameters():
        #     if param.requires_grad and torch.isnan(param).any():
        #         print(f"❌ Parameter {name} is NaN!")




        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class MultimodalVideoChatGPTLlamaForCausalLM(VideoChatGPTLlamaForCausalLM):
    """
    Multimodal Wrapper: 负责初始化 inner model 并传递 forward 参数
    """
    config_class = VideoChatGPTConfig

    def __init__(self, config, pose_feature_dim: int = 68):
        # 1. 初始化父类 (这会创建原始的 self.model)
        super(LlamaForCausalLM, self).__init__(config)
        
        # 2. 【关键】用我们自定义的多模态模型替换掉父类创建的 self.model
        # 这样 forward 调用时就会走我们上面的 MultimodalVideoChatGPTLlamaModel
        self.model = MultimodalVideoChatGPTLlamaModel(
            config, 
            pose_feature_dim=pose_feature_dim
        )
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
        pose_spatio_temporal_features: Optional[torch.FloatTensor] = None, # 接收 Pose 参数
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用我们自定义的 self.model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            video_spatio_temporal_features=video_spatio_temporal_features,
            pose_spatio_temporal_features=pose_spatio_temporal_features # 传递 Pose 参数
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 确保 Pose 参数在 generate() 时也能被传入
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            "pose_spatio_temporal_features": kwargs.get("pose_spatio_temporal_features", None),
        })
        return model_inputs