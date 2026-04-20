import argparse
import hashlib
import html
import os
import pickle
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None
    cpu = None

try:
    from mmpose.apis import MMPoseInferencer
except ImportError:
    MMPoseInferencer = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

from clip_model import CLIPNetwork
from config.classes import INVERSE_EVENT_DICTIONARY
from load_model import safe_load_model
from video_chatgpt.video_conversation import conv_templates

import shutil
import subprocess

try:
    import cv2
except ImportError:
    cv2 = None

# Supported video formats
SUPPORTED_VIDEO_EXTENSIONS = {".mp4"}
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"
VIDEO_TOKEN_LEN = 300
CONV_MODE = "video-chatgpt_v1"

DEFAULT_WELCOME_MESSAGE = (
    "I am the VARdict demo assistant. Upload or select a match clip, "
    "and I can summarize events, explain controversial decisions, or output "
    "a referee analysis suitable for presentation."
)

RUNTIME_CONFIG: Dict[str, Any] = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "base_weights": os.environ.get(
        "VARDICT_BASE_WEIGHTS",
        "/userhome/cs/u3598820/X-VARS_weight/X-VARS_weights",
    ),
    "phase1_path": os.environ.get("VARDICT_PHASE1_PATH", "./checkpoints/phase1/final"),
    "lora_path": os.environ.get("VARDICT_LORA_PATH", "./checkpoints/phase2/final_lora"),
    "clip_weights": os.environ.get("VARDICT_CLIP_WEIGHTS", ""),
    "cache_dir": os.environ.get("VARDICT_DEMO_CACHE_DIR", "./demo_cache"),
    "max_new_tokens": int(os.environ.get("VARDICT_MAX_NEW_TOKENS", "256")),
    "pose_feature_dim": 68,
}

RUNTIME_OBJECTS: Dict[str, Any] = {
    "vardict_model": None,
    "tokenizer": None,
    "clip_model": None,
    "clip_image_processor": None,
    "pose_inferencer": None,
}


@dataclass
class SessionState:
    active_video: Optional[str] = None
    video_source: str = "none"
    video_label: str = "No video selected"
    conversation: List[Tuple[str, str]] = field(default_factory=list)


def init_runtime_from_args(args: argparse.Namespace) -> None:
    RUNTIME_CONFIG["device"] = args.device
    RUNTIME_CONFIG["base_weights"] = args.base_weights
    RUNTIME_CONFIG["phase1_path"] = args.phase1_path
    RUNTIME_CONFIG["lora_path"] = args.lora_path
    RUNTIME_CONFIG["clip_weights"] = args.clip_weights
    RUNTIME_CONFIG["cache_dir"] = args.cache_dir
    RUNTIME_CONFIG["max_new_tokens"] = args.max_new_tokens
    os.makedirs(RUNTIME_CONFIG["cache_dir"], exist_ok=True)


def find_example_videos(examples_dir: Optional[str]) -> Dict[str, str]:
    if not examples_dir:
        return {}

    root = Path(examples_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return {}

    candidates: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            label = path.relative_to(root).as_posix()
            candidates[label] = str(path.resolve())
    return candidates


def build_brand_html(app_name: str, subtitle: str, logo_path: Optional[str]) -> str:
    safe_name = html.escape(app_name)
    safe_subtitle = html.escape(subtitle)

    if logo_path and Path(logo_path).exists():
        safe_logo = html.escape(str(Path(logo_path).resolve())).replace("\\", "/")
        logo_markup = (
            f'<img class="brand-logo-image" src="file/{safe_logo}" alt="{safe_name} logo">'
        )
    else:
        logo_markup = '<div class="brand-logo-fallback">V</div>'

    return f"""
    <div class="brand-shell">
      <div class="brand-mark">{logo_markup}</div>
      <div class="brand-copy">
        <div class="brand-title">{safe_name}</div>
        <div class="brand-subtitle">{safe_subtitle}</div>
      </div>
    </div>
    """


def build_css() -> str:
    return """
    :root {
        --brand-red: #c62828;
        --brand-red-dark: #8e1c1c;
        --brand-cream: #f8f2e7;
        --brand-ink: #171717;
        --brand-muted: #6b6b6b;
        --panel: rgba(255, 255, 255, 0.88);
        --panel-border: rgba(23, 23, 23, 0.08);
        --shadow: 0 18px 40px rgba(43, 18, 18, 0.12);
    }

    .gradio-container {
        background:
            radial-gradient(circle at top left, rgba(198, 40, 40, 0.18), transparent 30%),
            linear-gradient(180deg, #f4efe7 0%, #ebe1d2 100%);
    }

    .app-shell {
        max-width: 1440px;
        margin: 0 auto;
        padding: 20px 10px 28px;
    }

    .brand-shell {
        display: flex;
        align-items: center;
        gap: 18px;
        padding: 22px 26px;
        background:
            linear-gradient(135deg, rgba(255,255,255,0.92), rgba(255,255,255,0.75)),
            linear-gradient(135deg, #efe1d0, #f9f4ec);
        border: 1px solid rgba(23, 23, 23, 0.08);
        border-radius: 28px;
        box-shadow: var(--shadow);
    }

    .brand-mark {
        width: 76px;
        height: 76px;
        border-radius: 24px;
        background: linear-gradient(145deg, var(--brand-red), var(--brand-red-dark));
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
        flex-shrink: 0;
    }

    .brand-logo-fallback {
        color: white;
        font-size: 42px;
        font-weight: 800;
        letter-spacing: 0.04em;
    }

    .brand-logo-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }

    .brand-title {
        font-size: 34px;
        font-weight: 800;
        color: var(--brand-ink);
        line-height: 1.1;
        letter-spacing: 0.02em;
    }

    .brand-subtitle {
        margin-top: 6px;
        font-size: 14px;
        color: var(--brand-muted);
    }

    .card {
        border-radius: 24px !important;
        border: 1px solid var(--panel-border) !important;
        background: var(--panel) !important;
        box-shadow: var(--shadow);
        backdrop-filter: blur(8px);
    }

    .status-box {
        padding: 14px 16px;
        border-radius: 18px;
        background: rgba(23, 23, 23, 0.04);
        border: 1px solid rgba(23, 23, 23, 0.06);
    }

    .hint-box {
        padding: 14px 16px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(198, 40, 40, 0.08), rgba(198, 40, 40, 0.03));
        border: 1px solid rgba(198, 40, 40, 0.12);
        color: var(--brand-ink);
        font-size: 13px;
        line-height: 1.6;
    }

    #chatbot-panel {
        min-height: 620px;
    }

    @media (max-width: 960px) {
        .app-shell {
            padding: 12px 6px 20px;
        }

        .brand-shell {
            padding: 18px;
        }

        .brand-title {
            font-size: 28px;
        }
    }

    .markdown-title-padding {
    padding-left: 10px;
    padding-right: 10px;
    padding-top: 5px;
    }

    .markdown-padding {
    padding-left: 10px;
    padding-right: 10px;
    }
    """


def make_status_markdown(state: SessionState) -> str:
    video_text = html.escape(state.video_label)
    source_text = html.escape(state.video_source)
    active_path = html.escape(state.active_video or "None")
    turns = len(state.conversation)
    return (
        f"<div class='status-box'>"
        f"<b>Current Video</b><br>{video_text}<br><br>"
        f"<b>Source</b><br>{source_text}<br><br>"
        f"<b>File Path</b><br><code>{active_path}</code><br><br>"
        f"<b>Conversation Turns</b><br>{turns}"
        f"</div>"
    )


def normalize_uploaded_video(video_path: Optional[str]) -> Optional[str]:
    if not video_path:
        return None
    candidate = Path(video_path)
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())
    return None


def activate_example_video(selected_label: Optional[str], examples: Dict[str, str], state: SessionState):
    if not selected_label:
        return None, state, render_chatbot_pairs(state.conversation), make_status_markdown(state), "Please select an example video first."

    video_path = examples.get(selected_label)
    if not video_path:
        return None, state, render_chatbot_pairs(state.conversation), make_status_markdown(state), "Example video not found. Try refreshing."

    state.active_video = video_path
    state.video_source = "example"
    state.video_label = selected_label
    state.conversation = []   # 新视频 => 清空旧对话

    return (
        video_path,
        state,
        render_chatbot_pairs(state.conversation),
        make_status_markdown(state),
        f"Loaded example video: {selected_label}",
    )


def _copy_video_to_cache(src_path: str) -> str:
    src = Path(src_path).expanduser()
    if not src.exists() or not src.is_file():
        raise RuntimeError(f"Uploaded video file not found: {src_path}")

    uploads_dir = Path(RUNTIME_CONFIG["cache_dir"]) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.md5(str(src.resolve()).encode("utf-8")).hexdigest()[:12]
    suffix = src.suffix if src.suffix else ".mp4"
    dst = uploads_dir / f"{src.stem}_{digest}{suffix}"

    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)

    return str(dst.resolve())


def activate_uploaded_video(uploaded_video, state: SessionState):
    if not uploaded_video:
        return state, render_chatbot_pairs(state.conversation), make_status_markdown(state), "Please upload a playable video first."

    try:
        stable_video_path = _copy_video_to_cache(uploaded_video)
    except Exception as exc:
        return state, render_chatbot_pairs(state.conversation), make_status_markdown(state), f"Upload received, but failed to cache the video: {exc}"

    state.active_video = stable_video_path
    state.video_source = "upload"
    state.video_label = Path(stable_video_path).name if stable_video_path else "Uploaded Video"
    state.conversation = []   # 新视频 => 清空旧对话

    return state, render_chatbot_pairs(state.conversation), make_status_markdown(state), f"Uploaded video cached: {state.video_label}"

def refresh_example_choices(examples_dir: Optional[str]):
    examples = find_example_videos(examples_dir)
    choices = list(examples.keys())
    info = f"Found {len(choices)} example videos." if choices else "No example videos found in the directory."
    return examples, gr.update(choices=choices, value=choices[0] if choices else None), info


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _load_vardict_model_and_tokenizer():
    if RUNTIME_OBJECTS["vardict_model"] is not None and RUNTIME_OBJECTS["tokenizer"] is not None:
        return RUNTIME_OBJECTS["vardict_model"], RUNTIME_OBJECTS["tokenizer"]

    load_path = (
        RUNTIME_CONFIG["phase1_path"]
        if Path(RUNTIME_CONFIG["phase1_path"]).exists()
        else RUNTIME_CONFIG["base_weights"]
    )

    model, tokenizer = safe_load_model(
        load_path,
        pose_feature_dim=RUNTIME_CONFIG["pose_feature_dim"],
        device=RUNTIME_CONFIG["device"],
    )

    model.get_model().initialize_vision_modules(pretrain_mm_mlp_adapter=None)
    vision_config = model.get_model().vision_config
    model.config.tune_mm_mlp_adapter = True
    model.config.freeze_mm_mlp_adapter = False
    model.config.mm_use_vid_start_end = True
    vision_config.use_vid_start_end = True
    model.config.sep_video_conv_front = False

    model.initialize_vision_tokenizer(
        mm_use_vid_start_end=True,
        tokenizer=tokenizer,
        device=RUNTIME_CONFIG["device"],
        tune_mm_mlp_adapter=False,
        pretrain_mm_mlp_adapter=None,
    )
    model.resize_token_embeddings(len(tokenizer))

    lora_path = RUNTIME_CONFIG["lora_path"]
    if lora_path and Path(lora_path).exists():
        if PeftModel is None:
            raise RuntimeError("peft is not installed, but --lora-path was provided.")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    RUNTIME_OBJECTS["vardict_model"] = model
    RUNTIME_OBJECTS["tokenizer"] = tokenizer
    return model, tokenizer


def _load_clip_components():
    if (
        RUNTIME_OBJECTS["clip_model"] is not None
        and RUNTIME_OBJECTS["clip_image_processor"] is not None
    ):
        return RUNTIME_OBJECTS["clip_model"], RUNTIME_OBJECTS["clip_image_processor"]

    clip_weights = RUNTIME_CONFIG["clip_weights"]
    if not clip_weights:
        raise RuntimeError("Missing CLIP weights. Set --clip-weights or VARDICT_CLIP_WEIGHTS.")

    clip_model = CLIPNetwork().to(RUNTIME_CONFIG["device"])
    load = torch.load(clip_weights, map_location=RUNTIME_CONFIG["device"])
    state_dict = load["state_dict"] if isinstance(load, dict) and "state_dict" in load else load
    clip_model.load_state_dict(state_dict)
    clip_model.eval()

    image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        torch_dtype=torch.float16,
    )

    RUNTIME_OBJECTS["clip_model"] = clip_model
    RUNTIME_OBJECTS["clip_image_processor"] = image_processor
    return clip_model, image_processor


def _load_pose_inferencer():
    if RUNTIME_OBJECTS["pose_inferencer"] is not None:
        return RUNTIME_OBJECTS["pose_inferencer"]

    if MMPoseInferencer is None:
        raise RuntimeError("mmpose is not installed, so pose extraction cannot run.")

    inferencer = MMPoseInferencer("human", device=RUNTIME_CONFIG["device"])
    RUNTIME_OBJECTS["pose_inferencer"] = inferencer
    return inferencer


def get_seq_frames(total_num_frames: int, desired_num_frames: int) -> List[int]:
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq: List[int] = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)
    return seq


def _decode_frames_with_opencv(video_path: str, num_frm: int = 1000) -> List[Image.Image]:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed for fallback decoding.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    total_frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frame_num <= 0:
        frames = []
        while len(frames) < num_frm:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frames.append(Image.fromarray(frame))
        cap.release()
        if not frames:
            raise RuntimeError(f"OpenCV read 0 frames from: {video_path}")
        return frames

    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    frames = []
    for idx in frame_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
        frames.append(Image.fromarray(frame))
    cap.release()

    if not frames:
        raise RuntimeError(f"OpenCV failed to decode sampled frames from: {video_path}")
    return frames


def _ffmpeg_reencode_video(src_path: str) -> Optional[str]:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return None

    src = Path(src_path)
    repaired_dir = Path(RUNTIME_CONFIG["cache_dir"]) / "reencoded"
    repaired_dir.mkdir(parents=True, exist_ok=True)
    repaired_path = repaired_dir / f"{src.stem}_fixed.mp4"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(repaired_path),
    ]

    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except Exception:
        return None

    return str(repaired_path.resolve()) if repaired_path.exists() else None


def _ensure_readable_video_path(video_path: str) -> str:
    path = Path(video_path).expanduser()
    if not path.exists():
        raise RuntimeError(f"Video path does not exist: {video_path}")

    try:
        if VideoReader is not None and cpu is not None:
            vr = VideoReader(str(path), ctx=cpu(0))
            if len(vr) > 0:
                return str(path.resolve())
    except Exception:
        pass

    repaired = _ffmpeg_reencode_video(str(path))
    if repaired is not None:
        try:
            if VideoReader is not None and cpu is not None:
                vr = VideoReader(repaired, ctx=cpu(0))
                if len(vr) > 0:
                    return repaired
        except Exception:
            pass
        if cv2 is not None:
            cap = cv2.VideoCapture(repaired)
            ok = cap.isOpened()
            cap.release()
            if ok:
                return repaired

    if cv2 is not None:
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        cap.release()
        if ok:
            return str(path.resolve())

    raise RuntimeError(
        f"Video decode failed for {video_path}. decord could not open it, and fallback repair/decode also failed."
    )


def load_video_frames(video_path: str, num_frm: int = 1000) -> List[Image.Image]:
    readable_path = _ensure_readable_video_path(video_path)

    if VideoReader is not None and cpu is not None:
        try:
            vr = VideoReader(readable_path, ctx=cpu(0))
            total_frame_num = len(vr)
            if total_frame_num > 0:
                total_num_frm = min(total_frame_num, num_frm)
                frame_idx = get_seq_frames(total_frame_num, total_num_frm)
                img_array = vr.get_batch(frame_idx).asnumpy()

                h, w = 224, 224
                if img_array.shape[1] != h or img_array.shape[2] != w:
                    img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
                    img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
                    img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

                return [Image.fromarray(img_array[i]) for i in range(img_array.shape[0])]
        except Exception:
            pass

    return _decode_frames_with_opencv(readable_path, num_frm=num_frm)


def get_spatio_temporal_features(features: np.ndarray, num_temporal_tokens: int = 44) -> np.ndarray:
    t, s, _ = features.shape
    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode="constant")
    spatial_tokens = np.mean(features, axis=0)
    return np.concatenate([temporal_tokens, spatial_tokens], axis=0)


def _format_prediction_text(stage1_prior: Dict[str, str]) -> str:
    pred_action = stage1_prior.get("Action class", "unknown")
    pred_off = stage1_prior.get("Offence", "unknown")
    pred_card = str(stage1_prior.get("Severity", "unknown"))

    if pred_off == "Offence":
        pred_off = ", foul and "
    elif pred_off == "No offence":
        pred_off = "and no foul."

    if pred_card in {"1.0", "1"}:
        pred_off += "no card."
    elif pred_card in {"3.0", "3"}:
        pred_off += "a yellow card."
    elif pred_card in {"5.0", "5"}:
        pred_off += "a red card."

    action_map = {
        "Tackling": "a tackle ",
        "Standing tackling": "a foot duel ",
        "Elbowing": "using his elbows or arms ",
        "Holding": "holding ",
        "High leg": "a high leg ",
        "Pushing": "pushing ",
        "Challenge": "a shoulder challenge ",
        "Dive": "a simulation ",
    }
    pred_action = action_map.get(pred_action, pred_action + " ")
    return pred_action + pred_off


def _align_clip_length(clip_features: np.ndarray, target_len: int = VIDEO_TOKEN_LEN) -> np.ndarray:
    clip_features = np.asarray(clip_features, dtype=np.float32)
    if clip_features.ndim != 2:
        raise RuntimeError(f"CLIP features should be 2D [T, C], got shape={clip_features.shape}")

    cur_len = clip_features.shape[0]
    if cur_len == target_len:
        return clip_features
    if cur_len > target_len:
        return clip_features[:target_len]

    pad = np.zeros((target_len - cur_len, clip_features.shape[1]), dtype=clip_features.dtype)
    return np.concatenate([clip_features, pad], axis=0)


def extract_clip_features(video_path: str) -> Tuple[np.ndarray, Dict[str, str]]:
    clip_model, image_processor = _load_clip_components()

    start_frame = 63
    end_frame = 87
    fps = 17
    fps_beginning = 25
    factor = (end_frame - start_frame) / (((end_frame - start_frame) / fps_beginning) * fps)

    video_frames = load_video_frames(video_path)
    if len(video_frames) <= start_frame:
        raise RuntimeError(f"Video is too short for CLIP feature extraction: {video_path}")

    video_frames = video_frames[start_frame:end_frame]
    if not video_frames:
        raise RuntimeError(f"No frames left after slicing for CLIP extraction: {video_path}")

    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"]
    sampled_frames: List[torch.Tensor] = []
    for j in range(len(frames)):
        if j % factor < 1:
            sampled_frames.append(frames[j])
    if not sampled_frames:
        sampled_frames = [frames[0]]

    final_frames = torch.stack(sampled_frames, dim=0).to(RUNTIME_CONFIG["device"])

    with torch.inference_mode():
        out_off, out_act, video_features = clip_model(final_frames)

    preds_sev = int(torch.argmax(out_off.detach().cpu(), 0).item())
    preds_act = int(torch.argmax(out_act.detach().cpu(), 0).item())

    values: Dict[str, str] = {}
    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act]
    if preds_sev == 0:
        values["Offence"] = "No offence"
        values["Severity"] = ""
    elif preds_sev == 1:
        values["Offence"] = "Offence"
        values["Severity"] = "1.0"
    elif preds_sev == 2:
        values["Offence"] = "Offence"
        values["Severity"] = "3.0"
    else:
        values["Offence"] = "Offence"
        values["Severity"] = "5.0"

    video_np = video_features.detach().cpu().numpy().astype("float16")
    clip_features = get_spatio_temporal_features(video_np)
    clip_features = _align_clip_length(clip_features, VIDEO_TOKEN_LEN)
    return clip_features, values


def extract_pose_array(video_path: str) -> np.ndarray:
    inferencer = _load_pose_inferencer()
    result_generator = inferencer(video_path, return_vis=False, batch_size=4)

    all_frames_data: List[np.ndarray] = []
    max_people = 2
    num_keypoints = 17

    for result in result_generator:
        predictions = result["predictions"][0]
        frame_keypoints = []

        for person in predictions:
            kpts = np.array(person["keypoints"], dtype=np.float32)
            frame_keypoints.append(kpts)

        tensor_frame = np.zeros((max_people, num_keypoints, 2), dtype=np.float32)
        if len(frame_keypoints) > 0:
            actual_people = min(len(frame_keypoints), max_people)
            for i in range(actual_people):
                tensor_frame[i] = frame_keypoints[i]

        all_frames_data.append(tensor_frame)

    if not all_frames_data:
        raise RuntimeError(f"Pose extraction produced 0 frames: {video_path}")

    return np.stack(all_frames_data, axis=0)


def _make_cache_dir_for_video(video_path: str) -> Path:
    resolved = str(Path(video_path).resolve())
    key = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:12]
    sample_dir = Path(RUNTIME_CONFIG["cache_dir"]) / f"{Path(video_path).stem}_{key}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def prepare_features_for_video(video_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    video_path = _ensure_readable_video_path(video_path)
    sample_dir = _make_cache_dir_for_video(video_path)
    clip_pkl = sample_dir / f"PRE_CLIP_feature_{Path(video_path).stem}.pkl"
    pose_npy = sample_dir / f"{Path(video_path).stem}_pose.npy"
    prior_pkl = sample_dir / "stage1_prior.pkl"

    if clip_pkl.exists() and prior_pkl.exists():
        with open(clip_pkl, "rb") as f:
            clip_features = pickle.load(f)
        with open(prior_pkl, "rb") as f:
            stage1_prior = pickle.load(f)
    else:
        clip_features, stage1_prior = extract_clip_features(video_path)
        with open(clip_pkl, "wb") as f:
            pickle.dump(clip_features, f)
        with open(prior_pkl, "wb") as f:
            pickle.dump(stage1_prior, f)

    if pose_npy.exists():
        pose_array = np.load(pose_npy)
    else:
        pose_array = extract_pose_array(video_path)
        np.save(pose_npy, pose_array)

    return np.asarray(clip_features), np.asarray(pose_array), stage1_prior


def preprocess_pose_like_dataset(pose_array: np.ndarray, clip_len: int) -> torch.Tensor:
    if pose_array.ndim != 4:
        raise RuntimeError(f"Pose array should be [T, 2, 17, 2], got shape={pose_array.shape}")

    if pose_array.shape[0] > 0:
        pose_flat = pose_array.reshape(pose_array.shape[0], -1)
    else:
        pose_flat = np.zeros((1, 68), dtype=np.float32)

    pose_tensor = torch.from_numpy(pose_flat).float()
    if pose_tensor.numel() > 0:
        pose_tensor = pose_tensor / (pose_tensor.abs().max() + 1e-6)

    pose_tensor = pose_tensor.permute(1, 0).unsqueeze(0)
    pose_tensor = torch.nn.functional.interpolate(
        pose_tensor,
        size=clip_len,
        mode="linear",
        align_corners=False,
    )
    pose_tensor = pose_tensor.squeeze(0).permute(1, 0)
    return pose_tensor


def preprocess_clip_like_dataset(clip_features: np.ndarray) -> torch.Tensor:
    clip_tensor = torch.from_numpy(np.asarray(clip_features)).float()
    clip_tensor = _align_clip_length(clip_tensor.numpy(), VIDEO_TOKEN_LEN)
    clip_tensor = torch.from_numpy(clip_tensor).float()
    if clip_tensor.numel() > 0:
        clip_tensor = torch.nn.functional.normalize(clip_tensor, p=2, dim=-1)
    return clip_tensor


def compact_conversation_for_prompt(
    conversation: List[Tuple[str, Optional[str]]],
    keep_last_followups: int = 2,
) -> List[Tuple[str, Optional[str]]]:
    # 保留首轮（因为首轮要放视频 token）+ 最近几轮
    if len(conversation) <= keep_last_followups + 1:
        return conversation
    return [conversation[0]] + conversation[-keep_last_followups:]


def build_inference_prompt(
    conversation: List[Tuple[str, Optional[str]]],
    stage1_prior: Dict[str, str],
) -> str:
    prediction_text = _format_prediction_text(stage1_prior)
    conv = conv_templates[CONV_MODE].copy()

    turns = compact_conversation_for_prompt(conversation)

    for idx, (user_msg, assistant_msg) in enumerate(turns):
        if idx == 0:
            # 只在首轮注入视频 token 和 stage-1 prior
            user_text = (
                user_msg
                + " The prediction for this video is "
                + prediction_text
                + "\n"
                + DEFAULT_VID_START_TOKEN
                + DEFAULT_VIDEO_PATCH_TOKEN * VIDEO_TOKEN_LEN
                + DEFAULT_VID_END_TOKEN
            )
        else:
            user_text = user_msg

        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], assistant_msg)

    return conv.get_prompt()


def run_vardict_inference(
    video_path: str,
    conversation: List[Tuple[str, Optional[str]]],
) -> Tuple[str, Dict[str, str]]:
    video_path = _ensure_readable_video_path(video_path)
    model, tokenizer = _load_vardict_model_and_tokenizer()
    device = _model_device(model)

    clip_features, pose_array, stage1_prior = prepare_features_for_video(video_path)
    clip_tensor = preprocess_clip_like_dataset(clip_features)
    pose_tensor = preprocess_pose_like_dataset(pose_array, clip_tensor.shape[0])

    prompt = build_inference_prompt(conversation, stage1_prior)

    max_ctx = int(getattr(tokenizer, "model_max_length", 2048))
    requested_new_tokens = int(RUNTIME_CONFIG["max_new_tokens"])
    max_prompt_tokens = max(512, max_ctx - requested_new_tokens - 8)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        max_length=max_prompt_tokens,
        truncation=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    video_spatio_temporal_features = clip_tensor.unsqueeze(0).to(device)
    pose_spatio_temporal_features = pose_tensor.unsqueeze(0).to(device)

    available_new_tokens = max(1, max_ctx - input_ids.shape[1] - 1)
    max_new_tokens = min(requested_new_tokens, available_new_tokens)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            video_spatio_temporal_features=video_spatio_temporal_features,
            pose_spatio_temporal_features=pose_spatio_temporal_features,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not answer:
        answer = "The model ran, but the decoded answer is empty."

    return answer, stage1_prior


def reset_chat(state: SessionState):
    state.conversation = []
    chatbot = render_chatbot_pairs(state.conversation)
    return state, chatbot, make_status_markdown(state), "Conversation cleared."


def render_chatbot_pairs(conversation: List[Tuple[str, str]]) -> List[dict]:
    messages: List[dict] = [{"role": "assistant", "content": DEFAULT_WELCOME_MESSAGE}]
    for user_msg, bot_msg in conversation:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    return messages


def submit_message(user_message: str, state: SessionState):
    cleaned = (user_message or "").strip()
    if not cleaned:
        return "", state, render_chatbot_pairs(state.conversation), make_status_markdown(state), "Please enter a question first."

    if not state.active_video:
        return "", state, render_chatbot_pairs(state.conversation), make_status_markdown(state), "Please select or upload a video first."

    pending_conversation = list(state.conversation) + [(cleaned, None)]

    try:
        answer, stage1_prior = run_vardict_inference(state.active_video, pending_conversation)
        state.conversation.append((cleaned, answer))
        status = (
            "Inference finished. "
            f"Stage-1 prior: {stage1_prior.get('Action class', 'Unknown')}, "
            f"{stage1_prior.get('Offence', 'Unknown')}, "
            f"{stage1_prior.get('Severity', 'None') or 'None'}"
        )
    except Exception as exc:
        answer = (
            "Inference failed.\n\n"
            f"Error: {exc}\n\n"
            "Traceback:\n"
            f"{traceback.format_exc()}"
        )
        state.conversation.append((cleaned, answer))
        status = "Inference failed."

    return "", state, render_chatbot_pairs(state.conversation), make_status_markdown(state), status


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    initial_examples = find_example_videos(args.examples_dir)
    initial_choices = list(initial_examples.keys())
    initial_state = SessionState()

    with gr.Blocks(title=args.app_name) as demo:
        example_store = gr.State(initial_examples)
        session_state = gr.State(initial_state)

        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(build_brand_html(args.app_name, args.subtitle, args.logo_path))

            with gr.Row(equal_height=False):
                with gr.Column(scale=7):
                    with gr.Column(elem_classes=["card"]):
                        gr.Markdown("### Video Panel", elem_classes=["markdown-title-padding"])
                        gr.Markdown(
                            "Load example videos or upload local videos on the left. "
                            "The video will be previewed below and set as current conversation context.",
                            elem_classes=["markdown-padding"],
                        )

                        with gr.Row():
                            example_dropdown = gr.Dropdown(
                                label="Example Video",
                                choices=initial_choices,
                                value=initial_choices[0] if initial_choices else None,
                                allow_custom_value=False,
                            )
                            refresh_button = gr.Button("Refresh Examples")
                            load_example_button = gr.Button("Load Example", variant="secondary")

                        video_player = gr.Video(
                            label="Match Video",
                            value=None,
                            sources=["upload"],
                            format="mp4",
                            height=420,
                            interactive=True,
                        )

                        with gr.Row():
                            use_upload_button = gr.Button("Use Uploaded Video", variant="primary")
                            clear_chat_button = gr.Button("Clear Conversation")

                        status_text = gr.Markdown("Waiting for video selection...", elem_classes=["markdown-padding"])
                        info_panel = gr.HTML(make_status_markdown(initial_state))

                        gr.HTML(
                            "<div class='hint-box'>"
                            "<b>Deployment Tip</b><br>"
                            "After server starts, access via SSH port forwarding, e.g.:<br>"
                            "<code>ssh -N -L 7860:&lt;COMPUTE_NODE_IP&gt;:7860 -J "
                            "&lt;USERNAME&gt;@gpu2gate1.cs.hku.hk &lt;USERNAME&gt;@&lt;COMPUTE_NODE_IP&gt;</code>"
                            "</div>"
                        )

                with gr.Column(scale=8):
                    with gr.Column(elem_classes=["card"], elem_id="chatbot-panel"):
                        gr.Markdown("### Referee Chat", elem_classes=["markdown-title-padding"])
                        gr.Markdown(
                            "Ask VARdict to summarize controversial clips, explain decisions, "
                            "or output a referee commentary suitable for demo presentation.",
                            elem_classes=["markdown-padding"],
                        )

                        chatbot = gr.Chatbot(
                            label="VARdict Assistant",
                            value=render_chatbot_pairs(initial_state.conversation),
                            height=520,
                        )

                        question_box = gr.Textbox(
                            label="Your Question",
                            placeholder="For example: Should this be a penalty? Explain the referee reasoning.",
                            lines=3,
                        )

                        with gr.Row():
                            send_button = gr.Button("Send", variant="primary")
                            reset_button = gr.Button("Reset Conversation", variant="secondary")

        load_example_button.click(
            fn=activate_example_video,
            inputs=[example_dropdown, example_store, session_state],
            outputs=[video_player, session_state, chatbot, info_panel, status_text],
        )

        refresh_button.click(
            fn=lambda: refresh_example_choices(args.examples_dir),
            inputs=[],
            outputs=[example_store, example_dropdown, status_text],
        )

        use_upload_button.click(
            fn=activate_uploaded_video,
            inputs=[video_player, session_state],
            outputs=[session_state, chatbot, info_panel, status_text],
        )

        send_button.click(
            fn=submit_message,
            inputs=[question_box, session_state],
            outputs=[question_box, session_state, chatbot, info_panel, status_text],
        )

        question_box.submit(
            fn=submit_message,
            inputs=[question_box, session_state],
            outputs=[question_box, session_state, chatbot, info_panel, status_text],
        )

        clear_chat_button.click(
            fn=reset_chat,
            inputs=[session_state],
            outputs=[session_state, chatbot, info_panel, status_text],
        )

        reset_button.click(
            fn=reset_chat,
            inputs=[session_state],
            outputs=[session_state, chatbot, info_panel, status_text],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VARdict Gradio demo launcher")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--examples-dir", default=os.environ.get("VARDICT_EXAMPLES_DIR", "/userhome/cs/u3598820/HKU-FYP25089-VARdict/demo-example"))
    parser.add_argument("--logo-path", default=os.environ.get("VARDICT_LOGO_PATH"))
    parser.add_argument("--app-name", default="VARdict")
    parser.add_argument(
        "--subtitle",
        default="AI Video Referee Demo for football incident review",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--base-weights",
        default=os.environ.get(
            "VARDICT_BASE_WEIGHTS",
            "/userhome/cs/u3598820/X-VARS_weight/X-VARS_weights",
        ),
    )
    parser.add_argument("--phase1-path", default=os.environ.get("VARDICT_PHASE1_PATH", "/userhome/cs/u3598820/HKU-FYP25089-VARdict/checkpoints/phase1/final"))
    parser.add_argument("--lora-path", default=os.environ.get("VARDICT_LORA_PATH", "/userhome/cs/u3598820/HKU-FYP25089-VARdict/checkpoints/phase2/checkpoint-463"))
    parser.add_argument("--clip-weights", default=os.environ.get("VARDICT_CLIP_WEIGHTS", "/userhome/cs/u3598820/14_model.pth.tar"))
    parser.add_argument("--cache-dir", default=os.environ.get("VARDICT_DEMO_CACHE_DIR", "./demo_cache"))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VARDICT_MAX_NEW_TOKENS", "256")))
    return parser.parse_args()


def main():
    args = parse_args()
    init_runtime_from_args(args)
    demo = build_demo(args)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        css=build_css(),
    )


if __name__ == "__main__":
    main()
