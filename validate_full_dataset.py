import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
TARGET_ROOT = "/userhome/cs/u3598820/HKU-FYP25089-VARdict/full_dataset"
SPLIT = "Train"

# Pose 数据期望的 shape: [帧数, 人数, 关键点数, 坐标维度]
EXPECTED_PEOPLE     = 2
EXPECTED_KEYPOINTS  = 17
EXPECTED_COORDS     = 2
MIN_FRAMES          = 1   # 至少要有这么多帧
# ===========================================

def check_npy(path):
    """检查 .npy 文件，返回 (ok, reason)"""
    try:
        arr = np.load(path)
    except Exception as e:
        return False, f"无法读取: {e}"

    if arr.size == 0:
        return False, "空文件 (size=0)"

    if arr.ndim != 4:
        return False, f"维度错误: 期望 4 维，实际 {arr.ndim} 维，shape={arr.shape}"

    frames, people, kpts, coords = arr.shape
    if frames < MIN_FRAMES:
        return False, f"帧数不足: {frames} < {MIN_FRAMES}"
    if people != EXPECTED_PEOPLE:
        return False, f"人数错误: 期望 {EXPECTED_PEOPLE}，实际 {people}"
    if kpts != EXPECTED_KEYPOINTS:
        return False, f"关键点数错误: 期望 {EXPECTED_KEYPOINTS}，实际 {kpts}"
    if coords != EXPECTED_COORDS:
        return False, f"坐标维度错误: 期望 {EXPECTED_COORDS}，实际 {coords}"

    if np.isnan(arr).any():
        return False, "包含 NaN 值"
    if np.isinf(arr).any():
        return False, "包含 Inf 值"

    return True, f"OK  shape={arr.shape}"


def check_pkl(path):
    """检查 .pkl 文件，返回 (ok, reason)"""
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        return False, f"无法读取: {e}"

    if data is None:
        return False, "内容为 None"

    # 如果是 numpy array，检查是否为空
    if isinstance(data, np.ndarray):
        if data.size == 0:
            return False, "空数组 (size=0)"
        if np.isnan(data).any():
            return False, "包含 NaN 值"
        return True, f"OK  shape={data.shape}"

    return True, f"OK  type={type(data).__name__}"


def main():
    split_dir = os.path.join(TARGET_ROOT, SPLIT)
    action_dirs = sorted(glob.glob(os.path.join(split_dir, "action_*")))

    if not action_dirs:
        print(f"错误：在 {split_dir} 下没有找到 action_* 文件夹")
        return

    print(f"扫描目录: {split_dir}")
    print(f"共找到 {len(action_dirs)} 个 action 文件夹\n")

    broken_files = []   # [(path, reason), ...]
    stats = {
        "npy_ok": 0, "npy_bad": 0,
        "pkl_ok": 0, "pkl_bad": 0,
        "action_dirs": len(action_dirs),
    }

    for action_dir in tqdm(action_dirs, desc="验证中"):
        action_name = os.path.basename(action_dir)

        # 检查所有 .npy
        for npy_path in glob.glob(os.path.join(action_dir, "*_pose.npy")):
            ok, reason = check_npy(npy_path)
            if ok:
                stats["npy_ok"] += 1
            else:
                stats["npy_bad"] += 1
                broken_files.append((npy_path, reason))

        # 检查所有 .pkl
        for pkl_path in glob.glob(os.path.join(action_dir, "PRE_CLIP_feature_*.pkl")):
            ok, reason = check_pkl(pkl_path)
            if ok:
                stats["pkl_ok"] += 1
            else:
                stats["pkl_bad"] += 1
                broken_files.append((pkl_path, reason))

    # ---- 汇总报告 ----
    print("\n" + "=" * 60)
    print("验证结果汇总")
    print("=" * 60)
    print(f"  action 文件夹数 : {stats['action_dirs']}")
    print(f"  .npy  正常 / 损坏 : {stats['npy_ok']} / {stats['npy_bad']}")
    print(f"  .pkl  正常 / 损坏 : {stats['pkl_ok']} / {stats['pkl_bad']}")
    total_bad = stats["npy_bad"] + stats["pkl_bad"]
    print(f"  损坏文件总计    : {total_bad}")

    if broken_files:
        print("\n损坏文件列表：")
        print("-" * 60)
        for path, reason in broken_files:
            print(f"  [{reason}]")
            print(f"    {path}")

        # 询问是否自动删除
        print()
        ans = input("是否自动删除以上损坏文件？(输入 yes 确认，其他任意键取消): ").strip().lower()
        if ans == "yes":
            deleted = 0
            for path, _ in broken_files:
                try:
                    os.remove(path)
                    deleted += 1
                    print(f"  已删除: {path}")
                except Exception as e:
                    print(f"  删除失败 {path}: {e}")
            print(f"\n共删除 {deleted} 个文件。请重新运行 build_full_dataset.py 补跑。")
        else:
            print("已取消删除。你可以手动处理上述文件。")
    else:
        print("\n所有文件均完整，无需处理 ✓")


if __name__ == "__main__":
    main()