## 사용 방법
## python distribution_map_GPU.py --target_name book_1

import os
import argparse
import numpy as np
import cv2
import json
import torch

# ==============================================================================
# 파라미터 설정 (이 블록에서 모두 수정 가능)
# ==============================================================================

# --- Occlusion map 생성 ---
OCCLUSION_THRESHOLD  = 0.7   # 이 비율 이상 가려져야 occluded로 간주

# --- Distribution map 결합 ---
BETA                 = 0.7   # occlusion map 가중치 (similarity = 1-BETA)

# --- Visibility check (target 강조 적용 기준) ---
VISIBILITY_THRESHOLD = 0.3   # scene에서 target 픽셀의 몇 % 이상 보여야 강조 적용
DIM_FACTOR           = 0.7   # 강조 적용 시 배경을 어둡게 하는 비율

# --- GPU 배치 설정 ---
# SCENE_GPU_BATCH : 한 번에 VRAM에 올릴 scene 수.
#   메모리 계산: SCENE_GPU_BATCH × H × W × 4 bytes (float32)
#   예) 256 × 480 × 640 × 4 ≈ 300 MB
#   VRAM 부족 시 줄이세요 (최소 16 권장).
SCENE_GPU_BATCH  = 256

# TARGET_GPU_BATCH : 한 번에 disk에서 읽어 GPU에 올릴 target 수.
#   이 값을 크게 하면 disk I/O 횟수가 줄어들지만 CPU RAM을 더 사용합니다.
TARGET_GPU_BATCH = 1024

# ==============================================================================
# Argument parsing
# ==============================================================================
parser = argparse.ArgumentParser(description="Distribution Map Generator - GPU Version (Isaac Sim)")
parser.add_argument("--target_name", type=str, required=True,
                    help="타겟 오브젝트 폴더 이름 (예: book_1)")
args = parser.parse_args()

# ==============================================================================
# 경로 설정
# ==============================================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output", args.target_name)

target_base    = os.path.join(OUTPUT_DIR, "target")
scene_base     = os.path.join(OUTPUT_DIR, "scene")
sim_map_dir    = os.path.join(OUTPUT_DIR, "similarity_map")
dist_map_dir   = os.path.join(OUTPUT_DIR, "distribution_map")

target_seg_dir   = os.path.join(target_base, "seg")
target_depth_dir = os.path.join(target_base, "depth")
target_proc_dir  = os.path.join(target_base, "processed_depth")

scene_seg_dir    = os.path.join(scene_base, "seg")
scene_depth_dir  = os.path.join(scene_base, "depth")
scene_proc_dir   = os.path.join(scene_base, "processed_depth")
scene_dis_dir         = os.path.join(scene_base, "depth_dis_map")
empty_scene_depth_dir = os.path.join(target_base, "empty_scene", "depth")

# ==============================================================================
# target 정보 로드
# ==============================================================================
with open(os.path.join(target_base, "mapping.json")) as f:
    target_mapping = json.load(f)

target_usd_name = target_mapping["target_usd_name"]

target_bgr_in_target = tuple(target_mapping["classes"][target_usd_name]["color_bgr"])

scene_json_files = [f for f in os.listdir(scene_seg_dir) if f.endswith("_mapping.json")]
target_bgr_in_scene = None
for jf in sorted(scene_json_files):
    with open(os.path.join(scene_seg_dir, jf)) as f:
        m = json.load(f)
    if target_usd_name in m:
        target_bgr_in_scene = tuple(m[target_usd_name]["color_rgb"])
        break

if target_bgr_in_scene is None:
    raise RuntimeError(f"scene mapping JSON에서 '{target_usd_name}'을 찾을 수 없습니다. "
                       f"scene 데이터에 target이 포함되어 있는지 확인하세요.")

CAM_NAMES = ["center", "left", "right", "top", "bottom"]

print(f"=== Distribution Map Generator - GPU Version ===")
print(f"  타겟       : {args.target_name} ({target_usd_name})")
print(f"  BGR (target/seg) : {target_bgr_in_target}")
print(f"  BGR (scene/seg)  : {target_bgr_in_scene}")
print(f"  OCCLUSION_THRESHOLD = {OCCLUSION_THRESHOLD}")
print(f"  BETA = {BETA}  (similarity = {1 - BETA})")
print(f"  VISIBILITY_THRESHOLD = {VISIBILITY_THRESHOLD},  DIM_FACTOR = {DIM_FACTOR}")
print(f"  SCENE_GPU_BATCH = {SCENE_GPU_BATCH},  TARGET_GPU_BATCH = {TARGET_GPU_BATCH}")


# ==============================================================================
# 유틸: seg 이미지에서 target BGR 색상 픽셀 마스크 반환
# ==============================================================================
def get_target_mask(seg_img: np.ndarray, bgr: tuple) -> np.ndarray:
    b, g, r = bgr
    return (
        (seg_img[:, :, 0] == b) &
        (seg_img[:, :, 1] == g) &
        (seg_img[:, :, 2] == r)
    )


# ==============================================================================
# Step 1: combine_all_images
#   각 (seg, depth) 쌍에서 target 픽셀만 남긴 masked depth를 개별 파일로 저장
# ==============================================================================
def combine_all_images(obj_type: str, bgr: tuple):
    if obj_type == "target":
        seg_dir   = target_seg_dir
        depth_dir = target_depth_dir
        out_dir   = target_proc_dir
    else:
        seg_dir   = scene_seg_dir
        depth_dir = scene_depth_dir
        out_dir   = scene_proc_dir

    os.makedirs(out_dir, exist_ok=True)

    seg_files = sorted(f for f in os.listdir(seg_dir) if f.endswith(".png"))
    total = len(seg_files)
    print(f"\n[{obj_type}] masked depth 생성 ({total}장)...")

    skipped = 0
    for i, seg_fname in enumerate(seg_files):
        base        = os.path.splitext(seg_fname)[0]
        depth_fname = base + ".npy"
        out_path    = os.path.join(out_dir, depth_fname)

        if os.path.exists(out_path):
            skipped += 1
            continue

        depth_path = os.path.join(depth_dir, depth_fname)
        if not os.path.exists(depth_path):
            continue

        seg_img = cv2.imread(os.path.join(seg_dir, seg_fname))
        if seg_img is None:
            continue

        depth = np.load(depth_path).squeeze().astype(np.float32)
        mask  = get_target_mask(seg_img, bgr).astype(np.float32)

        masked_depth = np.nan_to_num(depth * mask, nan=0.0)
        np.save(out_path, masked_depth)

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"  [{i+1:6d}/{total}] 완료")

    if skipped > 0:
        print(f"  (기존 파일 {skipped}장 건너뜀)")


# ==============================================================================
# Step 2: compute_ref_pixels (visibility check용)
#   각 카메라 방향별 target 픽셀 수 최댓값
# ==============================================================================
def compute_ref_pixels() -> dict:
    import json
    cache_path = os.path.join(target_proc_dir, "_ref_pixels.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            ref = json.load(f)
        print(f"\n[ref_pixels] 캐시 로드: {cache_path}")
        for cam, val in ref.items():
            print(f"  ref_pixels[{cam:6s}] = {val}")
        return ref

    ref = {cam: 0 for cam in CAM_NAMES}

    seg_files = sorted(f for f in os.listdir(target_seg_dir) if f.endswith(".png"))
    total = len(seg_files)
    print(f"\n[ref_pixels] 계산 중 ({total}장)...")

    for i, fname in enumerate(seg_files):
        cam = fname.rsplit("_", 1)[1].replace(".png", "")
        if cam not in ref:
            continue

        seg_img = cv2.imread(os.path.join(target_seg_dir, fname))
        if seg_img is None:
            continue

        count = int(np.sum(get_target_mask(seg_img, target_bgr_in_target)))
        if count > ref[cam]:
            ref[cam] = count

        if (i + 1) % 10000 == 0 or (i + 1) == total:
            print(f"  [{i+1:6d}/{total}]")

    for cam, val in ref.items():
        print(f"  ref_pixels[{cam:6s}] = {val}")

    with open(cache_path, "w") as f:
        json.dump(ref, f, indent=2)
    print(f"  → 캐시 저장: {cache_path}")
    return ref


# ==============================================================================
# Step 3: create_depth_distribution_map (GPU 가속)
#
# [최적화 전략]
#   CPU 버전 (원본):
#     for scene:                    # M번
#       for target:                 # N번
#         비교 (CPU)
#     → disk 읽기: M × N  (예: 1000 × 70000 = 70M 회)
#
#   GPU 버전:
#     for cam:
#       for scene_batch (SCENE_GPU_BATCH씩):          # ceil(M/SCENE_GPU_BATCH) 회
#         scenes → VRAM
#         for target_batch (TARGET_GPU_BATCH씩):      # ceil(N/TARGET_GPU_BATCH) 회
#           targets → VRAM
#           GPU 벡터화: 각 target × scene_batch 동시 비교 (inner Python loop 제거)
#     → disk 읽기: N × ceil(M/SCENE_GPU_BATCH) + M
#        예: 70000 × 4 + 1000 = 281000 회  (약 250배 감소)
#
# [메모리 사용량 (SCENE_GPU_BATCH=256, TARGET_GPU_BATCH=64, 480×640)]
#   scenes_gpu : 256 × 480 × 640 × 4  ≈  300 MB
#   sum_gpu    : 256 × 480 × 640 × 8  ≈  600 MB (float64)
#   targets_gpu:  64 × 480 × 640 × 4  ≈   75 MB
#   임시 텐서  : 256 × 480 × 640 × 8  ≈  600 MB (above_f × target_k 연산 시)
#   총 피크    : ≈ 1.6 GB
# ==============================================================================
def create_depth_distribution_map_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("\n[WARN] CUDA를 찾을 수 없습니다. CPU로 실행합니다 (속도 저하 가능).")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"\n[GPU] {props.name}  ({props.total_memory // 1024**2} MB VRAM)")

    os.makedirs(scene_dis_dir, exist_ok=True)

    # target proc 파일 → cam별 분류
    target_proc_by_cam: dict[str, list[str]] = {cam: [] for cam in CAM_NAMES}
    for f in sorted(os.listdir(target_proc_dir)):
        if not f.endswith(".npy"):
            continue
        cam = f.rsplit("_", 1)[1].replace(".npy", "")
        if cam in target_proc_by_cam:
            target_proc_by_cam[cam].append(f)

    for cam, files in target_proc_by_cam.items():
        print(f"  target processed depth [{cam:6s}]: {len(files)}장")

    # scene proc 파일 → cam별 분류 (미완료만)
    scenes_by_cam: dict[str, list[str]] = {cam: [] for cam in CAM_NAMES}
    for f in sorted(os.listdir(scene_proc_dir)):
        if not f.endswith(".npy"):
            continue
        if os.path.exists(os.path.join(scene_dis_dir, f)):
            continue
        cam = f.rsplit("_", 1)[1].replace(".npy", "")
        if cam in scenes_by_cam:
            scenes_by_cam[cam].append(f)

    total_remaining = sum(len(v) for v in scenes_by_cam.values())
    print(f"\n[depth_dis_map GPU] 처리할 scene: {total_remaining}장\n")

    for cam in CAM_NAMES:
        scene_files  = scenes_by_cam[cam]
        target_files = target_proc_by_cam[cam]

        if not scene_files:
            print(f"[{cam}] 처리할 scene 없음, 건너뜀")
            continue
        if not target_files:
            print(f"[{cam}] target 파일 없음, 건너뜀")
            continue

        print(f"[{cam}] scene={len(scene_files)}장, target={len(target_files)}장 처리 시작")

        # ── empty scene depth 로드 (카메라당 1회) ─────────────────────────────
        _empty_path = os.path.join(empty_scene_depth_dir, f"{cam}.npy")
        if os.path.exists(_empty_path):
            _ed = np.nan_to_num(np.load(_empty_path).squeeze(), nan=0.0).astype(np.float32)
            empty_depth_gpu = torch.from_numpy(_ed).to(device)  # [H, W]
            print(f"  [{cam}] empty_scene depth 로드 완료")
        else:
            empty_depth_gpu = None
            print(f"  [{cam}] [WARN] empty_scene depth 없음 → empty_scene.py 먼저 실행하세요.")
            print(f"           경로: {_empty_path}")

        # ── scene 배치 단위로 처리 ─────────────────────────────────────────────
        for sc_start in range(0, len(scene_files), SCENE_GPU_BATCH):
            sc_batch = scene_files[sc_start : sc_start + SCENE_GPU_BATCH]
            SC = len(sc_batch)

            # scene 배치를 numpy로 읽어 GPU에 올림
            scenes_list: list[np.ndarray] = []
            H = W = None
            for sf in sc_batch:
                # scene/depth/ (전체 씬 depth, masking 없음) 로드
                s = np.load(os.path.join(scene_depth_dir, sf)).squeeze().astype(np.float32)
                s = np.nan_to_num(s, nan=0.0)
                if H is None:
                    H, W = s.shape
                scenes_list.append(s)

            scenes_gpu = torch.from_numpy(np.stack(scenes_list)).to(device)  # [SC, H, W]
            del scenes_list

            # 누적 버퍼 (float64: 정밀도 유지)
            sum_gpu   = torch.zeros(SC, H, W, dtype=torch.float64, device=device)
            count_gpu = torch.zeros(SC,       dtype=torch.float64, device=device)

            # ── target을 TARGET_GPU_BATCH 단위로 읽어 처리 ────────────────────
            num_tb = (len(target_files) + TARGET_GPU_BATCH - 1) // TARGET_GPU_BATCH

            for tb_idx, tc_start in enumerate(range(0, len(target_files), TARGET_GPU_BATCH)):
                tc_batch = target_files[tc_start : tc_start + TARGET_GPU_BATCH]

                # disk I/O: TARGET_GPU_BATCH개 target 로드
                valid_targets: list[np.ndarray] = []
                for tf in tc_batch:
                    t = np.load(os.path.join(target_proc_dir, tf)).squeeze().astype(np.float32)
                    if t.shape == (H, W):
                        valid_targets.append(t)

                if not valid_targets:
                    continue

                targets_gpu = torch.from_numpy(
                    np.stack(valid_targets)
                ).to(device)  # [TC, H, W]

                # 각 target k → SC scene 전체와 동시에 occlusion 계산
                for k in range(targets_gpu.shape[0]):
                    target_k = targets_gpu[k]             # [H, W]

                    object_mask  = (target_k != 0)         # [H, W]
                    total_pixels = int(object_mask.sum().item())
                    if total_pixels == 0:
                        continue

                    # object 픽셀만 추출: [SC, Npix]
                    scene_obj  = scenes_gpu[:, object_mask]   # [SC, Npix] - 전체 씬 depth
                    target_obj = target_k[object_mask]         # [Npix]

                    # ── 유효 위치 필터: empty 씬에서 shelf에 가려지지 않는 픽셀 ──
                    # empty_depth == 0 → 열린 공간 (유효)
                    # empty_depth >= target_depth → shelf이 scan 위치보다 멀리 있음 (유효)
                    # empty_depth < target_depth → shelf이 scan 위치 앞에 있음 (항상 가려짐 → 제외)
                    if empty_depth_gpu is not None:
                        empty_obj  = empty_depth_gpu[object_mask]                # [Npix]
                        valid_pos  = (empty_obj == 0) | (empty_obj >= target_obj)  # [Npix]
                    else:
                        valid_pos  = torch.ones(target_obj.shape, dtype=torch.bool, device=device)

                    # ── 은닉: 씬에서 scan 위치보다 가까운 물체가 있음 ─────────
                    # scene_full < target AND scene_full != 0 AND valid_pos
                    occ = (
                        (scene_obj < target_obj.unsqueeze(0)) &
                        (scene_obj != 0) &
                        valid_pos.unsqueeze(0)
                    )  # [SC, Npix]

                    ratio = occ.sum(dim=1).float() / total_pixels              # [SC]
                    above = (ratio >= OCCLUSION_THRESHOLD)                    # [SC] bool

                    if not above.any():
                        continue

                    # ── 누적: above인 scene에만 target_k 합산 ─────────────────
                    # above_f: [SC, 1, 1]  ×  target_k: [H, W]  →  [SC, H, W]
                    above_f = above.double().unsqueeze(1).unsqueeze(2)
                    sum_gpu   += above_f * target_k.double()
                    count_gpu += above.double()

                del targets_gpu

                # 진행률 출력 (10% 단위)
                if (tb_idx + 1) % max(1, num_tb // 10) == 0 or (tb_idx + 1) == num_tb:
                    sc_done = min(sc_start + SCENE_GPU_BATCH, len(scene_files))
                    print(f"  [{cam}] scene {sc_start+1}~{sc_done}/{len(scene_files)} | "
                          f"target batch {tb_idx+1}/{num_tb}")

            del scenes_gpu

            # ── 결과 저장 ─────────────────────────────────────────────────────
            for i, scene_fname in enumerate(sc_batch):
                out_path = os.path.join(scene_dis_dir, scene_fname)
                cnt = int(count_gpu[i].item())

                if cnt == 0:
                    print(f"  [WARN] {scene_fname}: occluded target 없음, 빈 map 저장")
                    np.save(out_path, np.zeros((H, W), dtype=np.uint8))
                    continue

                avg = (sum_gpu[i] / cnt).cpu().numpy().astype(np.float32)
                min_val, max_val = float(avg.min()), float(avg.max())
                if max_val > min_val:
                    normalized = 255.0 * (avg - min_val) / (max_val - min_val)
                else:
                    normalized = np.zeros_like(avg)

                result = np.clip(normalized, 0, 255).astype(np.uint8)
                np.save(out_path, result)

            del sum_gpu, count_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

        print(f"  [{cam}] 완료!")


# ==============================================================================
# Step 4: process_all_maps
#   depth_dis_map + similarity_map 결합, visibility check 적용
# ==============================================================================
def process_all_maps(ref_pixels: dict):
    os.makedirs(dist_map_dir, exist_ok=True)

    dis_files = sorted(f for f in os.listdir(scene_dis_dir) if f.endswith(".npy"))
    sim_files = sorted(f for f in os.listdir(sim_map_dir)   if f.endswith(".png"))

    dis_lookup = {os.path.splitext(f)[0]: f for f in dis_files}
    sim_lookup = {os.path.splitext(f)[0]: f for f in sim_files}

    common = sorted(set(dis_lookup) & set(sim_lookup))
    print(f"\n[distribution_map] {len(common)}장 처리 중...")

    for base in common:
        cam = base.rsplit("_", 1)[1]

        dis = np.load(os.path.join(scene_dis_dir, dis_lookup[base])).astype(np.float32)
        sim = cv2.imread(os.path.join(sim_map_dir, sim_lookup[base]),
                         cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if dis.shape != sim.shape:
            print(f"  [WARN] shape 불일치: {base}")
            continue

        combined = np.clip(BETA * dis + (1.0 - BETA) * sim, 0, 255).astype(np.uint8)

        seg_path = os.path.join(scene_seg_dir, base + ".png")
        if os.path.exists(seg_path):
            seg_img = cv2.imread(seg_path)
            if seg_img is not None:
                target_mask    = get_target_mask(seg_img, target_bgr_in_scene)
                visible_pixels = int(np.sum(target_mask))
                ref            = ref_pixels.get(cam, 1)
                vis_ratio      = visible_pixels / ref if ref > 0 else 0.0

                if vis_ratio >= VISIBILITY_THRESHOLD and visible_pixels > 0:
                    combined = np.clip(combined.astype(np.float32) * DIM_FACTOR,
                                       0, 255).astype(np.uint8)
                    combined[target_mask] = 255

        out_path = os.path.join(dist_map_dir, base + ".png")
        cv2.imwrite(out_path, combined)

    print(f"  → {len(common)}장 저장: {dist_map_dir}")


# ==============================================================================
# Main
# ==============================================================================
def _load_params_cache() -> dict:
    path = os.path.join(OUTPUT_DIR, "_params_cache.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_params_cache():
    path = os.path.join(OUTPUT_DIR, "_params_cache.json")
    with open(path, "w") as f:
        json.dump({
            "OCCLUSION_THRESHOLD":  OCCLUSION_THRESHOLD,
            "BETA":                 BETA,
            "VISIBILITY_THRESHOLD": VISIBILITY_THRESHOLD,
            "DIM_FACTOR":           DIM_FACTOR,
        }, f, indent=2)


if __name__ == "__main__":
    # 파라미터 변경 감지
    prev = _load_params_cache()
    if prev.get("OCCLUSION_THRESHOLD") != OCCLUSION_THRESHOLD:
        if prev:
            print(f"\n[파라미터 변경 감지] OCCLUSION_THRESHOLD: "
                  f"{prev.get('OCCLUSION_THRESHOLD')} → {OCCLUSION_THRESHOLD}")
            print(f"  scene/depth_dis_map/ 초기화 중...")
            if os.path.exists(scene_dis_dir):
                for f in os.listdir(scene_dis_dir):
                    if f.endswith(".npy"):
                        os.remove(os.path.join(scene_dis_dir, f))
                print(f"  → 초기화 완료")

    # Step 1: target / scene masked depth 생성
    combine_all_images("target", target_bgr_in_target)
    combine_all_images("scene",  target_bgr_in_scene)

    # Step 2: visibility check용 레퍼런스 픽셀 수 계산
    ref_pixels = compute_ref_pixels()

    # Step 3: occlusion map 생성 (GPU 가속)
    create_depth_distribution_map_gpu()

    # Step 4: distribution map 생성
    if os.path.exists(sim_map_dir) and any(
        f.endswith(".png") for f in os.listdir(sim_map_dir)
    ):
        process_all_maps(ref_pixels)
    else:
        print(f"\n[WARN] similarity_map 폴더가 비어있어 process_all_maps 건너뜀")
        print(f"  먼저 similarity_map_generator.py를 실행하세요.")

    _save_params_cache()

    print(f"\n=== 완료 ===")
    print(f"  저장 위치: {dist_map_dir}")
