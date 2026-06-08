"""
search_test.py
FCN-guided occlusion search evaluation on pre-generated test scenes.

For each step:
  1. Count target visible pixels via segmentation → compute visibility ratio
  2. If ratio >= visibility_threshold → target graspable, remove it, SUCCESS
  3. Else → FCN inference → EMA accumulate → find highest-activation cluster
             (excluding target pixels and already-removed objects)
          → identify object at peak pixel via segmentation → remove it
  4. Repeat until success or max_steps exceeded (FAIL)

Usage:
    python search_test.py --target book_1 --scene_num 0 [--camera_num 0] [--headless]
"""

# ═══════════════════════════════════════════════════════════════════════════
# 파라미터 설정 (이 블록에서 모두 수정 가능)
# ═══════════════════════════════════════════════════════════════════════════

# --- FCN 추론 ---
VISIBILITY_THRESHOLD = 0.7    # 타겟 픽셀 비율이 이 값 이상이면 graspable로 판단
MAX_STEPS            = 11     # 최대 제거 스텝 수 (초과 시 FAIL)
ALPHA                = 0.7    # EMA 가중치 (현재 FCN 맵 반영 비율)
CLUSTER_KERNEL       = 31     # Gaussian blur 커널 크기 (홀수, 클수록 넓게 탐색)

# --- 모델 경로 (None이면 src/outputs/ 기본 경로 사용) ---
MODEL_PATH_OVERRIDE  = None   # 예: "/path/to/best_model.pth"
STATS_PATH_OVERRIDE  = None   # 예: "/path/to/dataset_stats.txt"

# ═══════════════════════════════════════════════════════════════════════════
# 0. Argparse  (must be BEFORE SimulationApp)
# ═══════════════════════════════════════════════════════════════════════════
import argparse

CAMERA_ORDER = ["center", "left", "right", "top", "bottom"]

parser = argparse.ArgumentParser(description="FCN-guided occlusion search test")
parser.add_argument("--target",     type=str, required=True)
parser.add_argument("--scene_num",  type=int, required=True,
                    help="Scene index (matches scene_XXXX folder in test_scenes/)")
parser.add_argument("--camera_num", type=int, default=0,
                    help=f"Camera index: {list(enumerate(CAMERA_ORDER))}")
parser.add_argument("--headless",   action="store_true")
args, unknown = parser.parse_known_args()

# ═══════════════════════════════════════════════════════════════════════════
# 1. SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# ═══════════════════════════════════════════════════════════════════════════
# 2. Imports
# ═══════════════════════════════════════════════════════════════════════════
import os
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
from scipy.spatial.transform import Rotation as ScipyR
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms import Normalize

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
from pxr import UsdLux, UsdPhysics, Usd
from semantics.schema.editor import PrimSemanticData

from object_spawner import ObjectSpawner

# ═══════════════════════════════════════════════════════════════════════════
# 3. Constants & paths
# ═══════════════════════════════════════════════════════════════════════════
CLASS_NAMES = [
    "book_1", "book_2", "book_3", "book_4",
    "fruit_1", "fruit_2", "fruit_3", "fruit_4",
    "packaged_food_1", "packaged_food_2", "packaged_food_3", "packaged_food_4",
    "toy_1", "toy_2", "toy_3", "toy_4",
]
NUM_CLASSES = len(CLASS_NAMES)

if args.target not in CLASS_NAMES:
    print(f"[ERROR] '{args.target}' not in CLASS_NAMES.")
    print(f"  Valid targets: {CLASS_NAMES}")
    simulation_app.close()
    exit(1)

TARGET_CLASS_IDX = CLASS_NAMES.index(args.target)
CAMERA_NAME      = CAMERA_ORDER[args.camera_num]

SRC_DIR            = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR          = os.path.join(SRC_DIR, "asset")
USD_FILE_DIR       = os.path.join(ASSET_DIR, "260303")
WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "USD", "drawer.usd")

SCENE_DIR = os.path.join(
    SRC_DIR, "output", args.target, "test_scenes",
    f"scene_{args.scene_num:04d}"
)
REF_JSON  = os.path.join(
    SRC_DIR, "output", args.target, "test_scenes", "ref_pixels.json"
)
OUT_DIR   = os.path.join(
    SRC_DIR, "output", args.target, "test_results",
    f"scene_{args.scene_num:04d}", f"cam_{CAMERA_NAME}"
)
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH = MODEL_PATH_OVERRIDE or os.path.join(SRC_DIR, "output", "best_model.pth")
STATS_PATH = STATS_PATH_OVERRIDE or os.path.join(SRC_DIR, "output", "dataset_stats.txt")

# Validate inputs
for path, name in [(SCENE_DIR, "SCENE_DIR"), (MODEL_PATH, "MODEL_PATH"),
                   (STATS_PATH, "STATS_PATH"), (REF_JSON, "REF_JSON")]:
    if not os.path.exists(path):
        print(f"[ERROR] {name} not found: {path}")
        simulation_app.close()
        exit(1)

# Load poses
with open(os.path.join(SCENE_DIR, "poses.json")) as f:
    poses_data = json.load(f)

# Load ref pixels
with open(REF_JSON) as f:
    ref_pixel_data = json.load(f)
REF_PIXEL_COUNT = ref_pixel_data["cameras"][CAMERA_NAME]

if REF_PIXEL_COUNT == 0:
    print(f"[ERROR] ref_pixels for camera '{CAMERA_NAME}' is 0. Re-run generate_test_scenes.py.")
    simulation_app.close()
    exit(1)

print(f"\n[Config]")
print(f"  target           : {args.target} (class_idx={TARGET_CLASS_IDX})")
print(f"  scene_num        : {args.scene_num}")
print(f"  camera           : {CAMERA_NAME} (ref_pixels={REF_PIXEL_COUNT})")
print(f"  visibility_thr   : {VISIBILITY_THRESHOLD}")
print(f"  max_steps        : {MAX_STEPS}")
print(f"  alpha (EMA)      : {ALPHA}")
print(f"  cluster_kernel   : {CLUSTER_KERNEL}")
print(f"  output           : {OUT_DIR}\n")

# ═══════════════════════════════════════════════════════════════════════════
# 4. Camera configuration
# ═══════════════════════════════════════════════════════════════════════════
CAMERA_HEIGHT_OFFSET = 3.0
CAMERA_XY_OFFSET     = 1.0
CAMERA_RESOLUTION    = (640, 480)

CAMERA_CONFIGS = {
    "center": (0.0,               0.0),
    "left":   (-CAMERA_XY_OFFSET, 0.0),
    "right":  ( CAMERA_XY_OFFSET, 0.0),
    "top":    (0.0,               CAMERA_XY_OFFSET),
    "bottom": (0.0,              -CAMERA_XY_OFFSET),
}

WORKSPACE_BOUNDS = {
    "x":         (-0.15, 0.15),
    "y":         (-0.15, 0.15),
    "z_surface": 0.01,
    "z_drop":    0.20,
}

STABILIZATION_STEPS = 60   # steps when reconstructing scene from poses

# ═══════════════════════════════════════════════════════════════════════════
# 5. FCN model
# ═══════════════════════════════════════════════════════════════════════════
def load_fcn(model_path: str, stats_path: str):
    """Load trained FCN and normalization stats."""
    class FCNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
            self.model.classifier[4] = nn.Conv2d(512, NUM_CLASSES, kernel_size=1)
        def forward(self, x):
            return self.model(x)["out"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = FCNModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(stats_path) as f:
        lines = f.readlines()
    mean = np.array(eval(lines[0].split(": ")[1].strip()))
    std  = np.array(eval(lines[1].split(": ")[1].strip()))
    transform = Normalize(mean=mean.tolist(), std=std.tolist())

    return model, transform, device

print("[FCN] Loading model...")
fcn_model, fcn_transform, fcn_device = load_fcn(MODEL_PATH, STATS_PATH)
print("[FCN] Loaded.\n")

# ═══════════════════════════════════════════════════════════════════════════
# 6. World
# ═══════════════════════════════════════════════════════════════════════════
world = World(physics_dt=1 / 120.0, backend="torch", device="cuda")
pc    = world.get_physics_context()
pc.set_solver_type("TGS")
pc.enable_ccd(True)

GroundPlane(
    prim_path="/World/GroundPlane",
    z_position=0,
    color=torch.tensor([1.0, 1.0, 1.0]),
)

stage = omni.usd.get_context().get_stage()

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1000)
dlt  = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
dlt.CreateIntensityAttr(1000)
dlt.CreateAngleAttr(0.53)

# ═══════════════════════════════════════════════════════════════════════════
# 7. Workspace (static collider)
# ═══════════════════════════════════════════════════════════════════════════
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace")

def make_static_collider(path: str):
    root = stage.GetPrimAtPath(path)
    if not root.IsValid():
        return
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

make_static_collider("/World/workspace")

# ═══════════════════════════════════════════════════════════════════════════
# 8. ObjectSpawner
# ═══════════════════════════════════════════════════════════════════════════
object_spawner = ObjectSpawner(
    world=world,
    categories=["Book", "Toy", "Fruit", "Packaged_food"],
    usd_folder_dir=USD_FILE_DIR,
    container_prim_path="/World/Objects_0",
    workspace_bounds=WORKSPACE_BOUNDS,
    default_position=torch.tensor([0.0, 0.7, 0.05]),
    num_to_spawn=None,
    extensions=(".usd",),
)

for prim_path, obj_name in zip(object_spawner._spawned_paths,
                                object_spawner._spawned_names):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        sem = PrimSemanticData(prim)
        sem.add_entry("class", obj_name)

object_spawner.setup_cloned_views(num_envs=1)

# USD 에셋 이름(예: "Book_02") → 폴더/클래스 이름(예: "book_1") 역매핑
# (로그에는 실제 USD 파일명 대신 CLASS_NAMES와 동일한 폴더 이름으로 기록하기 위함)
USD_TO_CLASS_NAME = {usd_name: folder_name
                     for folder_name, usd_name in object_spawner._folder_to_usd.items()}

target_idx = object_spawner.get_object_index_by_name(args.target)
print(f"[Spawner] target_idx={target_idx}, name='{object_spawner._spawned_names[target_idx]}'\n")

# Prim path of target object (for segmentation identification)
TARGET_PRIM_PATH = object_spawner._spawned_paths[target_idx]
# e.g. "/World/Objects_0/object_3" → "object_3"
TARGET_PRIM_KEY  = TARGET_PRIM_PATH.split("/")[-1]

# ═══════════════════════════════════════════════════════════════════════════
# 9. Camera + segmentation annotator (only the selected camera)
# ═══════════════════════════════════════════════════════════════════════════
def look_at_quat(cam_pos, target=(0.0, 0.0, 0.0)):
    d   = np.array(target) - np.array(cam_pos)
    d  /= np.linalg.norm(d)
    rot, _ = ScipyR.align_vectors([d], [[0.0, 0.0, -1.0]])
    q   = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

ox, oy  = CAMERA_CONFIGS[CAMERA_NAME]
cam_pos = np.array([ox, oy, CAMERA_HEIGHT_OFFSET])
cam_q   = look_at_quat(cam_pos)

active_cam = Camera(
    prim_path=f"/World/camera_{CAMERA_NAME}",
    position=cam_pos,
    resolution=CAMERA_RESOLUTION,
)
active_cam.initialize()

rp      = rep.create.render_product(active_cam.prim_path, CAMERA_RESOLUTION)
seg_ann = rep.AnnotatorRegistry.get_annotator(
    "instance_segmentation", init_params={"colorize": False}
)
seg_ann.attach([rp])

# ═══════════════════════════════════════════════════════════════════════════
# 10. Start simulation
# ═══════════════════════════════════════════════════════════════════════════
world.play()
object_spawner.initialize()
for _ in range(30):
    world.step(render=True)

# Fix camera pose
cam_view = XFormPrimView(f"/World/camera_{CAMERA_NAME}")
cam_view.set_world_poses(
    torch.tensor(cam_pos, dtype=torch.float32, device="cuda").unsqueeze(0),
    torch.tensor(cam_q,   dtype=torch.float32, device="cuda").unsqueeze(0),
)
for _ in range(10):
    world.step(render=True)

# ═══════════════════════════════════════════════════════════════════════════
# 11. Reconstruct scene from saved poses
# ═══════════════════════════════════════════════════════════════════════════
print("[Scene] Reconstructing from saved poses...")

for obj_data in poses_data["objects"]:
    i       = obj_data["obj_index"]
    pos_np  = np.array(obj_data["position"],    dtype=np.float32)   # [x,y,z]
    ori_np  = np.array(obj_data["orientation"], dtype=np.float32)   # [w,x,y,z]
    pos_t   = torch.tensor(pos_np, device="cuda").unsqueeze(0)
    ori_t   = torch.tensor(ori_np, device="cuda").unsqueeze(0)
    object_spawner._item_views[i].set_world_poses(pos_t, ori_t)

# Let physics settle to stable state
for _ in range(STABILIZATION_STEPS):
    world.step(render=True)

print("[Scene] Reconstruction complete.\n")

# ═══════════════════════════════════════════════════════════════════════════
# 12. Utility functions
# ═══════════════════════════════════════════════════════════════════════════
def get_rgb_tensor() -> torch.Tensor:
    """Capture RGB and return normalized tensor (1,3,H,W) for FCN."""
    rgb = active_cam.get_rgb()              # (H,W,3) uint8
    img = rgb.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)            # (3,H,W)
    t   = torch.tensor(img, dtype=torch.float32)
    t   = fcn_transform(t)
    return t.unsqueeze(0).to(fcn_device)    # (1,3,H,W)

def fcn_infer() -> np.ndarray:
    """Run FCN and return target-channel output as float32 (H,W) in [0,1]."""
    with torch.no_grad():
        out = fcn_model(get_rgb_tensor())   # (1,16,H,W)
    pred = out[0, TARGET_CLASS_IDX].cpu().numpy()  # (H,W)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)

def get_seg_data():
    """Return (seg_ids [H,W], id_to_labels dict) from current segmentation."""
    data = seg_ann.get_data()
    if data is None or "data" not in data:
        return None, {}
    ids = data["data"]
    if ids.ndim == 3:
        ids = ids[:, :, 0]
    id_to_labels = data.get("info", {}).get("idToLabels", {})
    return ids, id_to_labels

def count_target_pixels(seg_ids: np.ndarray, id_to_labels: dict) -> int:
    """Count pixels belonging to the target object."""
    count = 0
    for uid in np.unique(seg_ids):
        label = id_to_labels.get(str(int(uid)), "")
        if TARGET_PRIM_KEY in label:
            count += int(np.sum(seg_ids == uid))
    return count

def get_object_index_at_pixel(row: int, col: int,
                               seg_ids: np.ndarray,
                               id_to_labels: dict) -> int | None:
    """Return spawner object index for the object at pixel (row, col).
    Returns None if background, target itself, or unrecognised."""
    uid   = int(seg_ids[row, col])
    label = id_to_labels.get(str(uid), "")
    if not label or label in ("BACKGROUND", "UNLABELLED"):
        return None
    # label looks like "/World/Objects_0/object_N" or similar
    for part in label.split("/"):
        if part.startswith("object_"):
            try:
                idx = int(part.split("_")[1])
                if idx == target_idx:
                    return None   # is the target → skip
                return idx
            except (IndexError, ValueError):
                pass
    return None

def remove_object(obj_idx: int):
    """Teleport object to its wait position (outside workspace)."""
    wait_pos = object_spawner._wait_positions[obj_idx].unsqueeze(0)
    # Convert local wait pos to world pos
    cont_pos, _ = object_spawner._container_view.get_world_poses()
    world_pos    = cont_pos + wait_pos
    object_spawner._item_views[obj_idx].set_world_poses(world_pos)

def remove_target():
    """Teleport target object to its wait position."""
    remove_object(target_idx)

def find_cluster_peak(acc_map: np.ndarray,
                      seg_ids: np.ndarray,
                      id_to_labels: dict,
                      removed_set: set,
                      kernel: int = 31) -> tuple[int, int] | None:
    """
    Blur accumulated map → mask out target pixels & removed-object pixels
    → find argmax → verify the object there is valid.
    Returns (row, col) of the best valid peak, or None.
    """
    k      = kernel if kernel % 2 == 1 else kernel + 1
    sigma  = k / 6.0
    blurred = cv2.GaussianBlur(acc_map, (k, k), sigma)

    # Build mask of pixels to suppress (target + already removed objects)
    suppress = np.zeros_like(blurred, dtype=bool)
    for uid in np.unique(seg_ids):
        label = id_to_labels.get(str(int(uid)), "")
        if not label or label in ("BACKGROUND", "UNLABELLED"):
            continue
        for part in label.split("/"):
            if part.startswith("object_"):
                try:
                    idx = int(part.split("_")[1])
                    if idx == target_idx or idx in removed_set:
                        suppress[seg_ids == uid] = True
                except (IndexError, ValueError):
                    pass

    search_map = blurred.copy()
    search_map[suppress] = 0.0

    # Iteratively find peak; if maps to invalid object, suppress region & retry
    max_tries = 20
    for _ in range(max_tries):
        if search_map.max() == 0.0:
            return None
        flat_idx = int(np.argmax(search_map))
        row, col = divmod(flat_idx, search_map.shape[1])

        obj_idx = get_object_index_at_pixel(row, col, seg_ids, id_to_labels)
        if obj_idx is not None and obj_idx not in removed_set:
            return row, col

        # Suppress a region around this peak and try again
        r_min = max(0, row - k // 2)
        r_max = min(search_map.shape[0], row + k // 2 + 1)
        c_min = max(0, col - k // 2)
        c_max = min(search_map.shape[1], col + k // 2 + 1)
        search_map[r_min:r_max, c_min:c_max] = 0.0

    return None

def save_step_rgb(step: int, label: str = ""):
    """Capture and save current RGB for the given step."""
    rgb = active_cam.get_rgb()
    if rgb is None:
        return
    suffix = f"_{label}" if label else ""
    path   = os.path.join(OUT_DIR, f"step_{step:04d}{suffix}.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# ═══════════════════════════════════════════════════════════════════════════
# 13. Main search loop
# ═══════════════════════════════════════════════════════════════════════════
log = {
    "target":               args.target,
    "scene_num":            args.scene_num,
    "camera":               CAMERA_NAME,
    "visibility_threshold": VISIBILITY_THRESHOLD,
    "max_steps":            MAX_STEPS,
    "alpha":                ALPHA,
    "cluster_kernel":       CLUSTER_KERNEL,
    "ref_pixel_count":      REF_PIXEL_COUNT,
    "success":              False,
    "steps_taken":          0,
    "removed_objects":      [],
}

removed_set   = set()    # obj indices already removed
acc_map       = None     # EMA-accumulated FCN output (H,W) float32

try:
    print("=== Search started ===\n")

    for step in range(MAX_STEPS + 1):  # +1: final visibility check after last removal
        # ── Render & grab segmentation ──────────────────────────────────
        for _ in range(5):
            world.step(render=True)

        seg_ids, id_to_labels = get_seg_data()

        # ── Visibility check ────────────────────────────────────────────
        if seg_ids is not None:
            vis_pixels = count_target_pixels(seg_ids, id_to_labels)
            vis_ratio  = vis_pixels / REF_PIXEL_COUNT
        else:
            vis_pixels, vis_ratio = 0, 0.0

        print(f"[step {step:03d}] vis={vis_ratio:.3f} ({vis_pixels}/{REF_PIXEL_COUNT} px)", end="")

        if vis_ratio >= VISIBILITY_THRESHOLD:
            print(f"  → TARGET GRASPABLE (>= {VISIBILITY_THRESHOLD})")
            save_step_rgb(step, "success")
            remove_target()
            for _ in range(10):
                world.step(render=True)
            save_step_rgb(step, "target_removed")
            log["success"]     = True
            log["steps_taken"] = step
            break

        if step == MAX_STEPS:
            print(f"  → FAILED (max_steps reached)")
            save_step_rgb(step, "fail")
            log["steps_taken"] = step
            break

        print()  # newline before removal info

        # ── FCN inference ───────────────────────────────────────────────
        current_map = fcn_infer()            # (H,W) float32

        # ── EMA accumulation ────────────────────────────────────────────
        if acc_map is None:
            acc_map = current_map.copy()
        else:
            acc_map = ALPHA * current_map + (1.0 - ALPHA) * acc_map

        # ── Save step RGB ────────────────────────────────────────────────
        save_step_rgb(step)

        # Also save accumulated map as grayscale for inspection
        acc_vis = (acc_map * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f"step_{step:04d}_fcnmap.png"), acc_vis)

        # ── Find object to remove ────────────────────────────────────────
        if seg_ids is None:
            print(f"  [WARN] No segmentation data at step {step}, skipping removal.")
            continue

        peak = find_cluster_peak(
            acc_map, seg_ids, id_to_labels,
            removed_set, kernel=CLUSTER_KERNEL
        )

        if peak is None:
            print(f"  [WARN] No valid removal candidate found at step {step}.")
            continue

        row, col   = peak
        obj_idx    = get_object_index_at_pixel(row, col, seg_ids, id_to_labels)
        if obj_idx is None:
            print(f"  [WARN] Peak pixel ({row},{col}) maps to no valid object.")
            continue

        obj_name   = object_spawner._spawned_names[obj_idx]
        class_name = USD_TO_CLASS_NAME.get(obj_name, obj_name)
        print(f"  → Removing obj_idx={obj_idx} '{class_name}' (USD='{obj_name}', peak=({row},{col}))")

        remove_object(obj_idx)
        removed_set.add(obj_idx)
        log["removed_objects"].append({
            "step":     step,
            "obj_idx":  obj_idx,
            "obj_name": class_name,
            "peak_row": int(row),
            "peak_col": int(col),
        })

        # Let physics settle after removal
        for _ in range(10):
            world.step(render=True)

finally:
    # ── Save log ────────────────────────────────────────────────────────
    log_path = os.path.join(OUT_DIR, "result.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    status = "SUCCESS" if log["success"] else "FAIL"
    print(f"\n=== {status} | steps={log['steps_taken']} | removed={len(log['removed_objects'])} ===")
    print(f"  Log: {log_path}")
    print(f"  Images: {OUT_DIR}\n")

    world.stop()
    simulation_app.close()
