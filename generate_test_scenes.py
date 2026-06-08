"""
generate_test_scenes.py
Test scene generator for occlusion-based object search evaluation.

Saves per-scene RGB images and all object poses for reproducible replay.
Also measures the reference pixel count for the target object (once, cached).

Output structure:
    output/{target}/test_scenes/
        ref_pixels.json               ← target max-visibility pixel counts per camera
        scene_0000/
            rgb_center.png
            rgb_left.png
            rgb_right.png
            rgb_top.png
            rgb_bottom.png
            poses.json                ← all object world poses for exact reproduction

Usage:
    python generate_test_scenes.py --target book_1 --scene_num 10 [--headless]
"""

# ═══════════════════════════════════════════════════════════════════════════
# 0. Argparse  (must be BEFORE SimulationApp)
# ═══════════════════════════════════════════════════════════════════════════
import argparse

parser = argparse.ArgumentParser(description="Test Scene Generator for Occlusion Search")
parser.add_argument("--target",    type=str, required=True,
                    help="Target object folder name (e.g. book_1)")
parser.add_argument("--scene_num", type=int, default=10,
                    help="Number of test scenes to generate (default: 10)")
parser.add_argument("--headless",  action="store_true",
                    help="Run without GUI")
args, unknown = parser.parse_known_args()

# ═══════════════════════════════════════════════════════════════════════════
# 1. SimulationApp
# ═══════════════════════════════════════════════════════════════════════════
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# ═══════════════════════════════════════════════════════════════════════════
# 2. Imports  (must come after SimulationApp)
# ═══════════════════════════════════════════════════════════════════════════
import os
import json
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as ScipyR

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
from pxr import UsdLux, UsdPhysics, UsdGeom, Usd, Gf
from semantics.schema.editor import PrimSemanticData

from object_spawner import ObjectSpawner

# ═══════════════════════════════════════════════════════════════════════════
# 3. Tunable Parameters
# ═══════════════════════════════════════════════════════════════════════════
# --- Scene generation physics ---
STABILIZATION_STEPS       = 60     # physics steps after each object drop
FINAL_STABILIZATION_STEPS = 120    # final stabilization after all drops
TARGET_NOT_BOTTOM_PROB    = 0.15   # prob that target is NOT the first spawned
TARGET_TOP_PROB           = 0.60   # given not-first, prob target is last

# --- Reference pixel measurement ---
REF_Z            = 0.01           # z height for reference placement (floor level)
REF_RENDER_STEPS = 10             # render steps for stabilization

# --- Camera configuration ---
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

# --- Workspace bounds ---
WORKSPACE_BOUNDS = {
    "x":         (-0.15, 0.15),
    "y":         (-0.15, 0.15),
    "z_surface": 0.01,
    "z_drop":    0.20,
}

# ═══════════════════════════════════════════════════════════════════════════
# 4. Asset paths
# ═══════════════════════════════════════════════════════════════════════════
SRC_DIR            = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR          = os.path.join(SRC_DIR, "asset")
USD_FILE_DIR       = os.path.join(ASSET_DIR, "260303")
WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "USD", "drawer.usd")

OUT_DIR = os.path.join(SRC_DIR, "output", args.target, "test_scenes")
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Target USD discovery
# ═══════════════════════════════════════════════════════════════════════════
def discover_assets(folder_dir, extensions=(".usd", ".usdc")):
    """Returns {folder_name_lower: (usd_name, usd_path, category)}"""
    assets = {}
    for cat in sorted(os.listdir(folder_dir)):
        cat_dir = os.path.join(folder_dir, cat)
        if not os.path.isdir(cat_dir):
            continue
        for sub in sorted(os.listdir(cat_dir)):
            sub_path = os.path.join(cat_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            for f in sorted(os.listdir(sub_path)):
                if f.lower().endswith(extensions):
                    assets[sub.lower()] = (
                        os.path.splitext(f)[0],
                        os.path.join(sub_path, f),
                        cat,
                    )
                    break
    return assets

all_assets = discover_assets(USD_FILE_DIR)
target_key = args.target.lower()

if target_key not in all_assets:
    print(f"[ERROR] '{args.target}' not found.")
    print(f"  Available: {sorted(all_assets.keys())}")
    simulation_app.close()
    exit(1)

target_usd_name, target_usd_path, target_category = all_assets[target_key]
print(f"\n[Target] {args.target} → {target_usd_name} ({target_category})")
print(f"[Output] {OUT_DIR}\n")

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
# 8. Static target prim for reference pixel measurement
#    Uses /World/target_ref (separate from the RigidPrim in ObjectSpawner)
# ═══════════════════════════════════════════════════════════════════════════
add_reference_to_stage(usd_path=target_usd_path, prim_path="/World/target_ref")
make_static_collider("/World/target_ref")

_ref_prim      = stage.GetPrimAtPath("/World/target_ref")

# Semantic label 필수: 없으면 instance_segmentation annotator가 prim을 UNLABELLED로 처리
_ref_sem = PrimSemanticData(_ref_prim)
_ref_sem.add_entry("class", target_usd_name)
_ref_xform     = UsdGeom.Xformable(_ref_prim)
_ref_xform.ClearXformOpOrder()
_ref_translate = _ref_xform.AddTranslateOp()
_ref_orient    = _ref_xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
# Park far away initially
_ref_translate.Set(Gf.Vec3d(1000.0, 1000.0, 1000.0))
_ref_orient.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

def set_ref_pose(x: float, y: float, z: float, yaw_deg: float):
    """Place static target_ref prim at (x, y, z) with given yaw (degrees)."""
    rot = ScipyR.from_euler("z", yaw_deg, degrees=True)
    q   = rot.as_quat()  # x, y, z, w
    _ref_translate.Set(Gf.Vec3d(float(x), float(y), float(z)))
    _ref_orient.Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))

def hide_ref():
    _ref_translate.Set(Gf.Vec3d(1000.0, 1000.0, 1000.0))

# ═══════════════════════════════════════════════════════════════════════════
# 9. Cameras + segmentation annotators
# ═══════════════════════════════════════════════════════════════════════════
def look_at_quat(cam_pos, target=(0.0, 0.0, 0.0)):
    d   = np.array(target) - np.array(cam_pos)
    d  /= np.linalg.norm(d)
    rot, _ = ScipyR.align_vectors([d], [[0.0, 0.0, -1.0]])
    q   = rot.as_quat()                           # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]])    # w, x, y, z

cameras   = {}
seg_anns  = {}
cam_quats = {}

for cam_name, (ox, oy) in CAMERA_CONFIGS.items():
    cam_pos = np.array([ox, oy, CAMERA_HEIGHT_OFFSET])
    quat    = look_at_quat(cam_pos)

    cam = Camera(
        prim_path=f"/World/camera_{cam_name}",
        position=cam_pos,
        resolution=CAMERA_RESOLUTION,
    )
    cam.initialize()

    rp      = rep.create.render_product(cam.prim_path, CAMERA_RESOLUTION)
    seg_ann = rep.AnnotatorRegistry.get_annotator(
        "instance_segmentation", init_params={"colorize": False}
    )
    seg_ann.attach([rp])

    cameras[cam_name]   = cam
    seg_anns[cam_name]  = seg_ann
    cam_quats[cam_name] = quat

# ═══════════════════════════════════════════════════════════════════════════
# 10. ObjectSpawner (single environment — no GridCloner needed)
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

# Add semantic labels so segmentation can identify objects in search_test.py
for prim_path, obj_name in zip(object_spawner._spawned_paths,
                                object_spawner._spawned_names):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        sem = PrimSemanticData(prim)
        sem.add_entry("class", obj_name)

# Single-env views (pattern /World/Objects_*/object_N matches /World/Objects_0/object_N)
object_spawner.setup_cloned_views(num_envs=1)

target_idx           = object_spawner.get_object_index_by_name(args.target)
resolved_target_name = object_spawner._spawned_names[target_idx]
print(f"[Spawner] target_idx={target_idx}, resolved='{resolved_target_name}'\n")

# Build reverse mapping: USD name → folder name (for poses.json)
usd_to_folder = {v: k for k, v in object_spawner._folder_to_usd.items()}

# ═══════════════════════════════════════════════════════════════════════════
# 11. Start simulation
# ═══════════════════════════════════════════════════════════════════════════
world.play()
object_spawner.initialize()
for _ in range(30):
    world.step(render=True)

# Fix camera world poses (must be done after world.play())
for cam_name, (ox, oy) in CAMERA_CONFIGS.items():
    cv   = XFormPrimView(f"/World/camera_{cam_name}")
    pos  = torch.tensor(
        [ox, oy, CAMERA_HEIGHT_OFFSET], dtype=torch.float32, device="cuda"
    ).unsqueeze(0)
    quat = torch.tensor(
        cam_quats[cam_name], dtype=torch.float32, device="cuda"
    ).unsqueeze(0)
    cv.set_world_poses(pos, quat)

for _ in range(20):
    world.step(render=True)

# ═══════════════════════════════════════════════════════════════════════════
# 12. Reference pixel measurement
#     Places target_ref at floor level, scans yaw angles, records max pixel
#     count per camera.  Result cached to ref_pixels.json.
# ═══════════════════════════════════════════════════════════════════════════
REF_JSON = os.path.join(OUT_DIR, "ref_pixels.json")

def _count_target_ref_pixels(seg_ann) -> int:
    """Count pixels belonging to /World/target_ref."""
    data = seg_ann.get_data()
    if data is None or "data" not in data:
        return 0
    ids = data["data"]
    if ids.ndim == 3:
        ids = ids[:, :, 0]
    id_to_labels = data.get("info", {}).get("idToLabels", {})
    count = 0
    for uid in np.unique(ids):
        label = id_to_labels.get(str(int(uid)), "")
        # idToLabels 값은 prim path 또는 semantic class 이름일 수 있음
        if "target_ref" in label or label == target_usd_name:
            count += int(np.sum(ids == uid))
    return count

def measure_ref_pixels() -> dict:
    """Measure visible pixel count for target at floor center, all cameras."""
    if os.path.exists(REF_JSON):
        with open(REF_JSON) as f:
            cached = json.load(f)
        print(f"[ref_pixels] Loaded cache: {REF_JSON}")
        for cam, v in cached["cameras"].items():
            print(f"  {cam:6s}: {v} px")
        return cached

    print("[ref_pixels] Measuring (target at floor center, yaw=0)...")
    set_ref_pose(0.0, 0.0, REF_Z, 0.0)
    for _ in range(REF_RENDER_STEPS):
        world.step(render=True)

    ref = {}
    for cam_name, seg_ann in seg_anns.items():
        ref[cam_name] = _count_target_ref_pixels(seg_ann)

    hide_ref()
    for _ in range(5):
        world.step(render=True)

    result = {
        "target":          args.target,
        "target_usd_name": target_usd_name,
        "cameras":         ref,
    }
    with open(REF_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[ref_pixels] Saved → {REF_JSON}")
    for cam, v in ref.items():
        print(f"  {cam:6s}: {v} px")
    return result

ref_pixel_data = measure_ref_pixels()

# ═══════════════════════════════════════════════════════════════════════════
# 13. Scene generation helpers
# ═══════════════════════════════════════════════════════════════════════════
def capture_rgb(scene_dir: str):
    """Capture RGB from all cameras and save as rgb_{camera_name}.png."""
    for cam_name, cam in cameras.items():
        rgb = cam.get_rgb()
        if rgb is not None:
            out_path = os.path.join(scene_dir, f"rgb_{cam_name}.png")
            cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def save_poses(scene_dir: str, scene_idx: int):
    """Save world pose of every object to poses.json."""
    objects = []
    for i, view in enumerate(object_spawner._item_views):
        pos, orient = view.get_world_poses()   # shapes: (1,3), (1,4) w,x,y,z
        usd_name    = object_spawner._spawned_names[i]
        folder_name = usd_to_folder.get(usd_name, usd_name.lower())
        objects.append({
            "obj_index":   i,
            "usd_name":    usd_name,
            "folder_name": folder_name,
            "position":    pos[0].cpu().numpy().tolist(),     # [x, y, z]
            "orientation": orient[0].cpu().numpy().tolist(),  # [w, x, y, z]
        })

    data = {
        "scene_idx":        scene_idx,
        "target":           args.target,
        "target_usd_name":  target_usd_name,
        "target_obj_index": target_idx,
        "objects":          objects,
    }
    with open(os.path.join(scene_dir, "poses.json"), "w") as f:
        json.dump(data, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# 14. Main scene generation loop
# ═══════════════════════════════════════════════════════════════════════════
try:
    print(f"\n=== Generating {args.scene_num} test scenes ===\n")

    for scene_idx in range(args.scene_num):
        scene_dir = os.path.join(OUT_DIR, f"scene_{scene_idx:04d}")
        os.makedirs(scene_dir, exist_ok=True)

        if os.path.exists(os.path.join(scene_dir, "poses.json")):
            print(f"[scene {scene_idx:04d}] SKIP (already exists)")
            continue

        print(f"[scene {scene_idx:04d}] Generating...")

        # Reset all objects to wait area
        object_spawner.initialize()
        for _ in range(30):
            world.step(render=True)

        # Spawn objects with similarity-based ordering
        object_spawner.spawn_with_similarity(
            target_name=args.target,
            world=world,
            stabilization_steps=STABILIZATION_STEPS,
            final_stabilization_steps=FINAL_STABILIZATION_STEPS,
            target_not_bottom_prob=TARGET_NOT_BOTTOM_PROB,
            target_top_prob=TARGET_TOP_PROB,
        )

        # Extra render frames for stable image
        for _ in range(20):
            world.step(render=True)

        # Save RGB images and object poses
        capture_rgb(scene_dir)
        save_poses(scene_dir, scene_idx)

        print(f"[scene {scene_idx:04d}] Done → {scene_dir}")

    print(f"\n=== All {args.scene_num} scenes generated ===")
    print(f"  Output: {OUT_DIR}")

finally:
    world.stop()
    simulation_app.close()
