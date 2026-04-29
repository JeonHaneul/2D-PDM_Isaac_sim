"""
vectorized_object_occlusion.py

Z_LEVELS개 환경을 병렬로 실행하여 object_occlusion.py 대비 ~Z_LEVELS배 속도 향상.
각 환경은 서로 다른 z층을 담당하고, 동일한 (x, y, yaw) 스캔을 동시에 수행.

usage: python vectorized_object_occlusion.py --target_name book_1 [--headless]

파일 번호 구조:
  z층 0: frame 000000 ~ 0+(frames_per_z-1)
  z층 1: frame frames_per_z ~ 2*frames_per_z-1
  z층 2: frame 2*frames_per_z ~ 3*frames_per_z-1
"""

import os
import argparse
import numpy as np
from isaacsim import SimulationApp

# ==============================================================================
# 0. Argument Parsing (SimulationApp 시작 전에 파싱)
# ==============================================================================
parser = argparse.ArgumentParser(description="Vectorized Object Occlusion Dataset Generator")
parser.add_argument("--target_name",  type=str, required=True, help="타겟 오브젝트 폴더 이름 (예: book_1)")
parser.add_argument("--headless",     action="store_true",     help="GUI 없이 실행")
parser.add_argument("--list_objects", action="store_true",     help="사용 가능한 오브젝트 목록 출력 후 종료")

args, unknown = parser.parse_known_args()

# ==============================================================================
# 1. Launch Simulation App
# ==============================================================================
simulation_app = SimulationApp({"headless": args.headless})

# ==============================================================================
# Isaac Sim imports (SimulationApp 시작 후에만 가능)
# ==============================================================================
import torch
import cv2
import json
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
import carb
from pxr import UsdLux, UsdPhysics, UsdGeom, Usd, Gf
from semantics.schema.editor import PrimSemanticData

# ==============================================================================
# 2. 가변 파라미터 (이 블록에서 모두 수정 가능)
# ==============================================================================

# --- XY 스캔 범위 ---
X_MIN, X_MAX = -0.17, 0.17     # x 스캔 범위 (m)
Y_MIN, Y_MAX = -0.17, 0.17     # y 스캔 범위 (m)
XY_STEP      = 0.01             # 이동 간격 (m), 기본 1cm

# --- 회전 파라미터 ---
YAW_STEP_DEG = 30               # yaw 회전 간격 (도), 기본 30도 → 12스텝/위치

# --- Z 파라미터 ---
BASE_Z   = 0.01                 # 기준 z 높이 (m) (book_1 : 0.01, book_2 : 0.01)
Z_OFFSET = 0.03                 # z층 간격 (m)
Z_LEVELS = 3                    # z층 횟수 (= 병렬 환경 개수)

# --- 환경 그리드 간격 ---
GRID_SPACING = 5.0              # 환경 간 거리 (m), 카메라 시야가 겹치지 않도록 충분히 크게

# --- 렌더링 안정화 스텝 ---
RENDER_STABILIZE_STEPS = 10     # 오브젝트 이동 후 안정화 스텝 수

# --- 카메라 파라미터 ---
CAMERA_HEIGHT_OFFSET = 3.0     # 카메라 z 높이 (m)
CAMERA_XY_OFFSET     = 1.0     # left/right/top/bottom 카메라 xy 오프셋 (m)
CAMERA_RESOLUTION    = (640, 480)

# ==============================================================================
# 3. 카메라 설정 (5개)
# ==============================================================================
CAMERA_CONFIGS = {
    "center": {"offset": (0.0,               0.0)},
    "left":   {"offset": (-CAMERA_XY_OFFSET, 0.0)},
    "right":  {"offset": ( CAMERA_XY_OFFSET, 0.0)},
    "top":    {"offset": (0.0,               CAMERA_XY_OFFSET)},
    "bottom": {"offset": (0.0,              -CAMERA_XY_OFFSET)},
}

# ==============================================================================
# 4. Asset 경로
# ==============================================================================
SRC_DIR            = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR          = os.path.join(SRC_DIR, "asset")
USD_FILE_DIR       = os.path.join(ASSET_DIR, "260303")
WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "USD", "drawer.usd")

# ==============================================================================
# 5. Asset 탐색
# ==============================================================================
def discover_assets(usd_folder_dir, extensions=(".usd", ".usdc")):
    """260303/ 하위의 모든 카테고리 폴더를 자동 탐색.
    구조: {usd_folder_dir}/{category}/{subdir}/{file}.usd(c)
    Returns: {folder_name(소문자): (usd_name, usd_path, category)}
    """
    assets = {}
    if not os.path.isdir(usd_folder_dir):
        return assets
    for category in sorted(os.listdir(usd_folder_dir)):
        cat_dir = os.path.join(usd_folder_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for subdir in sorted(os.listdir(cat_dir)):
            subdir_path = os.path.join(cat_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for f in sorted(os.listdir(subdir_path)):
                if f.lower().endswith(extensions):
                    usd_name = os.path.splitext(f)[0]
                    usd_path = os.path.join(subdir_path, f)
                    assets[subdir.lower()] = (usd_name, usd_path, category)
                    break
    return assets

all_assets = discover_assets(USD_FILE_DIR)

if args.list_objects:
    print("\n=== 사용 가능한 오브젝트 목록 ===")
    for folder_name, (usd_name, _, category) in sorted(all_assets.items()):
        print(f"  --target_name {folder_name}  →  {usd_name} ({category})")
    print("================================\n")
    simulation_app.close()
    exit(0)

target_key = args.target_name.lower()
if target_key not in all_assets:
    matches = [k for k in all_assets if target_key in k or k in target_key]
    if len(matches) == 1:
        target_key = matches[0]
        print(f"[INFO] '{args.target_name}' → '{target_key}'로 매칭")
    else:
        print(f"\n오류: '{args.target_name}'을(를) 찾을 수 없습니다.")
        if matches:
            print(f"비슷한 이름: {matches}")
        print("--list_objects 옵션으로 목록을 확인하세요.")
        simulation_app.close()
        exit(1)

target_usd_name, target_usd_path, target_category = all_assets[target_key]
print(f"\n[타겟 오브젝트] {args.target_name} → {target_usd_name} ({target_category})")
print(f"  USD 경로: {target_usd_path}\n")

# ==============================================================================
# 6. 출력 디렉토리 설정
# ==============================================================================
output_base = os.path.join(SRC_DIR, "output", args.target_name, "target")
rgb_dir     = os.path.join(output_base, "rgb")
depth_dir   = os.path.join(output_base, "depth")
seg_dir     = os.path.join(output_base, "seg")
os.makedirs(rgb_dir,   exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)
os.makedirs(seg_dir,   exist_ok=True)

empty_scene_base      = os.path.join(output_base, "empty_scene")
empty_scene_rgb_dir   = os.path.join(empty_scene_base, "rgb")
empty_scene_depth_dir = os.path.join(empty_scene_base, "depth")
empty_scene_seg_dir   = os.path.join(empty_scene_base, "seg")
os.makedirs(empty_scene_rgb_dir,   exist_ok=True)
os.makedirs(empty_scene_depth_dir, exist_ok=True)
os.makedirs(empty_scene_seg_dir,   exist_ok=True)

# ==============================================================================
# 7. 스캔 파라미터 사전 계산
# ==============================================================================
x_values   = np.arange(X_MIN, X_MAX + XY_STEP * 0.5, XY_STEP)
y_values   = np.arange(Y_MIN, Y_MAX + XY_STEP * 0.5, XY_STEP)
yaw_values = np.arange(0, 360, YAW_STEP_DEG)
z_values   = [BASE_Z + i * Z_OFFSET for i in range(Z_LEVELS)]

frames_per_z    = len(x_values) * len(y_values) * len(yaw_values)
total_positions = frames_per_z * Z_LEVELS

# ==============================================================================
# 8. World 설정
# ==============================================================================
world = World(physics_dt=1/120.0, backend="torch", device="cuda")

physics_context = world.get_physics_context()
physics_context.set_solver_type("TGS")
physics_context.enable_ccd(True)

GroundPlane(
    prim_path="/World/GroundPlane",
    z_position=0,
    color=np.array([1.0, 1.0, 1.0]),
)

stage = omni.usd.get_context().get_stage()

dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000)
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateAngleAttr(0.53)

# ==============================================================================
# 9. 유틸 함수
# ==============================================================================
def make_static_collider(prim_path: str):
    """prim과 모든 하위 prim에서 Rigid Body를 제거해 Static Collider로 만듦"""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

def look_at_rotation(cam_pos, target_pos=(0.0, 0.0, 0.0)):
    """카메라가 target_pos를 바라보는 quaternion (w,x,y,z) 반환"""
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    rotation, _ = R.align_vectors([direction], [np.array([0.0, 0.0, -1.0])])
    q = rotation.as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

# ==============================================================================
# 10. 베이스 환경 (_0) 구성
# ==============================================================================
# Workspace _0
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace_0")
make_static_collider("/World/workspace_0")

# Target _0
add_reference_to_stage(usd_path=target_usd_path, prim_path="/World/target_object_0")
make_static_collider("/World/target_object_0")

target_prim_0 = stage.GetPrimAtPath("/World/target_object_0")
if not target_prim_0.IsValid():
    print(f"[오류] target prim 로드 실패: /World/target_object_0")
    simulation_app.close()
    exit(1)

print(f"[OK] target prim 로드 성공: {target_prim_0.GetPath()}")
PrimSemanticData(target_prim_0).add_entry("class", target_usd_name)

# 카메라 _0 prim 생성 (initialize만, add_distance는 클론 후 일괄 처리)
for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset  = cam_config["offset"]
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    cam_0   = Camera(
        prim_path=f"/World/camera_{cam_name}_0",
        position=cam_pos,
        resolution=CAMERA_RESOLUTION,
    )
    cam_0.initialize()

# ==============================================================================
# 11. 환경 복제 (GridCloner)
# ==============================================================================
cloner = GridCloner(spacing=GRID_SPACING)

workspace_paths   = cloner.generate_paths("/World/workspace", Z_LEVELS)
camera_paths_dict = {
    cam_name: cloner.generate_paths(f"/World/camera_{cam_name}", Z_LEVELS)
    for cam_name in CAMERA_CONFIGS
}

cloner.clone(source_prim_path="/World/workspace_0", prim_paths=workspace_paths)
for cam_name in CAMERA_CONFIGS:
    cloner.clone(
        source_prim_path=f"/World/camera_{cam_name}_0",
        prim_paths=camera_paths_dict[cam_name],
    )

# ==============================================================================
# 12. target_object 직접 생성 (클로너 미사용 → quatf/quatd 타입 충돌 회피)
#     USD Xform API로 직접 제어 (set_world_poses는 metersPerUnit 불일치 시 scale 오적용)
# ==============================================================================
_target_translate_ops: list = []
_target_orient_ops:    list = []

for z_idx in range(1, Z_LEVELS):
    prim_path = f"/World/target_object_{z_idx}"
    add_reference_to_stage(usd_path=target_usd_path, prim_path=prim_path)
    make_static_collider(prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    PrimSemanticData(prim).add_entry("class", target_usd_name)
    print(f"[OK] target prim 로드 성공: {prim_path}")

# 모든 target prim에 USD Xform ops 설정 (z_idx=0은 이미 로드됨)
# metersPerUnit 불일치 scale을 보존하고, translate는 로컬 좌표로 환산
_unit_scale = 1.0  # scale factor (ref_mpu / main_mpu), 첫 prim에서 감지

for z_idx in range(Z_LEVELS):
    prim = stage.GetPrimAtPath(f"/World/target_object_{z_idx}")
    xf   = UsdGeom.Xformable(prim)

    scale_backups = []
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            parts    = op.GetOpName().split(":")
            suffix   = parts[2] if len(parts) > 2 else ""
            type_str = str(op.GetAttr().GetTypeName())
            prec     = (UsdGeom.XformOp.PrecisionFloat
                        if "float" in type_str
                        else UsdGeom.XformOp.PrecisionDouble)
            val = op.Get()
            scale_backups.append((suffix, val, prec))
            # 첫 prim에서 unit scale 감지 (uniform scale 가정)
            if z_idx == 0 and val is not None:
                _unit_scale = float(val[0])

    xf.ClearXformOpOrder()
    for suffix, val, prec in scale_backups:
        s_op = xf.AddScaleOp(prec, opSuffix=suffix)
        if val is not None:
            s_op.Set(val)

    t_op = xf.AddTranslateOp()
    o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    # 초기 off-screen 위치: world 1000m → 로컬 좌표 = 1000 / unit_scale
    t_op.Set(Gf.Vec3d(1000.0 / _unit_scale, 1000.0 / _unit_scale, 1000.0 / _unit_scale))
    o_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    _target_translate_ops.append(t_op)
    _target_orient_ops.append(o_op)

# ==============================================================================
# 13. 카메라 wrapper / render product / seg annotator (모든 환경)
# ==============================================================================
cameras_all:        list[dict] = [{} for _ in range(Z_LEVELS)]
seg_annotators_all: list[dict] = [{} for _ in range(Z_LEVELS)]

for z_idx in range(Z_LEVELS):
    for cam_name in CAMERA_CONFIGS:
        prim_path = f"/World/camera_{cam_name}_{z_idx}"

        cam = Camera(prim_path=prim_path, resolution=CAMERA_RESOLUTION)
        cam.initialize()
        cam.add_distance_to_image_plane_to_frame()

        rp      = rep.create.render_product(prim_path, CAMERA_RESOLUTION)
        seg_ann = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation", init_params={"colorize": False}
        )
        seg_ann.attach([rp])

        cameras_all[z_idx][cam_name]        = cam
        seg_annotators_all[z_idx][cam_name] = seg_ann

# ==============================================================================
# 14. Simulation 시작
# ==============================================================================
world.play()

# 실제 환경 원점 획득 (GridCloner 배치 확인 및 카메라 위치 계산에 사용)
workspaces_view        = XFormPrimView("/World/workspace_*")
env_pos_tensor, _      = workspaces_view.get_world_poses()
env_origins            = env_pos_tensor.cpu().numpy().astype(np.float64)  # (Z_LEVELS, 3)

# 초기 target 위치를 시야 밖으로 설정 (USD Xform API 사용, 이미 ops 생성 시 설정됨)

# 카메라 world pose 설정 (실측 env_origins 기반)
for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset   = cam_config["offset"]
    cam_view = XFormPrimView(f"/World/camera_{cam_name}_*")

    cam_positions = env_pos_tensor.clone().float()
    cam_positions[:, 0] += offset[0]
    cam_positions[:, 1] += offset[1]
    cam_positions[:, 2] += CAMERA_HEIGHT_OFFSET

    local_cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    quat_wxyz     = look_at_rotation(local_cam_pos)
    cam_orients   = (
        torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda")
        .unsqueeze(0)
        .repeat(Z_LEVELS, 1)
    )
    cam_view.set_world_poses(cam_positions, cam_orients)

# 초기 안정화
for _ in range(20):
    world.step(render=True)

# ==============================================================================
# 15. 주요 함수
# ==============================================================================
def set_all_targets(x: float, y: float, yaw_rad: float):
    """모든 환경의 target을 동일한 (x, y, yaw)로, z는 각 환경의 z_level로 배치"""
    rot = R.from_euler("z", yaw_rad)
    q   = rot.as_quat()  # x,y,z,w
    quat_f = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))

    for i in range(Z_LEVELS):
        # world 좌표 → 로컬 좌표: scale(metersPerUnit 보정) 적용된 prim은
        # translate를 scale로 나눠야 world position이 정확히 반영됨
        _target_translate_ops[i].Set(Gf.Vec3d(
            float((env_origins[i, 0] + x) / _unit_scale),
            float((env_origins[i, 1] + y) / _unit_scale),
            float(z_values[i]             / _unit_scale),
        ))
        _target_orient_ops[i].Set(quat_f)


def capture_and_save_env(z_idx: int, frame_idx: int, mapping_saved_flag: list):
    """환경 z_idx의 5개 카메라에서 RGB/Depth/Seg 캡처 및 저장"""
    class_colors  = capture_and_save_env.__dict__.setdefault("_class_colors", {})
    target_substr = f"target_object_{z_idx}"

    for cam_name, cam in cameras_all[z_idx].items():
        filename_base = f"{frame_idx:06d}_{cam_name}"

        # ── RGB ──────────────────────────────────────────────────────────────
        rgb = cam.get_rgb()
        if rgb is not None:
            cv2.imwrite(
                os.path.join(rgb_dir, f"{filename_base}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        # ── Depth ─────────────────────────────────────────────────────────────
        depth = cam.get_depth()
        if depth is not None:
            np.save(os.path.join(depth_dir, f"{filename_base}.npy"), depth)

        # ── Segmentation ──────────────────────────────────────────────────────
        seg_data = seg_annotators_all[z_idx][cam_name].get_data()
        if seg_data is None or not isinstance(seg_data, dict) or "data" not in seg_data:
            continue

        seg_ids = seg_data["data"]
        if seg_ids.ndim == 3:
            seg_ids = seg_ids[:, :, 0]

        id_to_labels = seg_data.get("info", {}).get("idToLabels", {})
        seg_color    = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)
        scene_classes = {}

        for uid in np.unique(seg_ids):
            prim_label = id_to_labels.get(str(int(uid)), "")
            if not prim_label or prim_label in ("BACKGROUND", "UNLABELLED"):
                continue
            # 이 환경의 target prim만 필터링
            if target_substr not in prim_label:
                continue

            class_name = target_usd_name
            if class_name not in class_colors:
                hue   = abs(hash(class_name)) % 180
                color = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
                class_colors[class_name] = color.tolist()

            color = class_colors[class_name]
            seg_color[seg_ids == uid] = color
            scene_classes[class_name] = {"color_bgr": color}

        cv2.imwrite(os.path.join(seg_dir, f"{filename_base}.png"), seg_color)

        # JSON mapping: 최초 1회만 저장
        if not mapping_saved_flag[0] and scene_classes:
            mapping_data = {
                "target_folder_name": args.target_name,
                "target_usd_name":    target_usd_name,
                "category":           target_category,
                "classes":            scene_classes,
            }
            json_path = os.path.join(output_base, "mapping.json")
            with open(json_path, "w") as f:
                json.dump(mapping_data, f, indent=2, ensure_ascii=False)
            mapping_saved_flag[0] = True
            print(f"  [mapping.json 저장 완료] {json_path}")


def capture_empty_scene():
    """모든 환경의 target을 숨기고 env_0 카메라로 빈 씬 캡처"""
    print("\n[빈 서랍 환경 캡처 시작]")

    # 모든 환경 target 숨기기
    for z_idx in range(Z_LEVELS):
        prim = stage.GetPrimAtPath(f"/World/target_object_{z_idx}")
        UsdGeom.Imageable(prim).MakeInvisible()
    for _ in range(20):
        world.step(render=True)

    # env_0 카메라로 캡처 (모든 환경 동일한 빈 씬이므로 env_0만 저장)
    for cam_name, cam in cameras_all[0].items():
        rgb = cam.get_rgb()
        if rgb is not None:
            cv2.imwrite(
                os.path.join(empty_scene_rgb_dir, f"{cam_name}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        depth = cam.get_depth()
        if depth is not None:
            np.save(os.path.join(empty_scene_depth_dir, f"{cam_name}.npy"), depth)

        seg_data = seg_annotators_all[0][cam_name].get_data()
        if seg_data is not None and isinstance(seg_data, dict) and "data" in seg_data:
            seg_ids = seg_data["data"]
            if seg_ids.ndim == 3:
                seg_ids = seg_ids[:, :, 0]
            cv2.imwrite(
                os.path.join(empty_scene_seg_dir, f"{cam_name}.png"),
                np.zeros((*seg_ids.shape, 3), dtype=np.uint8),
            )

        print(f"  [{cam_name}] 캡처 완료")

    # 모든 환경 target 복원
    for z_idx in range(Z_LEVELS):
        prim = stage.GetPrimAtPath(f"/World/target_object_{z_idx}")
        UsdGeom.Imageable(prim).MakeVisible()
    for _ in range(20):
        world.step(render=True)

    print(f"  → 저장 위치: {empty_scene_base}\n")


# ==============================================================================
# 16. 스캔 파라미터 출력
# ==============================================================================
print("=== 스캔 파라미터 ===")
print(f"  XY 범위    : x=[{X_MIN}, {X_MAX}], y=[{Y_MIN}, {Y_MAX}], step={XY_STEP}m")
print(f"  XY 격자    : {len(x_values)} × {len(y_values)} = {len(x_values)*len(y_values)}점")
print(f"  회전       : {YAW_STEP_DEG}도 간격 → {len(yaw_values)}스텝/위치")
print(f"  Z 레벨     : {Z_LEVELS}층 {z_values}")
print(f"  z층당 프레임: {frames_per_z}")
print(f"  총 위치    : {total_positions}")
print(f"  총 이미지  : {total_positions} × 5 카메라 = {total_positions * 5}장")
print(f"  병렬 환경  : {Z_LEVELS}개  →  예상 속도 향상 ~{Z_LEVELS}×")
print(f"  출력 위치  : {output_base}")
print("=====================\n")

# ==============================================================================
# 17. 빈 서랍 환경 캡처
# ==============================================================================
_empty_done = all(
    os.path.exists(os.path.join(empty_scene_depth_dir, f"{c}.npy"))
    for c in CAMERA_CONFIGS
)
if _empty_done:
    print(f"[빈 서랍 환경] 이미 존재, 건너뜀: {empty_scene_base}")
else:
    capture_empty_scene()

# ==============================================================================
# 18. 스캔 루프 (x, y, yaw 공통 / z는 각 환경이 동시에 처리)
# ==============================================================================
mapping_saved_flag = [False]

try:
    local_idx = 0  # z층 내 로컬 프레임 인덱스 (0 ~ frames_per_z - 1)

    for y_idx, y in enumerate(y_values):
        for x_idx, x in enumerate(x_values):
            for r_idx, yaw_deg in enumerate(yaw_values):
                yaw_rad = np.radians(yaw_deg)

                # 모든 환경의 target 동시 배치 (각자의 z층)
                set_all_targets(x, y, yaw_rad)

                # PT accumulation 초기화 (2회)
                carb.settings.get_settings().set("/rtx/resetPtAccumulation", True)
                world.step(render=True)
                carb.settings.get_settings().set("/rtx/resetPtAccumulation", True)

                # 렌더링 안정화 (나머지 스텝)
                for _ in range(RENDER_STABILIZE_STEPS - 1):
                    world.step(render=True)

                # 모든 환경에서 캡처 및 저장
                for z_idx in range(Z_LEVELS):
                    frame_idx = z_idx * frames_per_z + local_idx
                    capture_and_save_env(z_idx, frame_idx, mapping_saved_flag)

                local_idx += 1

        # Y 진행률
        progress = (y_idx + 1) / len(y_values) * 100
        print(
            f"  Y {y_idx + 1:3d}/{len(y_values)} ({progress:5.1f}%)"
            f"  |  로컬 프레임: {local_idx} / {frames_per_z}"
            f"  |  총 저장: {local_idx * Z_LEVELS * 5}장"
        )

    print(f"\n=== 스캔 완료 ===")
    print(f"  {local_idx}포즈 × {Z_LEVELS}층 × 5 카메라 = {local_idx * Z_LEVELS * 5}장 저장")
    print(f"  저장 위치: {output_base}")

finally:
    world.stop()
    simulation_app.close()
