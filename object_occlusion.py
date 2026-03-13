import os
import argparse
import numpy as np
from isaacsim import SimulationApp

# ==============================================================================
# 0. Argument Parsing (SimulationApp 시작 전에 파싱)
# ==============================================================================
parser = argparse.ArgumentParser(description="Object Occlusion Dataset Generator (Isaac Sim)")
parser.add_argument("--target_name", type=str, required=True, help="타겟 오브젝트 폴더 이름 (예: book_1)")
parser.add_argument("--headless", action="store_true", help="GUI 없이 실행")
parser.add_argument("--list_objects", action="store_true", help="사용 가능한 오브젝트 목록 출력 후 종료")

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
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import omni.replicator.core as rep
import omni.usd
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
BASE_Z   = 0.01                 # 기준 z 높이 (m)
Z_OFFSET = 0.03                 # z층 간격 (m)
Z_LEVELS = 3                    # z층 횟수 (1 = BASE_Z만, 2 = BASE_Z + BASE_Z+Z_OFFSET, ...)

# --- 렌더링 안정화 스텝 ---
RENDER_STABILIZE_STEPS  = 2    # 오브젝트 이동 후 안정화 스텝 수
CAMERA_STABILIZE_STEPS  = 5    # 카메라 텔레포트 후 안정화 스텝 수

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
                    break  # 서브폴더당 첫 번째 파일만 사용
    return assets

all_assets = discover_assets(USD_FILE_DIR)

# --list_objects 처리
if args.list_objects:
    print("\n=== 사용 가능한 오브젝트 목록 ===")
    for folder_name, (usd_name, _, category) in sorted(all_assets.items()):
        print(f"  --target_name {folder_name}  →  {usd_name} ({category})")
    print("================================\n")
    simulation_app.close()
    exit(0)

# target_name 검증
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

# ==============================================================================
# 7. World 설정
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

# Lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000)
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateAngleAttr(0.53)

# ==============================================================================
# 8. Drawer (Static Collider)
# ==============================================================================
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace")

def make_static_collider(prim_path: str):
    """prim과 모든 하위 prim에서 Rigid Body를 제거해 Static Collider로 만듦"""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

make_static_collider("/World/workspace")

# ==============================================================================
# 9. Target Object (Physics 없는 XFormPrim → 강제 배치 가능)
# ==============================================================================
add_reference_to_stage(usd_path=target_usd_path, prim_path="/World/target_object")

# USD 내부의 RigidBody 제거 (physics 없이 강제 배치)
make_static_collider("/World/target_object")

# Semantic 레이블 부여 (segmentation에 사용)
target_prim_usd = stage.GetPrimAtPath("/World/target_object")
if not target_prim_usd.IsValid():
    print(f"[오류] target prim 로드 실패: /World/target_object")
    print(f"  USD 경로: {target_usd_path}")
    simulation_app.close()
    exit(1)

print(f"[OK] target prim 로드 성공: {target_prim_usd.GetPath()}")
sem_data = PrimSemanticData(target_prim_usd)
sem_data.add_entry("class", target_usd_name)

# USD Xform 직접 제어 (physics 없는 static prim에 가장 안정적)
_target_xformable = UsdGeom.Xformable(target_prim_usd)
_target_xformable.ClearXformOpOrder()
_target_translate_op = _target_xformable.AddTranslateOp()
_target_orient_op    = _target_xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
# 초기 위치
_target_translate_op.Set(Gf.Vec3d(0.0, 0.0, float(BASE_Z)))
_target_orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

# ==============================================================================
# 10. 카메라 설정 (텔레포트 방식: 카메라 1개를 5위치로 이동)
# ==============================================================================
def look_at_rotation(cam_pos, target_pos=(0.0, 0.0, 0.0)):
    """카메라가 target_pos를 바라보는 quaternion (w,x,y,z) 반환"""
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    cam_forward = np.array([0.0, 0.0, -1.0])
    rotation, _ = R.align_vectors([direction], [cam_forward])
    q = rotation.as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

# 단일 캡처 카메라 생성
capture_cam = Camera(
    prim_path="/World/capture_camera",
    position=np.array([0.0, 0.0, CAMERA_HEIGHT_OFFSET]),
    resolution=CAMERA_RESOLUTION,
)
capture_cam.initialize()
capture_cam.add_distance_to_image_plane_to_frame()

cam_view = XFormPrimView("/World/capture_camera")

# Replicator instance segmentation annotator
_rep_rp = rep.create.render_product(capture_cam.prim_path, CAMERA_RESOLUTION)
_seg_annotator = rep.AnnotatorRegistry.get_annotator(
    "instance_segmentation", init_params={"colorize": False}
)
_seg_annotator.attach([_rep_rp])

# 카메라 5개 위치/orientation 사전 계산
camera_poses = {}
for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    quat_wxyz = look_at_rotation(cam_pos)
    camera_poses[cam_name] = {
        "position":    cam_pos,
        "orientation": quat_wxyz,
    }

# ==============================================================================
# 11. Simulation 시작
# ==============================================================================
world.play()
for _ in range(20):
    world.step(render=True)

# ==============================================================================
# 12. 유틸 함수
# ==============================================================================
def set_target_pose(x: float, y: float, z: float, yaw_rad: float):
    """target object를 (x, y, z)에 yaw 회전으로 강제 배치 (USD API 직접 사용)"""
    rot = R.from_euler("z", yaw_rad)
    q = rot.as_quat()  # x, y, z, w
    _target_translate_op.Set(Gf.Vec3d(float(x), float(y), float(z)))
    _target_orient_op.Set(Gf.Quatd(float(q[3]), float(q[0]), float(q[1]), float(q[2])))


def capture_and_save(frame_idx: int, mapping_saved_flag: list):
    """카메라 5개를 텔레포트하며 RGB/Depth/Seg 캡처 및 저장.
    mapping_saved_flag: [bool] 리스트로 뮤터블하게 전달 (첫 저장 여부 추적)
    """
    class_colors = capture_and_save.__dict__.setdefault("_class_colors", {})

    for cam_name, pose_info in camera_poses.items():
        pos    = pose_info["position"].astype(np.float32)
        orient = pose_info["orientation"].astype(np.float32)

        pos_tensor    = torch.from_numpy(pos).unsqueeze(0).to("cuda")
        orient_tensor = torch.from_numpy(orient).unsqueeze(0).to("cuda")
        cam_view.set_world_poses(pos_tensor, orient_tensor)

        for _ in range(CAMERA_STABILIZE_STEPS):
            world.step(render=True)

        filename_base = f"{frame_idx:06d}_{cam_name}"

        # ── RGB ──────────────────────────────────────────────────────────────
        rgb = capture_cam.get_rgb()
        if rgb is not None:
            cv2.imwrite(
                os.path.join(rgb_dir, f"{filename_base}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        # ── Depth ─────────────────────────────────────────────────────────────
        depth = capture_cam.get_depth()
        if depth is not None:
            np.save(os.path.join(depth_dir, f"{filename_base}.npy"), depth)

        # ── Segmentation ──────────────────────────────────────────────────────
        seg_data = _seg_annotator.get_data()
        if seg_data is None or not isinstance(seg_data, dict) or "data" not in seg_data:
            continue

        seg_ids = seg_data["data"]
        if seg_ids.ndim == 3:
            seg_ids = seg_ids[:, :, 0]

        info         = seg_data.get("info", {})
        id_to_labels = info.get("idToLabels", {})

        seg_color    = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)
        scene_classes = {}

        for uid in np.unique(seg_ids):
            prim_label = id_to_labels.get(str(int(uid)), "")
            if not prim_label or prim_label in ("BACKGROUND", "UNLABELLED"):
                continue
            if "target_object" not in prim_label:
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

        # JSON mapping: 씬 고정이므로 최초 1회만 저장
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


# ==============================================================================
# 13. 스캔 루프
# ==============================================================================
x_values   = np.arange(X_MIN, X_MAX + XY_STEP * 0.5, XY_STEP)
y_values   = np.arange(Y_MIN, Y_MAX + XY_STEP * 0.5, XY_STEP)
yaw_values = np.arange(0, 360, YAW_STEP_DEG)
z_values   = [BASE_Z + i * Z_OFFSET for i in range(Z_LEVELS)]

num_rotations = len(yaw_values)
total_positions = len(z_values) * len(x_values) * len(y_values) * num_rotations

print("=== 스캔 파라미터 ===")
print(f"  XY 범위  : x=[{X_MIN}, {X_MAX}], y=[{Y_MIN}, {Y_MAX}], step={XY_STEP}m")
print(f"  XY 격자  : {len(x_values)} × {len(y_values)} = {len(x_values)*len(y_values)}점")
print(f"  회전     : {YAW_STEP_DEG}도 간격 → {num_rotations}스텝/위치")
print(f"  Z 레벨   : {Z_LEVELS}층 {z_values}")
print(f"  총 위치  : {total_positions}")
print(f"  총 이미지: {total_positions} × 5 카메라 = {total_positions * 5}장")
print(f"  출력 위치: {output_base}")
print("=====================\n")

mapping_saved_flag = [False]  # 리스트로 뮤터블하게 관리

try:
    frame_idx = 0
    for z_idx, z in enumerate(z_values):
        print(f"[Z층 {z_idx + 1}/{len(z_values)}]  z = {z:.4f}m")

        for y_idx, y in enumerate(y_values):
            for x_idx, x in enumerate(x_values):
                for r_idx, yaw_deg in enumerate(yaw_values):
                    yaw_rad = np.radians(yaw_deg)

                    # 강제 배치
                    set_target_pose(x, y, z, yaw_rad)

                    # 렌더링 안정화
                    for _ in range(RENDER_STABILIZE_STEPS):
                        world.step(render=True)

                    # 5개 카메라 캡처 & 저장
                    capture_and_save(frame_idx, mapping_saved_flag)
                    frame_idx += 1

            # Y 진행률
            progress = (y_idx + 1) / len(y_values) * 100
            print(
                f"  Y {y_idx + 1:3d}/{len(y_values)} ({progress:5.1f}%)"
                f"  |  누적 프레임: {frame_idx}"
            )

    print(f"\n=== 스캔 완료 ===")
    print(f"  총 {frame_idx}개 위치 × 5 카메라 = {frame_idx * 5}장 저장")
    print(f"  저장 위치: {output_base}")

finally:
    world.stop()
    simulation_app.close()
