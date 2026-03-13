import os
import argparse
import torch
import numpy as np
from isaacsim import SimulationApp


# ==============================================================================
# 0. Argument Parsing (SimulationApp 시작 전에 파싱)
# ==============================================================================
parser = argparse.ArgumentParser(description="Isaac Sim Scene Generator with Similarity-based Spawning")

parser.add_argument(
    "--target", 
    type=str, 
    default=None, 
    help="타겟 오브젝트 이름 (예: book_1, fruit_2). 지정하면 유사도 기반 순차 스폰"
)
parser.add_argument(
    "--num_scenes",
    type=int,
    default=1,
    help="생성할 씬 개수 (각 환경에서 반복)"
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="GUI 없이 실행"
)
parser.add_argument(
    "--list_objects",
    action="store_true",
    help="사용 가능한 오브젝트 목록 출력 후 종료"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="병렬로 실행할 환경 개수 (기본값: 4)"
)

args, unknown = parser.parse_known_args()

# ==============================================================================
# 1. Launch Simulation App
# ==============================================================================
simulation_app = SimulationApp({"headless": args.headless})

from isaacsim.core.api import World
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.cloner import GridCloner
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from omni.isaac.core.prims import XFormPrimView
import isaacsim.core.utils.torch.rotations as rot_utils

from object_spawner import ObjectSpawner

# ==============================================================================
# 2. Configuration
# ==============================================================================
NUM_ENVS = args.num_envs
GRID_SPACING = 3

# --- 카메라 파라미터 ---
CAMERA_HEIGHT_OFFSET = 3    # 모든 카메라의 z 높이
CAMERA_XY_OFFSET = 1       # left/right/top/bottom 카메라의 x 또는 y 오프셋

# --- 스폰 안정화 스텝 ---
STABILIZATION_STEPS       = 60   # 각 오브젝트 투하 후 안정화 스텝 수
FINAL_STABILIZATION_STEPS = 120  # 모든 오브젝트 투하 후 최종 안정화 스텝 수

# --- 타겟 레이어 확률 ---
TARGET_NOT_BOTTOM_PROB = 0.15  # target이 맨 아래가 아닐 확률 (0.0 = 항상 아래)
TARGET_TOP_PROB        = 0.6  # 맨 아래가 아닌 경우 중 맨 위(마지막)일 확률

# 카메라 설정: offset은 (x, y)만, z는 CAMERA_HEIGHT_OFFSET 사용
CAMERA_CONFIGS = {
    "center": {"offset": (0.0, 0.0)},
    "left":   {"offset": (-CAMERA_XY_OFFSET, 0.0)},
    "right":  {"offset": (CAMERA_XY_OFFSET, 0.0)},
    "top":    {"offset": (0.0, CAMERA_XY_OFFSET)},
    "bottom": {"offset": (0.0, -CAMERA_XY_OFFSET)},
}

SRC_DIR = os.path.dirname(__file__)
ASSET_DIR = os.path.join(SRC_DIR, "asset")
USD_FILE_DIR = os.path.join(ASSET_DIR, "260303")

WORKSPACE_USD_PATH = os.path.join(ASSET_DIR, "USD", "drawer.usd")

# Workspace surface bounds for random object placement
WORKSPACE_BOUNDS = {
    "x": (-0.15, 0.15),
    "y": (-0.15, 0.15),
    "z_surface": 0.01,
    "z_drop": 0.2,  # 오브젝트 투하 높이
}

# ==============================================================================
# 3. Create World & Ground Plane
# ==============================================================================
world = World(physics_dt=1 / 120.0, backend="torch", device="cuda")  # 더 작은 timestep

# 물리 안정성 설정 (충돌 투과 방지)
physics_context = world.get_physics_context()
physics_context.set_solver_type("TGS")  # Temporal Gauss-Seidel (더 안정적)
physics_context.enable_ccd(True)  # Continuous Collision Detection 활성화

GroundPlane(
    prim_path="/World/GroundPlane",
    z_position=0,
    color=torch.tensor([1.0, 1.0, 1.0]),
)

# 3-1. Add Lighting (Stage Light에서 보이도록)
from pxr import UsdLux, UsdPhysics, Usd
import omni.usd
stage = omni.usd.get_context().get_stage()

# Dome Light - 환경 전체를 비추는 조명 (HDRI 대신 사용)
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(1000)

# Distant Light - 태양광처럼 평행한 방향성 조명
distant_light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
distant_light.CreateIntensityAttr(1000)
distant_light.CreateAngleAttr(0.53)

# ==============================================================================
# 4. Build Base Environment (env_0)
# ==============================================================================
# 4-1. Workspace (Static Collider로 설정 - 고정된 물체)
add_reference_to_stage(usd_path=WORKSPACE_USD_PATH, prim_path="/World/workspace_0")

# drawer의 모든 하위 prim에서 Rigid Body 제거 (Static Collider만 유지)
def make_static_collider(prim_path: str):
    """prim과 모든 하위 prim에서 Rigid Body를 제거하고 Static Collider로 만듦"""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim.IsValid():
        return
    
    for prim in Usd.PrimRange(root_prim):
        # Rigid Body API 제거 (있으면)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

make_static_collider("/World/workspace_0")

# 4-2. Create objects at default position (outside workspace)
object_spawner = ObjectSpawner(
    world=world,
    categories=["Book", "Toy", "Fruit", "Packaged_food"],
    usd_folder_dir=USD_FILE_DIR,
    container_prim_path="/World/Objects_0",
    workspace_bounds=WORKSPACE_BOUNDS,
    default_position=torch.tensor([0.0, 0.7, 0.05]),
    num_to_spawn=None,          # load all available assets
    extensions=(".usd",),       # only .usd files
)

# 각 오브젝트 prim에 semantic 레이블 추가 (클론 전에 해야 클론에도 상속됨)
# instance_segmentation이 오브젝트 단위로 그룹핑하기 위해 필요
from semantics.schema.editor import PrimSemanticData
for prim_path, obj_name in zip(object_spawner._spawned_paths, object_spawner._spawned_names):
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        sem_data = PrimSemanticData(prim)
        sem_data.add_entry("class", obj_name)

# 4-3. Cameras (5개: center, left, right, top, bottom)
# 모든 카메라가 같은 높이(z=3.5)에서 서랍 중앙(0,0)을 바라봄
import numpy as np
from scipy.spatial.transform import Rotation as R

def look_at_rotation(cam_pos, target_pos=[0, 0, 0]):
    """카메라가 target을 바라보는 quaternion (w,x,y,z) 반환"""
    direction = np.array(target_pos) - np.array(cam_pos)
    direction = direction / np.linalg.norm(direction)
    # 기본 카메라 방향: -Z
    cam_forward = np.array([0, 0, -1])
    rotation, _ = R.align_vectors([direction], [cam_forward])
    q = rotation.as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])  # w,x,y,z

cameras = {}

for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_pos = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    
    # 카메라 생성 (orientation은 클론 후 설정)
    cam = Camera(
        prim_path=f"/World/camera_{cam_name}_0",
        position=np.array(cam_pos),
        resolution=(640, 480),
    )
    cam.initialize()
    
    cameras[cam_name] = cam

# center 카메라에 depth/segmentation 어노테이터 추가 (캡처에 사용되는 카메라)
cameras["center"].add_distance_to_image_plane_to_frame()

# Camera 래퍼가 colorize 파라미터를 지원하지 않으므로 Replicator API 직접 사용
import omni.replicator.core as rep
_rep_rp = rep.create.render_product(cameras["center"].prim_path, (640, 480))
_seg_annotator = rep.AnnotatorRegistry.get_annotator(
    "instance_segmentation", init_params={"colorize": False}
)
_seg_annotator.attach([_rep_rp])

# ==============================================================================
# 5. Clone Environments
# ==============================================================================
cloner = GridCloner(spacing=GRID_SPACING)

workspace_paths = cloner.generate_paths("/World/workspace", NUM_ENVS)
object_paths = cloner.generate_paths("/World/Objects", NUM_ENVS)

# 각 카메라 타입별로 경로 생성 및 클론
camera_paths_dict = {}
for cam_name in CAMERA_CONFIGS.keys():
    camera_paths_dict[cam_name] = cloner.generate_paths(f"/World/camera_{cam_name}", NUM_ENVS)

cloner.clone(source_prim_path="/World/workspace_0", prim_paths=workspace_paths)
cloner.clone(source_prim_path="/World/Objects_0", prim_paths=object_paths)

for cam_name in CAMERA_CONFIGS.keys():
    cloner.clone(
        source_prim_path=f"/World/camera_{cam_name}_0",
        prim_paths=camera_paths_dict[cam_name]
    )

# ==============================================================================
# 6. Arrange Cloned Poses
# ==============================================================================
workspaces_view = XFormPrimView("/World/workspace_*")
objects_view = XFormPrimView("/World/Objects_*")

# 각 카메라 타입별 View 생성
camera_views_dict = {}
for cam_name in CAMERA_CONFIGS.keys():
    camera_views_dict[cam_name] = XFormPrimView(f"/World/camera_{cam_name}_*")

# Align object containers with workspace positions
positions, orientations = workspaces_view.get_world_poses()
workspaces_view.set_world_poses(positions, orientations)
objects_view.set_world_poses(positions, orientations)

# 각 카메라를 workspace 위치 기준으로 배치
# 모든 카메라: z = CAMERA_HEIGHT_OFFSET, xy는 각 카메라별 오프셋 적용
# offset 값에 따라 자동으로 서랍 중앙(0,0,0)을 바라보는 orientation 계산

# 카메라 orientation 저장 (텔레포트 시 재사용)
camera_orientations = {}

for cam_name, cam_config in CAMERA_CONFIGS.items():
    offset = cam_config["offset"]
    cam_view = camera_views_dict[cam_name]
    
    # workspace 위치 + xy오프셋 + z높이
    cam_positions = positions.clone()
    cam_positions[:, 0] += offset[0]  # x 오프셋
    cam_positions[:, 1] += offset[1]  # y 오프셋
    cam_positions[:, 2] += CAMERA_HEIGHT_OFFSET  # z 높이 (모든 카메라 동일)
    
    # offset 기반으로 서랍 중앙을 바라보는 orientation 계산
    cam_pos_local = np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
    quat_wxyz = look_at_rotation(cam_pos_local, target_pos=[0, 0, 0])
    
    # orientation 저장
    camera_orientations[cam_name] = quat_wxyz
    
    # 모든 환경에 동일한 orientation 적용
    cam_orients = torch.tensor(quat_wxyz, dtype=torch.float32, device="cuda").unsqueeze(0).repeat(NUM_ENVS, 1)
    cam_view.set_world_poses(cam_positions, cam_orients)

# workspace 원점 위치 저장 (텔레포트 시 사용)
workspace_origins = positions.clone()

# 텔레포트용 카메라 뷰 (center 카메라 하나만 사용)
# XFormPrimView를 통해 텔레포트하면 GUI와 동일하게 작동
teleport_cam = cameras["center"]
teleport_cam_view = XFormPrimView("/World/camera_center_0")

# Create per-item views that span all cloned environments
object_spawner.setup_cloned_views(num_envs=NUM_ENVS)

# ==============================================================================
# 7. Simulation Loop
# ==============================================================================
world.play()

# Start with objects at default (outside workspace)
object_spawner.initialize()
world.step(render=True)

# 사용 가능한 오브젝트 목록 출력 (항상)
available_objects = object_spawner.get_target_candidates()
print("\n=== 사용 가능한 오브젝트 목록 ===")
print("  [입력 가능한 이름] → [실제 USD 파일]")
for folder_name, usd_name in available_objects:
    category = object_spawner.objects_class.get(usd_name, "unknown")
    print(f"  --target {folder_name}  →  {usd_name} ({category})")
print("================================\n")

# --list_objects 옵션: 목록 출력 후 종료
if args.list_objects:
    simulation_app.close()
    exit(0)

# ==============================================================================
# 이미지 캡처 함수 (텔레포트 방식: 카메라 1개를 5개 위치로 이동하며 캡처)
# ==============================================================================
import cv2

def save_scene_images(scene_idx: int, target_name: str):
    """
    텔레포트 방식: 카메라 1개를 각 위치로 이동하며 캡처
    260305의 look_at_rotation 사용 (로컬 좌표 기준)
    
    저장 구조:
    output/{target_name}/scene/
        ├── rgb/scene{:05d}_env{:04d}_{camera}.png
        ├── depth/scene{:05d}_env{:04d}_{camera}.npy
        └── seg/scene{:05d}_env{:04d}_{camera}.png
    """
    output_base = os.path.join(os.path.dirname(__file__), "output", target_name, "scene")
    rgb_dir = os.path.join(output_base, "rgb")
    depth_dir = os.path.join(output_base, "depth")
    seg_dir = os.path.join(output_base, "seg")
    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(seg_dir, exist_ok=True)
    
    saved_count = 0
    all_scene_classes = {}  # 모든 env/카메라에서 보이는 물체 누적

    # 렌더링 안정화
    for _ in range(20):
        world.step(render=True)

    # XFormPrimView를 통해 텔레포트 (GUI와 동일하게 작동)
    for env_idx in range(NUM_ENVS):
        env_origin = workspace_origins[env_idx].cpu().numpy()
        
        for cam_name, cam_config in CAMERA_CONFIGS.items():
            offset = cam_config["offset"]
            
            # 월드 좌표에서의 카메라 위치
            cam_world_pos = env_origin + np.array([offset[0], offset[1], CAMERA_HEIGHT_OFFSET])
            
            # 저장된 orientation 사용
            quat_wxyz = camera_orientations[cam_name]
            
            # XFormPrimView로 텔레포트 (Camera.set_world_pose 대신!)
            pos_tensor = torch.from_numpy(np.array(cam_world_pos, dtype=np.float32)).unsqueeze(0).to("cuda")
            orient_tensor = torch.from_numpy(np.array(quat_wxyz, dtype=np.float32)).unsqueeze(0).to("cuda")
            teleport_cam_view.set_world_poses(pos_tensor, orient_tensor)
            
            # 렌더링 대기
            for _ in range(5):
                world.step(render=True)
            
            filename_base = f"scene{scene_idx+1:05d}_env{env_idx:04d}_{cam_name}"
            
            # RGB 캡처
            rgb = teleport_cam.get_rgb()
            if rgb is not None:
                cv2.imwrite(os.path.join(rgb_dir, f"{filename_base}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            
            # Depth 캡처
            depth = teleport_cam.get_depth()
            if depth is not None:
                np.save(os.path.join(depth_dir, f"{filename_base}.npy"), depth)

            # Segmentation 캡처 (Replicator annotator에서 직접 취득)
            seg_data = _seg_annotator.get_data()
            if seg_data is not None and isinstance(seg_data, dict) and "data" in seg_data:
                seg_ids = seg_data["data"]  # (H, W) 또는 (H, W, 1) uint32
                if seg_ids.ndim == 3:
                    seg_ids = seg_ids[:, :, 0]

                # id_to_labels: {id_str → prim_path_str}
                # 예: {'6': '/World/Objects_0/object_4', ...}
                info = seg_data.get("info", {})
                id_to_labels = info.get("idToLabels", {})

                # object_N 키 → 오브젝트 이름 매핑 (클론 환경 통합)
                # /World/Objects_2/object_4 → "object_4" → _spawned_names[4]
                obj_key_to_name = {
                    p.split("/")[-1]: n
                    for p, n in zip(object_spawner._spawned_paths, object_spawner._spawned_names)
                }

                # 클래스명 → 결정론적 색상 (클래스명 해시 기반, 전 씬 동일)
                if not hasattr(save_scene_images, "_class_colors"):
                    save_scene_images._class_colors = {}
                class_colors = save_scene_images._class_colors

                seg_color = np.zeros((*seg_ids.shape, 3), dtype=np.uint8)
                scene_classes = {}

                for uid in np.unique(seg_ids):
                    prim_label = id_to_labels.get(str(int(uid)), "")
                    if not prim_label or prim_label in ("BACKGROUND", "UNLABELLED"):
                        continue
                    obj_key = prim_label.split("/")[-1]  # "object_4"
                    class_name = obj_key_to_name.get(obj_key, "")
                    if not class_name:
                        continue

                    if class_name not in class_colors:
                        hue = abs(hash(class_name)) % 180
                        color = cv2.cvtColor(np.uint8([[[hue, 220, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
                        class_colors[class_name] = color.tolist()

                    color = class_colors[class_name]
                    seg_color[seg_ids == uid] = color
                    scene_classes[class_name] = {"color_rgb": color}

                cv2.imwrite(os.path.join(seg_dir, f"{filename_base}.png"), seg_color)

                # 모든 env/카메라에서 보이는 물체 누적
                all_scene_classes.update(scene_classes)

            saved_count += 1

    # 씬별 매핑 JSON 저장 (모든 env/카메라 누적 후 1회)
    import json
    json_path = os.path.join(seg_dir, f"scene{scene_idx+1:05d}_mapping.json")
    with open(json_path, "w") as f:
        json.dump(all_scene_classes, f, indent=2, ensure_ascii=False)

    print(f"  → {saved_count}개 이미지 세트 저장 (rgb/depth/seg), 매핑 {len(all_scene_classes)}개 오브젝트")
    return output_base


try:
    # 타겟이 지정된 경우: 유사도 기반 순차 스폰
    if args.target:
        # 타겟 이름 검증 (폴더 이름 또는 USD 이름)
        resolved_target = object_spawner.resolve_target_name(args.target)
        target_found = resolved_target in object_spawner._spawned_names

        if not target_found:
            print(f"\n⚠️  경고: '{args.target}'을(를) 찾을 수 없습니다!")
            print(f"   위 목록에서 폴더 이름(예: book_1)을 사용하세요.")
            exit(1)

        print(f"\n[유사도 기반 스폰] 타겟: {args.target} → {resolved_target}")
        print(f"생성할 씬 개수: {args.num_scenes}\n")


        for scene_idx in range(args.num_scenes):
            print(f"--- Scene {scene_idx + 1}/{args.num_scenes} ---")

            # 오브젝트 초기화 (대기 위치로)
            object_spawner.initialize()
            for _ in range(30):
                world.step(render=True)

            # 유사도 기반 순차 스폰
            object_spawner.spawn_with_similarity(
                target_name=args.target,
                world=world,
                stabilization_steps=STABILIZATION_STEPS,
                final_stabilization_steps=FINAL_STABILIZATION_STEPS,
                target_not_bottom_prob=TARGET_NOT_BOTTOM_PROB,
                target_top_prob=TARGET_TOP_PROB,
            )

            # 이미지 캡처 및 저장
            save_scene_images(scene_idx, args.target)

            print(f"Scene {scene_idx + 1} 완료!")

            # 잠시 대기 (결과 확인용)
            for _ in range(30):
                world.step(render=True)

        print("\n모든 씬 생성 완료!")

    # 타겟 미지정: 기존 랜덤 스폰 방식
    else:
        print("⚠️  --target 옵션이 지정되지 않았습니다. 랜덤 스폰 모드로 실행합니다.")
        print("   유사도 기반 스폰을 원하면: --target <오브젝트이름>\n")

        object_spawner.spawn(randomize=True)

        count = 0
        while simulation_app.is_running():
            if count % 200 == 0 and count > 0:
                object_spawner.spawn()
            world.step(render=True)
            count += 1

finally:
    world.stop()
    simulation_app.close()
