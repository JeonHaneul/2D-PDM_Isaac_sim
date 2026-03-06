"""ObjectSpawner — reusable class for spawning USD objects onto a workspace."""

from __future__ import annotations

import os
import random
import math
from typing import Optional
from collections import defaultdict

import torch

from isaacsim.core.prims import SingleRigidPrim, SingleXFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrimView, XFormPrimView


# 카테고리 간 유사도 맵 (소문자로 통일)
SIMILARITY_MAP = {
    "book": {"book": 0.8, "toy": 0.5, "fruit": 0.2, "packaged_food": 0.2},
    "toy": {"book": 0.5, "toy": 0.8, "fruit": 0.2, "packaged_food": 0.2},
    "fruit": {"book": 0.2, "toy": 0.2, "fruit": 0.8, "packaged_food": 0.5},
    "packaged_food": {"book": 0.2, "toy": 0.2, "fruit": 0.5, "packaged_food": 0.8},
}

# 유사도 점수에 따른 스폰 반경 (높은 유사도 = 가까이, 낮은 유사도 = 멀리)
SCORE_TO_RADIUS = {0.8: 0.05, 0.5: 0.10, 0.2: 0.15}


class ObjectSpawner:
    """Scans one or more USD category directories and spawns objects as
    rigid-body children of a single XForm container.

    Lifecycle
    ---------
    1. ``__init__``  – discovers assets across all *categories*, creates
       the XForm container, and loads all prims at a **default position
       outside** the workspace (staging area).
    2. ``spawn()``   – moves the prims **into** the workspace bounds
       (random or ordered placement).
    3. ``initialize()`` – moves all prims **back** to the default
       position (outside the workspace).  No prims are removed.

    Parameters
    ----------
    world : isaacsim.core.api.World
        The simulation world instance.
    categories : str | list[str]
        One or more category names.  Each must be a subdirectory of
        *usd_folder_dir* (e.g. ``["Food", "Toy"]``).
    usd_folder_dir : str
        Parent directory that contains per-category subdirectories.
    container_prim_path : str
        Stage path for the XForm container
        (e.g. ``"/World/Objects_0"``).
    workspace_bounds : dict
        Spawn area on the workspace surface.
        ``{"x": (min, max), "y": (min, max), "z_surface": float}``
    default_position : torch.Tensor | None
        Position outside the workspace where prims are parked
        when not in use.  Defaults to ``[10, 10, -1]``.
    num_to_spawn : int | None
        How many objects to spawn.  ``None`` → spawn all found assets.
    extensions : tuple[str, ...]
        File extensions to treat as valid USD assets.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        world,
        categories: str | list[str],
        usd_folder_dir: str,
        container_prim_path: str,
        workspace_bounds: dict,
        default_position: Optional[torch.Tensor] = None,
        num_to_spawn: Optional[int] = None,
        extensions: tuple[str, ...] = (".usd", ".usdc"),
    ) -> None:
        self._world = world
        self._categories = (
            [categories] if isinstance(categories, str) else list(categories)
        )
        self._usd_folder_dir = usd_folder_dir
        self._container_path = container_prim_path
        self._bounds = workspace_bounds
        self._extensions = extensions
        self._device = self._world.device
        self._default_position = (
            default_position.to(self._device) if default_position is not None
            else torch.tensor([10.0, 10.0, -1.0], device=self._device)
        )

        # Extract the container prefix from the prim path
        # e.g. "/World/Objects_0" → "Objects"
        self._container_prefix = (
            container_prim_path.rsplit("/", 1)[-1].rsplit("_", 1)[0]
        )

        # Discover assets across all categories
        # objects_dir:   {"cracker_box": "/…/Food/cracker_box.usd", …}
        # objects_class:  {"cracker_box": "Food", "teddy_bear": "Toy", …}
        # folder_to_usd:  {"book_1": "Book_02", "fruit_1": "Apple", …}  폴더이름 → USD이름 매핑
        self._objects_dir: dict[str, str] = {}
        self._objects_class: dict[str, str] = {}
        self._folder_to_usd: dict[str, str] = {}  # 폴더 이름 → USD 파일 이름 매핑

        for category in self._categories:
            usd_dir = os.path.join(self._usd_folder_dir, category)
            if not os.path.isdir(usd_dir):
                raise FileNotFoundError(
                    f"Category directory not found: {usd_dir}"
                )
            # 서브폴더 내의 USD 파일 검색 (예: Book/book_1/Book_02.usd)
            for subdir in os.listdir(usd_dir):
                subdir_path = os.path.join(usd_dir, subdir)
                if os.path.isdir(subdir_path):
                    for f in os.listdir(subdir_path):
                        if f.endswith(extensions):
                            name = os.path.splitext(f)[0]
                            self._objects_dir[name] = os.path.join(subdir_path, f)
                            self._objects_class[name] = category
                            # 폴더 이름(book_1) → USD 이름(Book_02) 매핑 저장
                            self._folder_to_usd[subdir.lower()] = name

        if not self._objects_dir:
            raise FileNotFoundError(
                f"No USD assets ({extensions}) found in categories "
                f"{self._categories} under {usd_folder_dir}"
            )

        self._num_to_spawn = num_to_spawn or len(self._objects_dir)
        self._spawned_prims: list[SingleRigidPrim] = []
        self._spawned_paths: list[str] = []
        self._spawned_names: list[str] = []  # object names in spawn order

        # Set after setup_cloned_views() is called
        self._num_envs: int = 1
        self._item_views: list[RigidPrimView] = []
        self._container_view: XFormPrimView | None = None

        # Create the container and load all prims at the default position
        self._create_prims()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def setup_cloned_views(self, num_envs: int) -> list[RigidPrimView]:
        """Create per-item RigidPrimViews that span all cloned envs.

        Must be called **after** ``GridCloner.clone()``.

        Parameters
        ----------
        num_envs : int
            Total number of cloned environments.

        Returns
        -------
        list[RigidPrimView]
            One view per item, each of shape (num_envs, …).
        """
        self._num_envs = num_envs
        self._item_views.clear()

        # Container view for world-position lookups
        self._container_view = XFormPrimView(
            f"/World/{self._container_prefix}_*"
        )

        for i in range(self._num_to_spawn):
            pattern = (
                f"/World/{self._container_prefix}_*/object_{i}"
            )
            view = RigidPrimView(pattern)
            self._item_views.append(view)

        return self._item_views

    def spawn(self, randomize: bool = True) -> None:
        """Move items into workspace bounds across all cloned envs.

        Computes **world-space** positions by adding local offsets to
        each container's world position, then uses
        ``RigidPrimView.set_world_poses()`` to properly teleport the
        rigid bodies.

        Parameters
        ----------
        randomize : bool
            If True, each environment gets independent random (x, y).
            If False, evenly-spaced placement (same across all envs).
        """
        container_world_pos, _ = self._container_view.get_world_poses()

        for i, view in enumerate(self._item_views):
            local_offsets = torch.stack([
                self._compute_workspace_position(i, randomize)
                for _ in range(self._num_envs)
            ])
            world_positions = container_world_pos + local_offsets
            view.set_world_poses(world_positions)

    # ------------------------------------------------------------------
    # 유사도 기반 순차 스폰 API (Genesis scene_generator.py 로직 포팅)
    # ------------------------------------------------------------------
    def resolve_target_name(self, name: str) -> str:
        """폴더 이름(book_1)을 USD 파일 이름(Book_02)으로 변환
        
        이미 USD 파일 이름이면 그대로 반환
        """
        name_lower = name.lower()
        
        # 1. 폴더 이름으로 매핑 검색 (book_1 → Book_02)
        if name_lower in self._folder_to_usd:
            return self._folder_to_usd[name_lower]
        
        # 2. 이미 USD 파일 이름인 경우
        for obj_name in self._spawned_names:
            if obj_name.lower() == name_lower:
                return obj_name
        
        # 3. 부분 매칭 시도
        for obj_name in self._spawned_names:
            if name_lower in obj_name.lower() or obj_name.lower() in name_lower:
                return obj_name
        
        return name  # 찾지 못하면 원본 반환
    
    def get_object_index_by_name(self, name: str) -> int:
        """오브젝트 이름으로 인덱스 찾기 (폴더 이름도 지원)"""
        # 먼저 폴더 이름을 USD 이름으로 변환
        resolved_name = self.resolve_target_name(name)
        
        for i, obj_name in enumerate(self._spawned_names):
            if obj_name.lower() == resolved_name.lower():
                return i
        
        # 부분 매칭 시도
        for i, obj_name in enumerate(self._spawned_names):
            if resolved_name.lower() in obj_name.lower() or obj_name.lower() in resolved_name.lower():
                return i
        
        raise ValueError(
            f"Object '{name}' (resolved: '{resolved_name}') not found.\n"
            f"Available objects: {self._spawned_names}\n"
            f"Folder mappings: {self._folder_to_usd}"
        )

    def get_target_candidates(self) -> list[tuple[str, str]]:
        """스폰 가능한 타겟 오브젝트 정보 리스트
        
        Returns
        -------
        list[tuple[str, str]]
            [(폴더이름, USD이름), ...] 형태의 리스트
        """
        result = []
        # 폴더 이름 → USD 이름 매핑 기반으로 리스트 생성
        for folder_name, usd_name in self._folder_to_usd.items():
            result.append((folder_name, usd_name))
        return result
    
    @property
    def folder_to_usd(self) -> dict[str, str]:
        """폴더 이름 → USD 파일 이름 매핑"""
        return dict(self._folder_to_usd)

    def spawn_single_object(self, obj_index: int, position: torch.Tensor, env_indices: list[int] = None) -> None:
        """단일 오브젝트를 특정 위치에 스폰 (특정 환경만 지정 가능)
        
        Parameters
        ----------
        obj_index : int
            오브젝트 인덱스
        position : torch.Tensor
            로컬 위치 (container 기준) - shape: (3,) 또는 (num_envs, 3)
        env_indices : list[int], optional
            스폰할 환경 인덱스 리스트. None이면 모든 환경에 적용
        """
        if obj_index >= len(self._item_views):
            return
        
        view = self._item_views[obj_index]
        container_world_pos, _ = self._container_view.get_world_poses()
        
        if position.dim() == 1:
            # 모든 환경에 동일 위치
            local_offsets = position.unsqueeze(0).expand(self._num_envs, -1)
        else:
            local_offsets = position
        
        world_positions = container_world_pos + local_offsets
        
        if env_indices is not None:
            # 특정 환경만 업데이트
            current_pos, current_orient = view.get_world_poses()
            for env_idx in env_indices:
                current_pos[env_idx] = world_positions[env_idx]
            view.set_world_poses(current_pos)
        else:
            view.set_world_poses(world_positions)

    def spawn_with_similarity(
        self,
        target_name: str,
        world,
        stabilization_steps: int = 60,
        final_stabilization_steps: int = 120,
    ) -> None:
        """유사도 기반으로 타겟 오브젝트 주변에 순차적으로 오브젝트를 스폰
        
        Genesis scene_generator.py의 generate_one_scene 로직을 Isaac Sim용으로 포팅.
        각 환경마다 독립적인 랜덤 scene을 생성합니다.
        
        Parameters
        ----------
        target_name : str
            타겟 오브젝트 이름 (예: "book_1", "fruit_2")
        world : World
            시뮬레이션 월드 (물리 스텝용)
        stabilization_steps : int
            각 오브젝트 투하 후 안정화 스텝 수
        final_stabilization_steps : int
            모든 오브젝트 투하 후 최종 안정화 스텝 수
        """
        # 1. 타겟 오브젝트 찾기
        target_idx = self.get_object_index_by_name(target_name)
        target_category = self._objects_class[self._spawned_names[target_idx]].lower()
        
        container_world_pos, _ = self._container_view.get_world_poses()
        
        # 2. 각 환경별로 타겟 위치 생성 (랜덤)
        x_min, x_max = self._bounds["x"]
        y_min, y_max = self._bounds["y"]
        z_drop = self._bounds.get("z_drop", self._bounds["z_surface"] + 0.15)
        
        target_positions = torch.zeros(self._num_envs, 3, device=self._device)
        for env_idx in range(self._num_envs):
            target_positions[env_idx] = torch.tensor([
                random.uniform(x_min * 0.5, x_max * 0.5),  # 중앙 부근
                random.uniform(y_min * 0.5, y_max * 0.5),
                z_drop
            ], device=self._device)
        
        # 3. 타겟 오브젝트 투하 및 안정화
        world_positions = container_world_pos + target_positions
        self._item_views[target_idx].set_world_poses(world_positions)
        
        for _ in range(stabilization_steps):
            world.step(render=True)
        
        # 4. 각 오브젝트의 유사도 점수 계산 (스폰 반경 결정용)
        obj_scores: dict[int, float] = {}
        for i, obj_name in enumerate(self._spawned_names):
            if i == target_idx:
                continue
            obj_category = self._objects_class[obj_name].lower()
            score = SIMILARITY_MAP.get(target_category, {}).get(obj_category, 0.2)
            obj_scores[i] = score
        
        # 5. 각 환경별로 완전히 독립적인 랜덤 스폰 순서 생성
        # 유사도 순서 무시, 완전 랜덤 (단, 스폰 반경은 각 물체의 유사도에 따라 적용)
        all_obj_indices = list(obj_scores.keys())
        
        env_spawn_orders: list[list[tuple[int, float]]] = []  # [(obj_idx, score), ...]
        for env_idx in range(self._num_envs):
            shuffled = all_obj_indices.copy()
            random.shuffle(shuffled)  # 완전 랜덤 순서
            env_order = [(obj_idx, obj_scores[obj_idx]) for obj_idx in shuffled]
            env_spawn_orders.append(env_order)
        
        # 6. 최대 스폰 횟수 (모든 환경에서 가장 긴 리스트 기준)
        max_spawn_count = max(len(order) for order in env_spawn_orders)
        
        # 7. 순차적으로 각 환경에서 물체 투하 (환경마다 다른 순서)
        for spawn_step in range(max_spawn_count):
            # 각 환경별로 이번 스텝에 떨어뜨릴 물체 결정
            for env_idx in range(self._num_envs):
                if spawn_step >= len(env_spawn_orders[env_idx]):
                    continue
                
                obj_idx, score = env_spawn_orders[env_idx][spawn_step]
                radius = SCORE_TO_RADIUS.get(score, 0.15)
                
                # 현재 타겟 위치 기준으로 스폰 위치 계산
                cur_target_pos, _ = self._item_views[target_idx].get_world_poses()
                local_target = cur_target_pos[env_idx] - container_world_pos[env_idx]
                
                # 원형 분포로 스폰 위치 계산
                r = radius * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                
                spawn_x = torch.clamp(
                    local_target[0] + r * math.cos(theta),
                    torch.tensor(x_min, device=self._device),
                    torch.tensor(x_max, device=self._device)
                )
                spawn_y = torch.clamp(
                    local_target[1] + r * math.sin(theta),
                    torch.tensor(y_min, device=self._device),
                    torch.tensor(y_max, device=self._device)
                )
                spawn_z = z_drop + random.uniform(-0.05, 0.1)
                
                # 해당 환경에서만 물체 이동
                current_pos, current_orient = self._item_views[obj_idx].get_world_poses()
                new_pos = container_world_pos[env_idx] + torch.tensor([spawn_x, spawn_y, spawn_z], device=self._device)
                current_pos[env_idx] = new_pos
                self._item_views[obj_idx].set_world_poses(current_pos, current_orient)
            
            # 물리 안정화 (모든 환경 동시에)
            for _ in range(stabilization_steps):
                world.step(render=True)
        
        # 8. 최종 안정화
        for _ in range(final_stabilization_steps):
            world.step(render=True)

    def initialize(self) -> None:
        """Move all items back to their individual waiting positions (outside the
        workspace) across all cloned environments."""
        container_world_pos, _ = self._container_view.get_world_poses()
        
        for i, view in enumerate(self._item_views):
            # 각 물체마다 다른 대기 위치 사용
            wait_pos = self._wait_positions[i].unsqueeze(0).expand(self._num_envs, -1)
            world_positions = container_world_pos + wait_pos
            view.set_world_poses(world_positions)

    def get_prim_paths(self) -> list[str]:
        """Return prim paths of currently managed objects."""
        return list(self._spawned_paths)

    @property
    def container_path(self) -> str:
        return self._container_path

    @property
    def num_spawned(self) -> int:
        return len(self._spawned_prims)

    @property
    def default_position(self) -> torch.Tensor:
        return self._default_position

    @property
    def objects_dir(self) -> dict[str, str]:
        """Mapping of object name → USD file path."""
        return dict(self._objects_dir)

    @property
    def objects_class(self) -> dict[str, str]:
        """Mapping of object name → category name."""
        return dict(self._objects_class)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_prims(self) -> None:
        """Create the XForm container and load all asset prims at the
        default (outside-workspace) position.  Called once in __init__."""
        stage = self._world.scene.stage

        # Create or recreate the container XForm
        if stage.GetPrimAtPath(self._container_path):
            stage.RemovePrim(self._container_path)
        SingleXFormPrim(
            prim_path=self._container_path,
            name=f"{self._container_prefix.lower()}_container",
        )

        # Load assets (up to num_to_spawn)
        asset_items = list(self._objects_dir.items())[: self._num_to_spawn]
        
        # 대기 위치 설정: 8개씩 2줄로 배치 (워크스페이스 밖)
        # 배치: x=0.8 (워크스페이스 밖), y는 물체마다 다르게 (좌우 나열), z는 고정
        ITEMS_PER_ROW = 8
        ROW_SPACING = 0.2  # 줄 간격 (x)
        ITEM_SPACING = 0.15  # 물체 간격 (y, 좌우)
        WAIT_X = 0.8  # 대기 x 위치 (워크스페이스 밖)
        WAIT_Z = 0.1  # 대기 z 높이 (고정)
        
        self._wait_positions: list[torch.Tensor] = []  # 각 물체의 대기 위치 저장

        for i, (name, asset_path) in enumerate(asset_items):
            prim_path = f"{self._container_path}/object_{i}"
            
            # 대기 위치 계산: 8개씩 2줄로 배치 (Y축으로 좌우 나열)
            row = i // ITEMS_PER_ROW  # 0, 1, 2, ... (줄 번호)
            col = i % ITEMS_PER_ROW   # 0~7 (열 번호)
            wait_pos = torch.tensor([
                WAIT_X + row * ROW_SPACING,  # X: 줄마다 뒤로 배치
                -0.5 + col * ITEM_SPACING,   # Y: 좌우로 나열 (-0.5 ~ 0.55)
                WAIT_Z  # Z: 고정 높이
            ], device=self._device)
            self._wait_positions.append(wait_pos)

            add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)

            prim = SingleRigidPrim(
                prim_path=prim_path,
                name=f"{self._container_prefix}_{name}",
                position=wait_pos,
            )
            prim.set_default_state(position=wait_pos)
            self._world.scene.add(prim)

            self._spawned_prims.append(prim)
            self._spawned_paths.append(prim_path)
            self._spawned_names.append(name)

    def _compute_workspace_position(
        self, index: int, randomize: bool
    ) -> torch.Tensor:
        """Compute a position on the workspace surface."""
        x_min, x_max = self._bounds["x"]
        y_min, y_max = self._bounds["y"]
        z = self._bounds["z_surface"]

        if randomize:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
        else:
            t = index / max(self._num_to_spawn - 1, 1)
            x = x_min + t * (x_max - x_min)
            y = (y_min + y_max) / 2.0

        return torch.tensor([x, y, z], device=self._device)
