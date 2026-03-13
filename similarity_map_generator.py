import os
import argparse
import numpy as np
import cv2
import json

# ==============================================================================
# 설정
# ==============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR    = os.path.join(BASE_DIR, "asset")
USD_FILE_DIR = os.path.join(ASSET_DIR, "260303")

SIMILARITY_MAP = {
    "book":          {"book": 0.8, "toy": 0.5, "fruit": 0.2, "packaged_food": 0.2},
    "toy":           {"book": 0.5, "toy": 0.8, "fruit": 0.2, "packaged_food": 0.2},
    "fruit":         {"book": 0.2, "toy": 0.2, "fruit": 0.8, "packaged_food": 0.5},
    "packaged_food": {"book": 0.2, "toy": 0.2, "fruit": 0.5, "packaged_food": 0.8},
}

# ==============================================================================
# Asset 탐색 (Isaac Sim 없이 순수 Python)
# ==============================================================================
def discover_assets(usd_folder_dir, extensions=(".usd", ".usdc")):
    """
    Returns:
        folder_to_info: {folder_name_lower: (usd_name, category)}
        usd_to_category: {usd_name: category}
    """
    folder_to_info  = {}
    usd_to_category = {}
    if not os.path.isdir(usd_folder_dir):
        return folder_to_info, usd_to_category
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
                    folder_to_info[subdir.lower()]  = (usd_name, category)
                    usd_to_category[usd_name]        = category
                    break
    return folder_to_info, usd_to_category


# ==============================================================================
# 메인
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Similarity Map Generator")
    parser.add_argument("--target_name", type=str, required=True,
                        help="타겟 오브젝트 폴더 이름 (예: book_1)")
    args, _ = parser.parse_known_args()

    # --- asset 탐색 ---
    folder_to_info, usd_to_category = discover_assets(USD_FILE_DIR)

    target_key = args.target_name.lower()
    if target_key not in folder_to_info:
        raise ValueError(f"'{args.target_name}'을(를) asset 폴더에서 찾을 수 없습니다.")

    target_usd_name, target_category_raw = folder_to_info[target_key]
    target_category = target_category_raw.lower()          # e.g. "book"
    target_sim = SIMILARITY_MAP.get(target_category, {})  # {category: score}

    print(f"[타겟] {args.target_name} → usd={target_usd_name}, category={target_category}")

    # --- 경로 ---
    scene_seg_dir = os.path.join(BASE_DIR, "output", args.target_name, "scene", "seg")
    output_dir    = os.path.join(BASE_DIR, "output", args.target_name, "similarity_map")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(scene_seg_dir):
        raise FileNotFoundError(f"scene seg 폴더가 없습니다: {scene_seg_dir}")

    # --- 파일 목록: PNG만 (JSON 제외) ---
    seg_files = sorted(
        f for f in os.listdir(scene_seg_dir)
        if f.endswith(".png")
    )

    # --- 모든 mapping JSON을 합쳐서 완전한 color→value 룩업 구성 ---
    # vectorized_scene.py는 color를 cv2.COLOR_HSV2BGR로 생성 → 실제로 BGR 값을 저장
    # 따라서 "color_rgb" 필드는 사실 BGR 값임 → cv2.imread(BGR)와 직접 비교 가능
    combined_color_to_value: dict[tuple, int] = {}

    json_files = [f for f in os.listdir(scene_seg_dir) if f.endswith("_mapping.json")]
    for jf in json_files:
        with open(os.path.join(scene_seg_dir, jf)) as f:
            mapping = json.load(f)
        for usd_name, info in mapping.items():
            bgr = tuple(info["color_rgb"])  # 실제로 BGR (vectorized_scene.py의 저장 버그)
            if bgr in combined_color_to_value:
                continue  # 이미 등록됨
            if usd_name == target_usd_name:
                value = 255
            else:
                obj_cat = usd_to_category.get(usd_name, "").lower()
                score = target_sim.get(obj_cat, 0.0) if obj_cat else 0.0
                value = int(255 * score)
            combined_color_to_value[bgr] = value

    print(f"전체 룩업 항목: {len(combined_color_to_value)}개 색상")

    total = len(seg_files)
    print(f"처리할 이미지: {total}장\n")

    for idx, fname in enumerate(seg_files):
        # 이미지 로드 (BGR 그대로 사용 - mapping도 BGR이므로 직접 비교)
        img_bgr = cv2.imread(os.path.join(scene_seg_dir, fname))
        if img_bgr is None:
            print(f"  [WARN] 이미지 로드 실패: {fname}")
            continue

        # similarity map 생성 (grayscale)
        sim_map = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        for bgr_tuple, value in combined_color_to_value.items():
            b, g, r = bgr_tuple
            mask = (
                (img_bgr[:, :, 0] == b) &
                (img_bgr[:, :, 1] == g) &
                (img_bgr[:, :, 2] == r)
            )
            sim_map[mask] = value

        # 저장
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, sim_map)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            print(f"  [{idx + 1:5d}/{total}] {fname} → 저장 완료")

    print(f"\n=== 완료 ===")
    print(f"  저장 위치: {output_dir}")


if __name__ == "__main__":
    main()
