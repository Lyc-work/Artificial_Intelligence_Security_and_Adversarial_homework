import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def parse_task_name(filename: str):
    """
    Example:
        id0_id1_0000_00060.png

    target_id = id0
    source_id = id1
    video_idx = 0000
    frame_idx = 00060
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Bad filename format: {filename}")
    return parts[0], parts[1], parts[2], parts[3]


def load_image_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize(x, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def l2_normalize(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def cosine_sim(a: np.ndarray, b: np.ndarray):
    if a is None or b is None:
        return None
    a = l2_normalize(a)
    b = l2_normalize(b)
    return float(np.dot(a, b))


def try_init_insightface(providers):
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=providers
        )
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("[INFO] Using InsightFace backend.")
        return app
    except Exception as e:
        print(f"[WARN] InsightFace init failed: {e}")
        print("[WARN] This script needs InsightFace for identity score.")
        return None


def clamp_bbox(bbox, w, h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def blur_score(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_contrast(face_crop):
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def brightness_quality(mean_brightness):
    # 亮度越接近 128 越好
    return 1.0 - min(1.0, abs(mean_brightness - 128.0) / 128.0)


def eye_roll_angle(kps):
    """
    InsightFace 5 points:
        0 left eye
        1 right eye
        2 nose
        3 left mouth
        4 right mouth
    """
    try:
        left_eye = kps[0]
        right_eye = kps[1]
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        return math.degrees(math.atan2(dy, dx))
    except Exception:
        return 0.0


def extract_face_features(img_path: Path, app):
    img = cv2.imread(str(img_path))
    if img is None:
        return {
            "status": "read_failed",
            "path": str(img_path),
        }

    h, w = img.shape[:2]

    faces = app.get(img)
    if len(faces) == 0:
        return {
            "status": "no_face",
            "path": str(img_path),
            "img_w": w,
            "img_h": h,
            "n_faces": 0,
        }

    # 取最大脸
    def face_area(face):
        x1, y1, x2, y2 = face.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    faces = sorted(faces, key=face_area, reverse=True)
    face = faces[0]

    x1, y1, x2, y2 = clamp_bbox(face.bbox, w, h)
    fw = max(1, x2 - x1)
    fh = max(1, y2 - y1)

    face_crop = img[y1:y2, x1:x2]
    area_ratio = (fw * fh) / float(w * h)

    sharpness = blur_score(face_crop)
    bright, contrast = brightness_contrast(face_crop)

    det_score = float(getattr(face, "det_score", 0.5))
    kps = getattr(face, "kps", None)

    if kps is not None:
        kps = np.array(kps).astype(np.float32)

    # embedding
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        embedding = getattr(face, "embedding", None)
        if embedding is not None:
            embedding = l2_normalize(np.array(embedding).astype(np.float32))
    else:
        embedding = np.array(embedding).astype(np.float32)

    # 质量分数
    area_norm = normalize(area_ratio, 0.04, 0.40)
    sharp_norm = normalize(math.log10(sharpness + 1.0), 1.0, 3.2)
    det_norm = normalize(det_score, 0.4, 1.0)
    bright_q = brightness_quality(bright)

    face_center_x = (x1 + x2) / 2.0
    face_center_y = (y1 + y2) / 2.0
    img_center_x = w / 2.0
    img_center_y = h / 2.0
    center_dist = math.sqrt((face_center_x - img_center_x) ** 2 + (face_center_y - img_center_y) ** 2)
    max_dist = math.sqrt(img_center_x ** 2 + img_center_y ** 2)
    center_score = 1.0 - min(1.0, center_dist / max_dist)

    single_face_bonus = 1.0 if len(faces) == 1 else max(0.5, 1.0 - 0.15 * (len(faces) - 1))

    quality = (
        0.30 * area_norm +
        0.30 * sharp_norm +
        0.20 * det_norm +
        0.10 * bright_q +
        0.10 * center_score
    ) * single_face_bonus

    # 表情/姿态 proxy
    # 注意：InsightFace 只有 5 点关键点，所以这里是近似表情特征，不是精确表情识别。
    if kps is not None:
        left_eye = kps[0]
        right_eye = kps[1]
        nose = kps[2]
        left_mouth = kps[3]
        right_mouth = kps[4]

        eye_dist = float(np.linalg.norm(left_eye - right_eye)) + 1e-6
        mouth_width = float(np.linalg.norm(left_mouth - right_mouth)) / eye_dist

        mouth_center = (left_mouth + right_mouth) / 2.0
        mouth_y_rel = float((mouth_center[1] - y1) / fh)
        mouth_x_rel = float((mouth_center[0] - x1) / fw)

        nose_x_rel = float((nose[0] - x1) / fw)
        nose_y_rel = float((nose[1] - y1) / fh)

        yaw_proxy = float(nose_x_rel - 0.5)
        roll = eye_roll_angle(kps)
    else:
        mouth_width = 0.0
        mouth_y_rel = 0.0
        mouth_x_rel = 0.0
        nose_x_rel = 0.5
        nose_y_rel = 0.5
        yaw_proxy = 0.0
        roll = 0.0

    return {
        "status": "ok",
        "path": str(img_path),
        "img_w": w,
        "img_h": h,
        "n_faces": len(faces),
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "area_ratio": area_ratio,
        "sharpness": sharpness,
        "brightness": bright,
        "contrast": contrast,
        "det_score": det_score,
        "area_norm": area_norm,
        "sharp_norm": sharp_norm,
        "det_norm": det_norm,
        "brightness_quality": bright_q,
        "center_score": center_score,
        "single_face_bonus": single_face_bonus,
        "quality_score": quality,
        "embedding": embedding,
        "mouth_width": mouth_width,
        "mouth_y_rel": mouth_y_rel,
        "mouth_x_rel": mouth_x_rel,
        "nose_x_rel": nose_x_rel,
        "nose_y_rel": nose_y_rel,
        "yaw_proxy": yaw_proxy,
        "roll": roll,
    }


def expression_similarity(src_feat, tgt_feat):
    """
    使用 5 点关键点构造的近似表情相似度：
    - mouth_width：嘴角宽度/眼距，粗略表示笑/嘴部张力
    - mouth_y_rel：嘴部在脸框中的相对高度
    - mouth_x_rel：嘴部中心横向位置
    """
    diffs = [
        abs(src_feat["mouth_width"] - tgt_feat["mouth_width"]) / 0.45,
        abs(src_feat["mouth_y_rel"] - tgt_feat["mouth_y_rel"]) / 0.20,
        abs(src_feat["mouth_x_rel"] - tgt_feat["mouth_x_rel"]) / 0.20,
    ]
    d = math.sqrt(sum(x * x for x in diffs) / len(diffs))
    return max(0.0, min(1.0, 1.0 - d))


def pose_similarity(src_feat, tgt_feat):
    """
    粗略姿态相似：
    - yaw_proxy：鼻子相对脸框中心的左右偏移
    - roll：眼睛连线角度
    """
    yaw_diff = abs(src_feat["yaw_proxy"] - tgt_feat["yaw_proxy"]) / 0.35
    roll_diff = abs(src_feat["roll"] - tgt_feat["roll"]) / 30.0

    d = 0.65 * yaw_diff + 0.35 * roll_diff
    return max(0.0, min(1.0, 1.0 - d))


def lighting_similarity(src_feat, tgt_feat):
    bright_diff = abs(src_feat["brightness"] - tgt_feat["brightness"]) / 100.0
    contrast_diff = abs(src_feat["contrast"] - tgt_feat["contrast"]) / 80.0

    d = 0.65 * bright_diff + 0.35 * contrast_diff
    return max(0.0, min(1.0, 1.0 - d))


def identity_center(candidate_feats: List[dict]):
    embs = []
    for feat in candidate_feats:
        if feat.get("status") == "ok" and feat.get("embedding") is not None:
            embs.append(feat["embedding"])

    if not embs:
        return None

    center = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(center)


def identity_score(src_feat, center):
    if center is None or src_feat.get("embedding") is None:
        return 0.5

    cos = cosine_sim(src_feat["embedding"], center)
    if cos is None:
        return 0.5

    # cos [-1,1] -> [0,1]
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def clean_for_csv(row: dict):
    new_row = {}
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, float):
            new_row[k] = round(v, 6)
        else:
            new_row[k] = v
    return new_row


def write_csv(rows: List[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("work/target_frames"))
    parser.add_argument("--source-candidates-dir", type=Path, default=Path("work/source_candidates"))
    parser.add_argument("--selected-source-dir", type=Path, default=Path("work/selected_sources_per_target"))
    parser.add_argument("--log-dir", type=Path, default=Path("work/logs/source_per_target_selection"))

    parser.add_argument("--providers", nargs="+", default=["CUDAExecutionProvider", "CPUExecutionProvider"])

    parser.add_argument("--w-quality", type=float, default=0.35)
    parser.add_argument("--w-identity", type=float, default=0.25)
    parser.add_argument("--w-expression", type=float, default=0.20)
    parser.add_argument("--w-pose", type=float, default=0.15)
    parser.add_argument("--w-lighting", type=float, default=0.05)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-all-scores", action="store_true")

    args = parser.parse_args()

    image_list_path = args.image_list.resolve()
    target_dir = args.target_dir.resolve()
    source_candidates_dir = args.source_candidates_dir.resolve()
    selected_source_dir = args.selected_source_dir.resolve()
    log_dir = args.log_dir.resolve()

    selected_source_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    app = try_init_insightface(args.providers)
    if app is None:
        raise RuntimeError("InsightFace is required for this target-specific weighted selector.")

    task_names = load_image_list(image_list_path)

    # 找出所有 source_id
    source_ids = sorted(set(parse_task_name(name)[1] for name in task_names))

    print(f"[INFO] Loaded tasks: {len(task_names)}")
    print(f"[INFO] Source ids: {len(source_ids)}")

    # 预计算所有 source candidate 的特征
    source_feats_by_id: Dict[str, List[dict]] = {}
    centers_by_id: Dict[str, Optional[np.ndarray]] = {}

    for sid in source_ids:
        cand_dir = source_candidates_dir / sid
        candidate_paths = sorted(cand_dir.glob("*.png"))

        feats = []
        for p in candidate_paths:
            feat = extract_face_features(p, app)
            feat["source_id"] = sid
            feat["candidate_name"] = p.name
            feats.append(feat)

        source_feats_by_id[sid] = feats
        centers_by_id[sid] = identity_center(feats)

        valid_count = sum(1 for f in feats if f.get("status") == "ok")
        print(f"[SOURCE] {sid}: candidates={len(feats)}, valid={valid_count}")

    selected_rows = []
    all_score_rows = []
    failed_rows = []

    for idx, name in enumerate(task_names, start=1):
        print(f"[{idx}/{len(task_names)}] {name}")

        try:
            target_id, source_id, video_idx, frame_idx = parse_task_name(name)
        except Exception as e:
            failed_rows.append({
                "output_name": name,
                "reason": f"parse_failed: {e}",
            })
            continue

        target_path = target_dir / name
        if not target_path.exists():
            failed_rows.append({
                "output_name": name,
                "source_id": source_id,
                "reason": f"target_missing: {target_path}",
            })
            continue

        target_feat = extract_face_features(target_path, app)
        if target_feat.get("status") != "ok":
            failed_rows.append({
                "output_name": name,
                "source_id": source_id,
                "reason": f"target_face_invalid: {target_feat.get('status')}",
            })
            continue

        candidates = source_feats_by_id.get(source_id, [])
        valid_candidates = [f for f in candidates if f.get("status") == "ok"]

        if not valid_candidates:
            failed_rows.append({
                "output_name": name,
                "source_id": source_id,
                "reason": "no_valid_source_candidate",
            })
            continue

        center = centers_by_id.get(source_id)

        scored = []
        for src_feat in valid_candidates:
            q = float(src_feat["quality_score"])
            ids = identity_score(src_feat, center)
            expr = expression_similarity(src_feat, target_feat)
            pose = pose_similarity(src_feat, target_feat)
            light = lighting_similarity(src_feat, target_feat)

            total = (
                args.w_quality * q +
                args.w_identity * ids +
                args.w_expression * expr +
                args.w_pose * pose +
                args.w_lighting * light
            )

            row = {
                "output_name": name,
                "target_path": str(target_path),
                "target_id": target_id,
                "source_id": source_id,
                "video_idx": video_idx,
                "frame_idx": frame_idx,
                "candidate_path": src_feat["path"],
                "candidate_name": src_feat["candidate_name"],
                "total_score": total,
                "quality_score": q,
                "identity_score": ids,
                "expression_similarity": expr,
                "pose_similarity": pose,
                "lighting_similarity": light,
                "src_area_ratio": src_feat["area_ratio"],
                "src_sharpness": src_feat["sharpness"],
                "src_brightness": src_feat["brightness"],
                "src_contrast": src_feat["contrast"],
                "src_det_score": src_feat["det_score"],
                "src_n_faces": src_feat["n_faces"],
                "src_mouth_width": src_feat["mouth_width"],
                "src_yaw_proxy": src_feat["yaw_proxy"],
                "src_roll": src_feat["roll"],
                "target_mouth_width": target_feat["mouth_width"],
                "target_yaw_proxy": target_feat["yaw_proxy"],
                "target_roll": target_feat["roll"],
            }

            scored.append(row)

        scored = sorted(scored, key=lambda r: r["total_score"], reverse=True)
        best = scored[0]

        src_path = Path(best["candidate_path"])
        selected_copy_path = selected_source_dir / name

        if selected_copy_path.exists() and not args.overwrite:
            pass
        else:
            shutil.copy2(src_path, selected_copy_path)

        best["selected_source_path"] = str(src_path)
        best["selected_copy_path"] = str(selected_copy_path)

        selected_rows.append(clean_for_csv(best))

        if args.save_all_scores:
            for r in scored:
                all_score_rows.append(clean_for_csv(r))

    source_map_path = log_dir / "selected_source_map.csv"
    failed_path = log_dir / "failed_selection.csv"
    all_scores_path = log_dir / "all_task_candidate_scores.csv"

    write_csv(selected_rows, source_map_path)
    write_csv(failed_rows, failed_path)

    if args.save_all_scores:
        write_csv(all_score_rows, all_scores_path)

    print("\n" + "=" * 90)
    print("[DONE] Per-target source selection finished.")
    print(f"[OUT] selected source copies : {selected_source_dir}")
    print(f"[LOG] source map             : {source_map_path}")
    print(f"[LOG] failed selection       : {failed_path}")
    if args.save_all_scores:
        print(f"[LOG] all scores             : {all_scores_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=2 python scripts/select_source_per_target.py --image-list data/image_list.txt --target-dir work/target_frames --source-candidates-dir work/source_candidates --selected-source-dir work/selected_sources_per_target --log-dir work/logs/source_per_target_selection --w-quality 0.45 --w-identity 0.25 --w-expression 0.15 --w-pose 0.10 --w-lighting 0.05 --overwrite