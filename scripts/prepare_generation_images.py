import argparse
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set

import cv2
from tqdm import tqdm


@dataclass
class Task:
    output_name: str
    target_id: str
    source_id: str
    video_idx: str
    frame_idx: int
    target_video: str


def parse_task(filename: str) -> Task:
    """
    Example:
        id0_id1_0000_00060.png

    Meaning:
        target video: id0_0000.mp4
        source id: id1
        frame index: 60
    """
    filename = filename.strip()
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()

    if suffix != ".png":
        raise ValueError(f"Only .png is expected, got: {filename}")

    parts = stem.split("_")
    if len(parts) != 4:
        raise ValueError(f"Bad filename format: {filename}")

    target_id, source_id, video_idx, frame_idx = parts

    return Task(
        output_name=filename,
        target_id=target_id,
        source_id=source_id,
        video_idx=video_idx,
        frame_idx=int(frame_idx),
        target_video=f"{target_id}_{video_idx}.mp4",
    )


def load_tasks(image_list_path: Path) -> List[Task]:
    tasks = []
    with image_list_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                tasks.append(parse_task(name))
    return tasks


def extract_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    return frame


def save_image(image, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(save_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save image: {save_path}")


def extract_target_frames(
    tasks: List[Task],
    videos_dir: Path,
    target_frames_dir: Path,
    overwrite: bool = False,
):
    """
    Extract target frames for face swapping.

    Input:
        videos/id0_0000.mp4

    Output:
        work/target_frames/id0_id1_0000_00060.png
    """
    records = []

    for task in tqdm(tasks, desc="Extract target frames"):
        video_path = videos_dir / task.target_video
        save_path = target_frames_dir / task.output_name

        record = {
            **asdict(task),
            "type": "target",
            "video_path": str(video_path),
            "save_path": str(save_path),
            "status": "ok",
            "message": "",
            "width": None,
            "height": None,
        }

        try:
            if save_path.exists() and not overwrite:
                img = cv2.imread(str(save_path))
                if img is not None:
                    h, w = img.shape[:2]
                    record["width"] = w
                    record["height"] = h
                    record["status"] = "skip_exists"
                    records.append(record)
                    continue

            frame = extract_frame(video_path, task.frame_idx)
            h, w = frame.shape[:2]
            save_image(frame, save_path)

            record["width"] = w
            record["height"] = h

        except Exception as e:
            record["status"] = "failed"
            record["message"] = str(e)

        records.append(record)

    return records


def find_source_videos(videos_dir: Path, source_id: str) -> List[Path]:
    """
    Find videos that belong to a source identity.

    Example:
        source_id = id1
        match:
            id1_0000.mp4
            id1_0001.mp4
            id1_xxxx.mp4
    """
    patterns = [
        f"{source_id}_*.mp4",
        f"{source_id}_*.avi",
        f"{source_id}_*.mov",
    ]

    results = []
    for pattern in patterns:
        results.extend(sorted(videos_dir.glob(pattern)))

    return results


def extract_source_candidates(
    tasks: List[Task],
    videos_dir: Path,
    source_candidates_dir: Path,
    frames_per_source: int = 8,
    sample_stride: int = 30,
    overwrite: bool = False,
):
    """
    Extract source candidate images for each source identity.

    Output:
        work/source_candidates/id1/id1_0000_f00000.png
        work/source_candidates/id1/id1_0000_f00030.png
        ...

    Then you manually choose the best one and copy it to:
        source_faces/id1.png
    """
    source_ids: Set[str] = set(task.source_id for task in tasks)
    records = []

    for source_id in tqdm(sorted(source_ids), desc="Extract source candidates"):
        source_videos = find_source_videos(videos_dir, source_id)

        if not source_videos:
            records.append({
                "source_id": source_id,
                "video_path": "",
                "save_path": "",
                "status": "failed",
                "message": f"No video found for source_id={source_id}",
            })
            continue

        saved_count = 0

        for video_path in source_videos:
            if saved_count >= frames_per_source:
                break

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                records.append({
                    "source_id": source_id,
                    "video_path": str(video_path),
                    "save_path": "",
                    "status": "failed",
                    "message": "Cannot open video",
                })
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_stem = video_path.stem

            # Sample frames: 0, stride, 2*stride...
            frame_indices = list(range(0, total_frames, sample_stride))

            for frame_idx in frame_indices:
                if saved_count >= frames_per_source:
                    break

                save_path = (
                    source_candidates_dir
                    / source_id
                    / f"{video_stem}_f{frame_idx:05d}.png"
                )

                if save_path.exists() and not overwrite:
                    saved_count += 1
                    records.append({
                        "source_id": source_id,
                        "video_path": str(video_path),
                        "save_path": str(save_path),
                        "status": "skip_exists",
                        "message": "",
                    })
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()

                if not ok or frame is None:
                    records.append({
                        "source_id": source_id,
                        "video_path": str(video_path),
                        "save_path": str(save_path),
                        "status": "failed",
                        "message": f"Cannot read frame {frame_idx}",
                    })
                    continue

                save_image(frame, save_path)
                saved_count += 1

                records.append({
                    "source_id": source_id,
                    "video_path": str(video_path),
                    "save_path": str(save_path),
                    "status": "ok",
                    "message": "",
                })

            cap.release()

    return records


def save_records(records: List[dict], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    fieldnames = sorted(set().union(*(r.keys() for r in records)))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved log: {csv_path}")


def check_final_outputs(
    tasks: List[Task],
    target_frames_dir: Path,
    swapped_outputs_dir: Path,
    log_dir: Path,
):
    """
    Check final swapped images.

    Requirements:
        1. Every image in image_list.txt exists.
        2. Every image can be read.
        3. Every swapped image has the same size as its original target frame.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    broken = []
    bad_size = []

    for task in tqdm(tasks, desc="Check final outputs"):
        target_path = target_frames_dir / task.output_name
        output_path = swapped_outputs_dir / task.output_name

        if not output_path.exists():
            missing.append(task.output_name)
            continue

        output_img = cv2.imread(str(output_path))
        if output_img is None:
            broken.append(task.output_name)
            continue

        target_img = cv2.imread(str(target_path))
        if target_img is None:
            broken.append(f"target frame missing: {task.output_name}")
            continue

        th, tw = target_img.shape[:2]
        oh, ow = output_img.shape[:2]

        if (tw, th) != (ow, oh):
            bad_size.append(
                f"{task.output_name}\toutput={ow}x{oh}\ttarget={tw}x{th}"
            )

    (log_dir / "missing.txt").write_text("\n".join(missing), encoding="utf-8")
    (log_dir / "broken.txt").write_text("\n".join(broken), encoding="utf-8")
    (log_dir / "bad_size.txt").write_text("\n".join(bad_size), encoding="utf-8")

    print("Check finished.")
    print(f"Missing:  {len(missing)}")
    print(f"Broken:   {len(broken)}")
    print(f"Bad size: {len(bad_size)}")
    print(f"Logs saved to: {log_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-list",
        type=Path,
        required=True,
        help="Path to generation/image_list.txt",
    )

    parser.add_argument(
        "--videos-dir",
        type=Path,
        required=True,
        help="Directory containing original videos, e.g. id0_0000.mp4",
    )

    parser.add_argument(
        "--target-frames-dir",
        type=Path,
        default=Path("work/target_frames"),
        help="Directory to save extracted target frames",
    )

    parser.add_argument(
        "--source-candidates-dir",
        type=Path,
        default=Path("work/source_candidates"),
        help="Directory to save source candidate images",
    )

    parser.add_argument(
        "--swapped-outputs-dir",
        type=Path,
        default=Path("outputs/YOUR_TEAM_NAME"),
        help="Directory containing final swapped png images",
    )

    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("work/logs"),
        help="Directory to save logs",
    )

    parser.add_argument(
        "--mode",
        choices=["extract-targets", "extract-sources", "extract-all", "check"],
        required=True,
        help=(
            "extract-targets: extract target frames; "
            "extract-sources: extract source candidate images; "
            "extract-all: extract both target frames and source candidates; "
            "check: check final swapped images."
        ),
    )

    parser.add_argument(
        "--frames-per-source",
        type=int,
        default=8,
        help="How many source candidate images to extract for each source id.",
    )

    parser.add_argument(
        "--sample-stride",
        type=int,
        default=5,
        help="Sample one frame every N frames when extracting source candidates.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images.",
    )

    args = parser.parse_args()

    tasks = load_tasks(args.image_list)
    print(f"Loaded tasks: {len(tasks)}")

    if args.mode in ["extract-targets", "extract-all"]:
        target_records = extract_target_frames(
            tasks=tasks,
            videos_dir=args.videos_dir,
            target_frames_dir=args.target_frames_dir,
            overwrite=args.overwrite,
        )
        save_records(target_records, args.log_dir / "target_frames.csv")

    if args.mode in ["extract-sources", "extract-all"]:
        source_records = extract_source_candidates(
            tasks=tasks,
            videos_dir=args.videos_dir,
            source_candidates_dir=args.source_candidates_dir,
            frames_per_source=args.frames_per_source,
            sample_stride=args.sample_stride,
            overwrite=args.overwrite,
        )
        save_records(source_records, args.log_dir / "source_candidates.csv")

    if args.mode == "check":
        check_final_outputs(
            tasks=tasks,
            target_frames_dir=args.target_frames_dir,
            swapped_outputs_dir=args.swapped_outputs_dir,
            log_dir=args.log_dir,
        )


if __name__ == "__main__":
    main()
