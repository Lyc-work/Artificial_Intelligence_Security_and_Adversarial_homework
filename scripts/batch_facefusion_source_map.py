import argparse
import csv
import subprocess
from pathlib import Path


def load_image_list(image_list_path: Path):
    if image_list_path is None:
        return None

    with image_list_path.open("r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_source_map(source_map_path: Path, allowed_names=None):
    rows = []

    with source_map_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            output_name = row.get("output_name", "").strip()
            if not output_name:
                continue

            if allowed_names is not None and output_name not in allowed_names:
                continue

            rows.append(row)

    return rows


def write_detail_log(
    detail_log_path: Path,
    cmd,
    cwd: Path,
    stdout: str,
    stderr: str,
    returncode: int,
):
    detail_log_path.parent.mkdir(parents=True, exist_ok=True)

    detail_log_path.write_text(
        "RETURN CODE:\n"
        + str(returncode)
        + "\n\nCMD:\n"
        + " ".join(cmd)
        + "\n\nCWD:\n"
        + str(cwd)
        + "\n\nSTDOUT:\n"
        + (stdout or "")
        + "\n\nSTDERR:\n"
        + (stderr or ""),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Batch run FaceFusion using per-target selected source map."
    )

    parser.add_argument(
        "--facefusion-script",
        type=Path,
        required=True,
        help="Path to facefusion.py",
    )

    parser.add_argument(
        "--source-map",
        type=Path,
        required=True,
        help="Path to selected_source_map.csv",
    )

    parser.add_argument(
        "--target-dir",
        type=Path,
        required=True,
        help="Directory of target frames",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save final swapped images",
    )

    parser.add_argument(
        "--image-list",
        type=Path,
        default=None,
        help="Optional image list split. If provided, only run tasks in this list.",
    )

    parser.add_argument(
        "--python-bin",
        type=str,
        default="python",
    )

    parser.add_argument(
        "--face-swapper-model",
        type=str,
        default="inswapper_128",
        help="Examples: inswapper_128, inswapper_128_fp16, simswap_256",
    )

    parser.add_argument(
        "--face-detector-model",
        type=str,
        default="retinaface",
        help="Examples: retinaface, scrfd",
    )

    parser.add_argument(
        "--face-detector-score",
        type=str,
        default="0.25",
    )

    parser.add_argument(
        "--execution-providers",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--execution-device-ids",
        type=str,
        default="0",
        help="When CUDA_VISIBLE_DEVICES is set, usually keep this as 0.",
    )

    parser.add_argument(
        "--execution-thread-count",
        type=str,
        default="1",
    )

    parser.add_argument(
        "--video-memory-strategy",
        type=str,
        default="moderate",
        choices=["strict", "moderate", "tolerant"],
    )

    parser.add_argument(
        "--output-image-quality",
        type=str,
        default="100",
    )

    parser.add_argument(
        "--temp-root",
        type=Path,
        default=Path("work/temp/source_map"),
        help="Temp root. Use different temp roots for different GPUs.",
    )

    parser.add_argument(
        "--jobs-root",
        type=Path,
        default=Path("work/jobs/source_map"),
        help="Jobs root. Use different jobs roots for different GPUs.",
    )

    parser.add_argument(
        "--skip-exists",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--keep-success-logs",
        action="store_true",
    )

    args = parser.parse_args()

    facefusion_script = args.facefusion_script.resolve()
    facefusion_root = facefusion_script.parent.resolve()

    source_map_path = args.source_map.resolve()
    target_dir = args.target_dir.resolve()
    output_dir = args.output_dir.resolve()
    temp_root = args.temp_root.resolve()
    jobs_root = args.jobs_root.resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    jobs_root.mkdir(parents=True, exist_ok=True)

    if not facefusion_script.exists():
        raise FileNotFoundError(f"FaceFusion script not found: {facefusion_script}")

    if not source_map_path.exists():
        raise FileNotFoundError(f"Source map not found: {source_map_path}")

    if not target_dir.exists():
        raise FileNotFoundError(f"Target dir not found: {target_dir}")

    allowed_names = None
    if args.image_list is not None:
        allowed_names = load_image_list(args.image_list.resolve())

    rows = load_source_map(source_map_path, allowed_names=allowed_names)

    if args.limit is not None:
        rows = rows[:args.limit]

    print("=" * 90)
    print("Batch FaceFusion with selected source map")
    print("=" * 90)
    print(f"FaceFusion script     : {facefusion_script}")
    print(f"FaceFusion cwd        : {facefusion_root}")
    print(f"Source map            : {source_map_path}")
    print(f"Target dir            : {target_dir}")
    print(f"Output dir            : {output_dir}")
    print(f"Temp root             : {temp_root}")
    print(f"Jobs root             : {jobs_root}")
    print(f"Tasks                 : {len(rows)}")
    print(f"Face swapper model    : {args.face_swapper_model}")
    print(f"Face detector model   : {args.face_detector_model}")
    print(f"Face detector score   : {args.face_detector_score}")
    print(f"Execution providers   : {args.execution_providers}")
    print(f"Execution device ids  : {args.execution_device_ids}")
    print("=" * 90)

    success = 0
    failed = 0
    skipped = 0
    failed_records = []

    failed_detail_dir = output_dir / "_failed_detail_logs"
    success_detail_dir = output_dir / "_success_detail_logs"

    for idx, row in enumerate(rows, start=1):
        output_name = row["output_name"].strip()

        # select_source_per_target.py 里通常有 selected_copy_path
        # 如果没有，则退回 candidate_path
        source_raw = row.get("selected_copy_path") or row.get("candidate_path")
        if not source_raw:
            print(f"[FAILED] no selected source path for {output_name}")
            failed += 1
            failed_records.append(f"{output_name}\tmissing_selected_source_path")
            continue

        source_path = Path(source_raw).resolve()
        target_path = (target_dir / output_name).resolve()
        output_path = (output_dir / output_name).resolve()

        print("\n" + "=" * 90)
        print(f"[{idx}/{len(rows)}] {output_name}")
        print("SOURCE:", source_path)
        print("TARGET:", target_path)
        print("OUTPUT:", output_path)

        if not source_path.exists():
            print(f"[FAILED] source missing: {source_path}")
            failed += 1
            failed_records.append(f"{output_name}\tmissing_source\t{source_path}")
            continue

        if not target_path.exists():
            print(f"[FAILED] target missing: {target_path}")
            failed += 1
            failed_records.append(f"{output_name}\tmissing_target\t{target_path}")
            continue

        if args.skip_exists and output_path.exists():
            print(f"[SKIP EXISTS] {output_path}")
            skipped += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)

        task_stem = Path(output_name).stem
        task_temp_path = temp_root / task_stem
        task_jobs_path = jobs_root / task_stem
        task_temp_path.mkdir(parents=True, exist_ok=True)
        task_jobs_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python_bin,
            str(facefusion_script),
            "headless-run",

            "-s", str(source_path),
            "-t", str(target_path),
            "-o", str(output_path),

            "--temp-path", str(task_temp_path),
            "--jobs-path", str(task_jobs_path),

            "--processors", "face_swapper",
            "--face-swapper-model", args.face_swapper_model,

            "--face-detector-model", args.face_detector_model,
            "--face-detector-score", args.face_detector_score,

            "--execution-providers", args.execution_providers,
            "--execution-device-ids", args.execution_device_ids,
            "--execution-thread-count", args.execution_thread_count,
            "--video-memory-strategy", args.video_memory_strategy,

            "--output-image-quality", args.output_image_quality,
            "--output-image-scale", "1.0",

            "--log-level", "info",
        ]

        print("CMD:", " ".join(cmd))
        print("CWD:", facefusion_root)

        if args.dry_run:
            continue

        result = subprocess.run(
            cmd,
            cwd=str(facefusion_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 有些任务 returncode 非 0 但实际图片已经写出，所以这里以 output 是否存在为主。
        if output_path.exists():
            if result.returncode == 0:
                print(f"[SUCCESS] {output_path}")
            else:
                print(f"[SUCCESS WITH NONZERO RETURN] {output_path}, returncode={result.returncode}")

            success += 1

            if args.keep_success_logs:
                success_log = success_detail_dir / f"{task_stem}.log"
                write_detail_log(
                    detail_log_path=success_log,
                    cmd=cmd,
                    cwd=facefusion_root,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
        else:
            print(f"[FAILED] {output_name}, returncode={result.returncode}")
            failed += 1
            failed_records.append(
                f"{output_name}\tfacefusion_failed\treturncode={result.returncode}"
            )

            detail_log = failed_detail_dir / f"{task_stem}.log"
            write_detail_log(
                detail_log_path=detail_log,
                cmd=cmd,
                cwd=facefusion_root,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )

            print(f"[DETAIL LOG] {detail_log}")

    failed_log = output_dir / "_failed_tasks.txt"
    failed_log.write_text("\n".join(failed_records), encoding="utf-8")

    print("\n" + "=" * 90)
    print("Batch finished.")
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    print(f"Skipped: {skipped}")
    print(f"Failed log: {failed_log}")
    print(f"Failed detail logs: {failed_detail_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
