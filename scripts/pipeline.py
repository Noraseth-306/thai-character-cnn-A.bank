"""End-to-end pipeline สำหรับ Thai Character CNN

ลำดับมาตรฐาน:
    1) prepare   สแกน/clean/split ข้อมูลจริงประมาณ 80:20
    2) synth     ปรับ PrintAksorn ให้ใกล้ domain ภาพสแกน
    3) mix       เติมข้อมูลเฉพาะคลาสที่มีภาพจริงน้อย
    4) train     ResNet-50 ImageNet transfer learning + fine-tuning
    5) evaluate  วิเคราะห์ผลรายคลาส
    6) smoke     โหลด Weight แล้วทำนายภาพจริงหนึ่งภาพ

ตัวอย่าง:
    python scripts/pipeline.py
    python scripts/pipeline.py --resume
    python scripts/pipeline.py --dry-run
    python scripts/pipeline.py --quick
    python scripts/pipeline.py --start-at train --resume
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STAGES = ("prepare", "synth", "mix", "train", "evaluate", "smoke")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


def command_text(command: list[str]) -> str:
    """แสดง command แบบที่ copy ไปรันบน Windows ได้"""
    return subprocess.list2cmdline(command)


def run_command(command: list[str], dry_run: bool) -> None:
    print(f"$ {command_text(command)}", flush=True)
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run(command, check=True, env=env)


def outputs_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)


def save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_range(start: str, stop: str) -> tuple[str, ...]:
    a, b = STAGES.index(start), STAGES.index(stop)
    if a > b:
        raise SystemExit("--start-at ต้องอยู่ก่อนหรือเท่ากับ --stop-after")
    return STAGES[a : b + 1]


def main() -> None:
    # ทำให้ path ค่าเริ่มต้นอ้างอิงจาก root ของโครงการเสมอ
    os.chdir(PROJECT_ROOT)
    ap = argparse.ArgumentParser(description="Thai Character CNN end-to-end pipeline")
    ap.add_argument("--real-root", default="ThaiCharacter Dataset")
    ap.add_argument("--print-root", default="PrintAksorn_dataset")
    ap.add_argument("--index", default=None)
    ap.add_argument("--synth-root", default=None)
    ap.add_argument("--synth-index", default=None)
    ap.add_argument("--mixed-index", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--below", type=int, default=100)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--per-class", type=int, default=406)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--min-accuracy", type=float, default=0.97)
    ap.add_argument("--min-macro-f1", type=float, default=0.95)
    ap.add_argument("--start-at", choices=STAGES, default="prepare")
    ap.add_argument("--stop-after", choices=STAGES, default="smoke")
    ap.add_argument("--resume", action="store_true", help="ข้ามขั้นที่มี output ครบแล้ว")
    ap.add_argument("--dry-run", action="store_true", help="แสดงคำสั่งโดยไม่แก้ไฟล์หรือฝึกโมเดล")
    ap.add_argument("--quick", action="store_true", help="smoke test pipeline ด้วยข้อมูล/epoch จำนวนน้อย")
    args = ap.parse_args()

    if args.quick:
        args.index = args.index or "index_quick.csv"
        args.synth_root = args.synth_root or "synth_quick"
        args.synth_index = args.synth_index or "synth_index_quick.csv"
        args.mixed_index = args.mixed_index or "index_mixed_quick.csv"
        args.model_dir = args.model_dir or "runs/pipeline_quick"
        args.per_class = min(args.per_class, 8)
        args.below = min(args.below, 10)
        args.cap = min(args.cap, 20)
        args.epochs = 1
        args.freeze_epochs = 0
    else:
        args.index = args.index or "index.csv"
        args.synth_root = args.synth_root or "synth"
        args.synth_index = args.synth_index or "synth_index.csv"
        args.mixed_index = args.mixed_index or "index_mixed.csv"
        args.model_dir = args.model_dir or "runs/pipeline_mix_resnet50"

    root = Path(args.real_root)
    print_root = Path(args.print_root)
    if not root.is_dir():
        raise SystemExit(f"ไม่พบชุดข้อมูลจริง: {root}")
    if not print_root.is_dir() and args.start_at in {"prepare", "synth"}:
        raise SystemExit(f"ไม่พบ PrintAksorn: {print_root}")

    py = sys.executable
    model_dir = Path(args.model_dir)
    state_path = Path("pipeline_state.json")
    state = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": vars(args),
        "stages": {},
    }

    sample = next(root.rglob("*.jpg"), None)
    commands: dict[str, tuple[list[str], list[Path]]] = {
        "prepare": (
            [py, str(SCRIPT_DIR / "prepare_data.py"), "--root", args.real_root, "--out", args.index,
             "--val-frac", "0.20", "--seed", str(args.seed)],
            [Path(args.index)],
        ),
        "synth": (
            [py, str(SCRIPT_DIR / "synth.py"), "--index", args.index, "--real-root", args.real_root,
             "--pim", args.print_root, "--out", args.synth_root,
             "--index-out", args.synth_index, "--per-class", str(args.per_class),
             "--seed", str(args.seed)],
            [Path(args.synth_index), Path(args.synth_root)],
        ),
        "mix": (
            [py, str(SCRIPT_DIR / "mix_synth.py"), "--real-index", args.index,
             "--synth-index", args.synth_index, "--real-root", args.real_root,
             "--synth-root", args.synth_root, "--below", str(args.below),
             "--cap", str(args.cap), "--out", args.mixed_index,
             "--seed", str(args.seed)],
            [Path(args.mixed_index)],
        ),
        "train": (
            [py, str(SCRIPT_DIR / "train.py"), "--index", args.index, "--root", args.real_root,
             "--train-index", args.mixed_index, "--train-root", ".",
             "--train-split", "train", "--arch", "resnet50", "--size", "96",
             "--bs", str(args.batch_size), "--epochs", str(args.epochs),
             "--freeze-epochs", str(args.freeze_epochs), "--lr", str(args.learning_rate),
             "--seed", str(args.seed), "--out", args.model_dir],
            [model_dir / "best.pt", model_dir / "history.csv", model_dir / "report.json"],
        ),
        "evaluate": (
            [py, str(SCRIPT_DIR / "analyze.py"), "--ckpt", str(model_dir / "best.pt"),
             "--index", args.index, "--root", args.real_root],
            [],
        ),
        "smoke": (
            [py, str(SCRIPT_DIR / "predict.py"), "--ckpt", str(model_dir / "best.pt"),
             "--input", str(sample)] if sample else [],
            [],
        ),
    }

    selected = stage_range(args.start_at, args.stop_after)
    print("\nPipeline:", " -> ".join(selected))
    print(f"Mode: {'quick' if args.quick else 'full'} | seed={args.seed} | resume={args.resume}\n")

    for name in selected:
        command, expected = commands[name]
        print(f"{'=' * 16} {name.upper()} {'=' * 16}", flush=True)
        if not command:
            raise SystemExit(f"ขั้น {name} ไม่มี command ที่ใช้งานได้")
        if args.resume and outputs_exist(expected):
            print(f"ข้าม {name}: พบ output ครบแล้ว")
            state["stages"][name] = {"status": "skipped", "outputs": [str(p) for p in expected]}
            continue
        started = time.time()
        try:
            run_command(command, args.dry_run)
        except subprocess.CalledProcessError as exc:
            state["stages"][name] = {"status": "failed", "returncode": exc.returncode}
            if not args.dry_run:
                save_state(state_path, state)
            raise SystemExit(f"Pipeline หยุดที่ขั้น {name} (exit code {exc.returncode})") from exc
        state["stages"][name] = {
            "status": "dry-run" if args.dry_run else "complete",
            "seconds": round(time.time() - started, 2),
            "outputs": [str(p) for p in expected],
        }
        if not args.dry_run:
            save_state(state_path, state)

    # Quality gate ใช้แถวเดียวกับ Weight ที่ถูกเลือกด้วย Macro-F1
    history_path = model_dir / "history.csv"
    if not args.dry_run and history_path.exists() and not args.quick:
        history = pd.read_csv(history_path)
        best = history.loc[history["val_macro_f1"].idxmax()]
        summary = {
            "checkpoint": str(model_dir / "best.pt"),
            "best_epoch": int(best["epoch"]),
            "val_accuracy": float(best["val_acc"]),
            "val_macro_f1": float(best["val_macro_f1"]),
            "min_accuracy": args.min_accuracy,
            "min_macro_f1": args.min_macro_f1,
        }
        summary["passed"] = (
            summary["val_accuracy"] >= args.min_accuracy
            and summary["val_macro_f1"] >= args.min_macro_f1
        )
        Path("pipeline_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("\nQuality gate:")
        print(f"  Accuracy : {summary['val_accuracy']:.4f} (ขั้นต่ำ {args.min_accuracy:.4f})")
        print(f"  Macro-F1: {summary['val_macro_f1']:.4f} (ขั้นต่ำ {args.min_macro_f1:.4f})")
        if not summary["passed"]:
            raise SystemExit("โมเดลไม่ผ่าน quality gate")
        print("  Result   : PASS")

    print("\nPipeline เสร็จสมบูรณ์")


if __name__ == "__main__":
    main()
