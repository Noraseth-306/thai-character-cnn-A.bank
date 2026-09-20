"""Build a non-overlapping train/validation index for handwritten datasets.

Only classes present in the selected checkpoint are included.  The output has
the same schema expected by ``train.py`` and ``evaluate.py`` and stores paths
relative to the project root, so use ``--root .`` with it.
"""
import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_handwritten import EXTS, code_from_filename

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sample_id(path: Path) -> str:
    """Best available sample identifier from the source filename."""
    return path.stem.rsplit("-", 1)[-1]


def content_hash(path: Path) -> bytes:
    """Group exact duplicates so they cannot cross the split boundary."""
    return hashlib.sha256(path.read_bytes()).digest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/resnet50_best.pt")
    ap.add_argument("--input", nargs="+", default=["Thai_char_sqr", "Thai_digit_sqr"])
    ap.add_argument("--out", default="handwritten_index.csv")
    ap.add_argument(
        "--include-main-index",
        default=None,
        help="ผนวก index ชุดหลักเพื่อสร้างชุด universal (เช่น index.csv)",
    )
    ap.add_argument("--main-root", default="ThaiCharacter Dataset")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio ต้องอยู่ระหว่าง 0 และ 1")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = ck["classes"]
    code_to_label = {char.encode("tis-620")[0]: i for i, char in enumerate(names)}

    by_code = defaultdict(list)
    skipped = 0
    for root in map(Path, args.input):
        for path in root.rglob("*"):
            if path.suffix.lower() not in EXTS:
                continue
            code = code_from_filename(path)
            if code in code_to_label:
                by_code[code].append(path)
            else:
                skipped += 1

    rng = random.Random(args.seed)
    rows = []
    for code, paths in sorted(by_code.items()):
        duplicate_groups = defaultdict(list)
        for path in paths:
            duplicate_groups[content_hash(path)].append(path)
        groups = list(duplicate_groups.values())
        rng.shuffle(groups)
        target_val = max(1, round(len(paths) * args.val_ratio))
        val_paths = set()
        for group in groups:
            if len(val_paths) >= target_val:
                break
            val_paths.update(group)
        label = code_to_label[code]
        char = names[label]
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            rows.append(
                {
                    "path": path.as_posix(),
                    "code": code,
                    "char": char,
                    "group": f"handwritten_{sample_id(path)}",
                    "width": width,
                    "height": height,
                    "split": "val" if path in val_paths else "train",
                    "label": label,
                }
            )

    frame = pd.DataFrame(rows)
    frame["domain"] = "handwritten"
    if args.include_main_index:
        main = pd.read_csv(args.include_main_index, encoding="utf-8-sig")
        main["path"] = main["path"].map(
            lambda path: (Path(args.main_root) / str(path)).as_posix()
        )
        main["domain"] = "main"
        frame = pd.concat([main, frame], ignore_index=True)
    frame = frame.sort_values(["label", "split", "path"])
    frame.to_csv(args.out, index=False, encoding="utf-8-sig")
    counts = frame["split"].value_counts()
    print(f"เขียน {args.out}")
    print(f"train={counts.get('train', 0):,}  val={counts.get('val', 0):,}")
    print(f"handwritten classes={len(by_code)}  skipped={skipped:,}  seed={args.seed}")


if __name__ == "__main__":
    main()
