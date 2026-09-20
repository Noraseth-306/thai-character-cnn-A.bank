"""Evaluate a checkpoint on the external handwritten Thai datasets.

The handwritten folders are arranged by ordinal class (1, 2, ...), while the
actual TIS-620 class code is embedded in each filename, for example::

    0-161-A1-KO KAI-1002.png  -> 161 -> ก

Reading the filename instead of the parent folder prevents an easy-to-miss
label mismatch with the main dataset, whose folders are named by TIS-620 code.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thai_char_cnn.data import build_transform
from thai_char_cnn.model import build_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CODE_RE = re.compile(r"(?:^|-)(\d{3})(?:-|$)")


def code_from_filename(path: Path) -> int:
    """Return the TIS-620 code embedded as a three-digit filename field."""
    match = CODE_RE.search(path.stem)
    if not match:
        raise ValueError(f"หารหัส TIS-620 ในชื่อไฟล์ไม่พบ: {path.name}")
    return int(match.group(1))


def load_image(path: Path) -> torch.Tensor:
    """Match training preprocessing: binarize, then make ink=255/background=0."""
    from PIL import Image

    pixels = np.asarray(Image.open(path).convert("L"))
    binary = np.where(pixels > 127, 0, 255).astype(np.uint8)
    return torch.from_numpy(binary).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/resnet50_best.pt")
    ap.add_argument(
        "--input",
        nargs="+",
        default=["Thai_char_sqr", "Thai_digit_sqr"],
        help="หนึ่งโฟลเดอร์ขึ้นไปที่ชื่อไฟล์มีรหัส TIS-620",
    )
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    files = sorted(
        path
        for root in map(Path, args.input)
        for path in root.rglob("*")
        if path.suffix.lower() in EXTS
    )
    if not files:
        raise SystemExit("ไม่พบภาพลายมือ")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = ck["classes"]
    code_to_label = {char.encode("tis-620")[0]: i for i, char in enumerate(names)}

    kept, truths, skipped = [], [], []
    for path in files:
        code = code_from_filename(path)
        if code in code_to_label:
            kept.append(path)
            truths.append(code_to_label[code])
        else:
            skipped.append((path, code))
    if not kept:
        raise SystemExit("ไม่มีคลาสในชุดลายมือที่ตรงกับ checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ck["args"].get("arch", "smallcnn"), len(names)).to(device).eval()
    model.load_state_dict(ck["model"])
    transform = build_transform(ck["args"]["size"], train=False, aug="none")

    preds = []
    with torch.no_grad():
        for start in range(0, len(kept), args.bs):
            batch = torch.stack(
                [transform(load_image(path)) for path in kept[start : start + args.bs]]
            ).to(device)
            preds.extend(model(batch).argmax(1).cpu().tolist())

    truths = np.asarray(truths)
    preds = np.asarray(preds)
    present = sorted(set(truths.tolist()))
    accuracy = float((preds == truths).mean())
    macro_f1 = f1_score(truths, preds, labels=present, average="macro", zero_division=0)

    print(f"Checkpoint       : {args.ckpt}")
    print(f"Device           : {device}")
    print(f"Handwritten data : {len(kept):,} images, {len(present)} classes")
    if skipped:
        print(f"Skipped          : {len(skipped):,} images (class not in checkpoint)")
    print(f"Accuracy         : {accuracy:.4f} ({(preds == truths).sum():,}/{len(truths):,})")
    print(f"Macro-F1         : {macro_f1:.4f}\n")

    matrix = confusion_matrix(truths, preds, labels=range(len(names)))
    support = np.bincount(truths, minlength=len(names))
    correct = np.diag(matrix).copy()
    np.fill_diagonal(matrix, 0)
    pairs = [(matrix[i, j], i, j) for i, j in zip(*np.nonzero(matrix))]
    pairs.sort(reverse=True)

    print(f"{args.top} คู่ที่สับสนที่สุด (จริง -> ทาย):")
    for count, true_label, pred_label in pairs[: args.top]:
        pct = 100 * count / support[true_label]
        print(
            f"  {names[true_label]} -> {names[pred_label]}  "
            f"{count:4d} ครั้ง ({pct:5.1f}% ของ {names[true_label]})"
        )

    recalls = [
        (correct[label] / support[label], names[label], support[label])
        for label in present
    ]
    print(f"\n{args.top} คลาสที่ recall ต่ำที่สุด:")
    for recall, char, count in sorted(recalls)[: args.top]:
        print(f"  {char}  recall={recall:.3f}  n={count}")


if __name__ == "__main__":
    main()
