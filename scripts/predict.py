"""Inference: โหลดโมเดลที่ฝึกแล้ว -> ทำนายภาพตัวอักษรไทย

ใช้งาน
    python scripts/predict.py --ckpt weights/resnet50_best.pt --input <โฟลเดอร์ภาพ>
    python scripts/predict.py --ckpt weights/resnet50_best.pt --input a.jpg
    python scripts/predict.py --ckpt weights/resnet50_best.pt --input <โฟลเดอร์> --out result.csv

ถ้าโฟลเดอร์จัดเป็น <input>/<รหัส TIS-620>/*.jpg (แบบเดียวกับชุดฝึกสอน)
จะคำนวณ accuracy ให้ด้วย

ไฟล์ .pt มี mapping label->ตัวอักษร ฝังอยู่แล้ว จึงรันได้โดยไม่ต้องมีไฟล์อื่น
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thai_char_cnn.data import build_transform
from thai_char_cnn.model import build_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_image(path: Path) -> torch.Tensor:
    """binarize + invert ให้ตรงกับตอนฝึกสอน (ink=255, พื้นหลัง=0)"""
    a = np.asarray(Image.open(path).convert("L"))
    return torch.from_numpy(np.where(a > 127, 0, 255).astype(np.uint8)).unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ไฟล์ weight (.pt)")
    ap.add_argument("--input", required=True, help="ภาพเดี่ยว หรือโฟลเดอร์")
    ap.add_argument("--out", default=None, help="เขียนผลเป็น CSV")
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    names = ck["classes"]
    size = ck["args"]["size"]

    model = build_model(ck["args"].get("arch", "smallcnn"), len(names)).to(device).eval()
    model.load_state_dict(ck["model"])
    tf = build_transform(size, train=False, aug="none")

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(
        f for f in src.rglob("*") if f.suffix.lower() in EXTS
    )
    if not files:
        raise SystemExit(f"ไม่พบไฟล์ภาพใน {src}")
    print(f"โมเดล {ck['args'].get('arch', 'smallcnn')} | {len(names)} คลาส | {len(files):,} ภาพ")

    preds, confs = [], []
    with torch.no_grad():
        for i in range(0, len(files), args.bs):
            batch = torch.stack([tf(load_image(f)) for f in files[i : i + args.bs]]).to(device)
            p = F.softmax(model(batch).float(), 1)
            c, k = p.max(1)
            preds += k.cpu().tolist()
            confs += c.cpu().tolist()

    import csv

    out_rows = []
    for f, k, c in zip(files, preds, confs):
        ch = names[k]
        # ชื่อโฟลเดอร์แม่เป็นรหัส TIS-620 ในชุดฝึกสอน -> ใช้เป็นเฉลยถ้ามี
        truth = f.parent.name if f.parent.name.isdigit() else ""
        out_rows.append([f.name, ch.encode("tis-620")[0], ch, round(c, 4), truth])

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "predicted_code", "predicted_char", "confidence", "true_code"])
            w.writerows(out_rows)
        print(f"เขียนผล -> {args.out}")
    else:
        for r in out_rows[:20]:
            print(f"  {r[0]:<34} -> {r[2]}  (รหัส {r[1]}, ความมั่นใจ {r[3]:.3f})")
        if len(out_rows) > 20:
            print(f"  ... อีก {len(out_rows) - 20:,} ภาพ (ใส่ --out เพื่อเขียนลงไฟล์)")

    truths = [r[4] for r in out_rows]
    if all(truths):
        acc = np.mean([str(r[1]) == str(r[4]) for r in out_rows])
        print(f"\nAccuracy = {acc:.4f}  ({int(acc * len(out_rows)):,}/{len(out_rows):,})")


if __name__ == "__main__":
    main()
