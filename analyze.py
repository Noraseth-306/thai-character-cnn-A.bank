"""Phase 6: ดูว่าโมเดลสับสนคู่ไหน -- ใช้ตัดสินใจว่า Phase ถัดไปควรแก้อะไร"""
import argparse
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from data import ThaiCharDataset, class_names
from model import build_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/baseline/best.pt")
    ap.add_argument("--index", default="index.csv")
    ap.add_argument("--root", default="ThaiCharacter Dataset")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    size = ck["args"]["size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    va = ThaiCharDataset(args.index, args.root, "val", size)
    names = class_names(args.index)
    n = len(names)

    model = build_model(ck["args"].get("arch", "smallcnn"), n).to(device).eval()
    model.load_state_dict(ck["model"])

    preds = []
    with torch.no_grad():
        for i in range(0, len(va), 512):
            x = torch.stack([va[j][0] for j in range(i, min(i + 512, len(va)))]).to(device)
            preds.append(model(x).argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    trues = va.labels.numpy()

    cm = confusion_matrix(trues, preds, labels=range(n))
    print(f"{args.ckpt}  |  val acc {(preds == trues).mean():.4f}\n")

    # เอาเฉพาะคู่ที่สับสนจริง (ตัดแนวทแยงทิ้ง)
    np.fill_diagonal(cm, 0)
    pairs = [(cm[i, j], i, j) for i, j in zip(*np.nonzero(cm))]
    pairs.sort(reverse=True)

    support = np.bincount(trues, minlength=n)
    print(f"{args.top} คู่ที่สับสนที่สุด (จริง -> ทาย):")
    for c, i, j in pairs[: args.top]:
        pct = 100 * c / max(support[i], 1)
        print(f"  {names[i]} -> {names[j]}   {c:4d} ครั้ง  ({pct:.1f}% ของ {names[i]} ทั้งหมดใน val)")

    # คลาสที่พังทั้งคลาส = สัญญาณว่าข้อมูลน้อยเกิน ไม่ใช่โมเดลไม่ดี
    dead = [(names[i], support[i]) for i in range(n) if support[i] > 0 and cm[i].sum() == support[i]]
    if dead:
        print(f"\nคลาสที่ทายผิดหมดทุกตัว: {', '.join(f'{c}(n={s})' for c, s in dead)}")

    novál = [names[i] for i in range(n) if support[i] == 0]
    if noval := novál:
        print(f"\nคลาสที่ไม่มี val เลย (ตัวเลขข้างบนมองไม่เห็น): {' '.join(noval)}")


if __name__ == "__main__":
    main()


def per_class_recall(ckpt: str, index: str, root: str) -> "pd.Series":
    """คำนวณ recall รายคลาสจาก checkpoint โดยตรง

    ไม่อ่านจาก report.json เพราะไฟล์นั้นอาจค้างมาจากการเทรนรอบก่อนแก้บั๊ก
    """
    import pandas as pd

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    names = class_names(index)
    va = ThaiCharDataset(index, root, "val", ck["args"]["size"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model(ck["args"].get("arch", "smallcnn"), len(names)).to(dev).eval()
    m.load_state_dict(ck["model"])
    pr = []
    with torch.no_grad():
        for i in range(0, len(va), 512):
            x = torch.stack([va[j][0] for j in range(i, min(i + 512, len(va)))]).to(dev)
            pr.append(m(x).argmax(1).cpu())
    pr, tr = torch.cat(pr).numpy(), va.labels.numpy()
    return pd.Series(
        {names[c]: (pr[tr == c] == c).mean() for c in range(len(names)) if (tr == c).sum()}
    )
