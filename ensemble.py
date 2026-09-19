"""รวมผลหลายโมเดลด้วยการเฉลี่ย softmax -- ไม่ต้องเทรนใหม่

โมเดลแต่ละตัวรับ input คนละขนาด (48/64/96) จึงต้องสร้าง dataset ของใครของมัน
แต่ลำดับแถวเหมือนกันเพราะอ่านจาก index.csv เดียวกัน
"""
import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from data import ThaiCharDataset, class_names
from model import build_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def probs_of(ckpt: str, index: str, root: str, n: int, device) -> tuple[np.ndarray, np.ndarray]:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    ds = ThaiCharDataset(index, root, "val", a["size"])
    m = build_model(a.get("arch", "smallcnn"), n).to(device).eval()
    m.load_state_dict(ck["model"])
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), 256):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 256, len(ds)))]).to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                out.append(F.softmax(m(x).float(), 1).cpu())
    return torch.cat(out).numpy(), ds.labels.numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--index", default="index.csv")
    ap.add_argument("--root", default="ThaiCharacter Dataset")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = len(class_names(args.index))

    all_p, y = [], None
    for c in args.ckpts:
        p, y = probs_of(c, args.index, args.root, n, device)
        acc = (p.argmax(1) == y).mean()
        print(f"  {acc:.4f}  {c}")
        all_p.append(p)

    from sklearn.metrics import f1_score

    print()
    for k in range(2, len(all_p) + 1):
        avg = np.mean(all_p[:k], axis=0)
        pred = avg.argmax(1)
        print(f"ensemble {k} ตัว: acc {(pred == y).mean():.4f}  "
              f"macroF1 {f1_score(y, pred, average='macro', zero_division=0):.4f}")


if __name__ == "__main__":
    main()
