"""Phase 1: เทรน baseline + เก็บ metric ไว้เทียบกับ Phase ถัด ๆ ไป

วัด macro-F1 ควบคู่ accuracy เสมอ
accuracy เฉย ๆ โกงได้ง่ายมากในชุดนี้ (คลาสใหญ่สุด 5,025 : เล็กสุด 1 รูป)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from thai_char_cnn.data import ThaiCharDataset, class_names
from thai_char_cnn.model import build_model

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    """กำหนด seed ให้การทดลองทำซ้ำได้ใกล้เคียงกันทุกครั้ง"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # เลือก reproducibility มากกว่าความเร็วเล็กน้อย
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, preds, trues = 0.0, [], []
    scaler = run_epoch.scaler

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * y.size(0)
            preds.append(out.argmax(1).cpu())
            trues.append(y.cpu())

    preds, trues = torch.cat(preds).numpy(), torch.cat(trues).numpy()
    return {
        "loss": total_loss / len(trues),
        "acc": (preds == trues).mean(),
        "macro_f1": f1_score(trues, preds, average="macro", zero_division=0),
        "preds": preds,
        "trues": trues,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.csv")
    ap.add_argument("--root", default="ThaiCharacter Dataset")
    # Phase 5: เทรนรอบแรกด้วยข้อมูลสังเคราะห์ (--train-* ชี้ไป synth) แล้วค่อย
    # fine-tune ด้วยของจริงล้วน (--init-from) -- val ใช้ของจริงเสมอ
    ap.add_argument("--train-index", default=None)
    ap.add_argument("--train-root", default=None)
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--arch", default="smallcnn",
                    help="smallcnn | resnet18 | resnet34 | resnet50 | vgg19")
    ap.add_argument("--freeze-epochs", type=int, default=0,
                    help="แช่แข็ง backbone กี่ epoch แรก (ให้ชั้นสุดท้ายตั้งตัวก่อน)")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", default="none", choices=["none", "affine", "domain"],
                    help="none | affine (วัดแล้วแย่ลง) | domain (dilate/erode ตามการสแกนจริง)")
    ap.add_argument("--balanced-sampler", action="store_true", help="Phase 4: แก้ imbalance")
    ap.add_argument("--class-weights", action="store_true", help="Phase 4: ถ่วงน้ำหนัก loss")
    args = ap.parse_args()

    seed_everything(args.seed)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  args: {vars(args)}\n")

    t0 = time.time()
    tr = ThaiCharDataset(
        args.train_index or args.index, args.train_root or args.root,
        args.train_split, args.size, aug=args.aug,
    )
    va = ThaiCharDataset(args.index, args.root, "val", args.size)
    names = class_names(args.index)
    n_classes = len(names)
    print(f"โหลด train {len(tr):,} / val {len(va):,} ({time.time() - t0:.0f}s) | {n_classes} คลาส")

    counts = tr.class_counts
    if args.balanced_sampler:
        # 1/sqrt(n) ไม่ใช่ 1/n: 1/n จะดันคลาสที่มี 1 รูปหนักเกินจน overfit รูปเดียวนั้น
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        sampler = WeightedRandomSampler(
            torch.as_tensor(w[tr.df["label"].to_numpy()], dtype=torch.double), len(tr), True
        )
        train_loader = DataLoader(tr, args.bs, sampler=sampler, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(tr, args.bs, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(va, args.bs, shuffle=False, pin_memory=True)

    model = build_model(args.arch, n_classes).to(device)
    if args.freeze_epochs:
        model.set_backbone_frozen(True)
        print(f"แช่แข็ง backbone {args.freeze_epochs} epoch แรก")
    if args.init_from:
        model.load_state_dict(
            torch.load(args.init_from, map_location=device, weights_only=False)["model"]
        )
        print(f"เริ่มจากน้ำหนักของ {args.init_from}")
    print(f"พารามิเตอร์: {sum(p.numel() for p in model.parameters()):,}\n")

    if args.balanced_sampler and args.class_weights:
        # เปิดพร้อมกันคือแก้ imbalance ซ้อนสองชั้น: sampler ปรับสมดุลไปแล้วรอบหนึ่ง
        # แล้ว loss ถ่วงซ้ำอีก -> คลาสใหญ่ถูกกดจนพัง (วัดได้: val acc 0.98 -> 0.01)
        raise SystemExit("เลือกอย่างใดอย่างหนึ่ง: --balanced-sampler หรือ --class-weights")

    cw = None
    if args.class_weights:
        # 1/sqrt(n) แล้ว normalize ให้เฉลี่ย 1 -- ไม่ใช้ 1/n เพราะคลาสที่มี 1 รูป
        # จะได้น้ำหนักสูงกว่าคลาสใหญ่ถึง 2,000 เท่า
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
        cw = torch.as_tensor(w / w.mean(), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_epoch.scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history, best = [], -1.0
    for ep in range(1, args.epochs + 1):
        if args.freeze_epochs and ep == args.freeze_epochs + 1:
            model.set_backbone_frozen(False)
            # lr ต่ำลง 10 เท่า ไม่งั้น gradient แรก ๆ จะทำลายน้ำหนัก pretrained ทิ้ง
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs
            )
            print(f"ปลดล็อก backbone ทั้งหมด (lr={args.lr * 0.1:g})")
        t = time.time()
        trm = run_epoch(model, train_loader, device, criterion, optimizer)
        vam = run_epoch(model, val_loader, device, criterion)
        sched.step()
        history.append(
            {k: v for k, v in
             {"epoch": ep, "train_loss": trm["loss"], "train_acc": trm["acc"],
              "val_loss": vam["loss"], "val_acc": vam["acc"], "val_macro_f1": vam["macro_f1"],
              "lr": sched.get_last_lr()[0], "sec": time.time() - t}.items()}
        )
        flag = ""
        if vam["macro_f1"] > best:
            best = vam["macro_f1"]
            torch.save(
                # ฝัง classes ลงไปด้วย -> ไฟล์ weight ใช้ inference ได้เองโดยไม่ต้องมี index.csv
                {"model": model.state_dict(), "args": vars(args), "classes": names},
                out / "best.pt",
            )
            flag = "  <- best"
        print(
            f"ep {ep:2d}/{args.epochs} | train loss {trm['loss']:.4f} acc {trm['acc']:.4f} "
            f"| val loss {vam['loss']:.4f} acc {vam['acc']:.4f} macroF1 {vam['macro_f1']:.4f} "
            f"| {time.time() - t:.0f}s{flag}"
        )

    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # รายงานรายคลาสด้วยตัวอักษรจริง อ่านง่ายกว่าเลข label
    rep = classification_report(
        vam["trues"], vam["preds"], labels=range(len(names)),
        target_names=names, zero_division=0, output_dict=True,
    )
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nดีสุด val macro-F1 = {best:.4f}  (เซฟที่ {out / 'best.pt'})")
    worst = sorted(
        ((n, rep[n]["f1-score"], rep[n]["support"]) for n in names if rep[n]["support"] > 0),
        key=lambda r: r[1],
    )[:10]
    print("\n10 คลาสที่แย่ที่สุด (char, F1, n_val):")
    for n, f1, sup in worst:
        print(f"  {n}  F1={f1:.3f}  n={int(sup)}")


if __name__ == "__main__":
    main()
