"""Phase 0: สแกน dataset -> index.csv พร้อม split แบบ group-by-document.

dataset นี้เป็น "ตัวพิมพ์สแกนแล้ว binarize" ไม่ใช่ลายมือ
ชื่อไฟล์ bc_001sg_3_118.jpg = <source>_<docid><style>_<page>_<idx>
group = source_docid+style  (เช่น bc_001sg) -> ใช้แบ่ง train/val
สุ่มแบ่งเฉย ๆ ไม่ได้ เพราะฟอนต์เดียวกันจะอยู่ทั้งสองฝั่ง -> val acc สูงหลอก
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# console Windows เป็น cp1252 พิมพ์ไทยแล้ว crash
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ชื่อโฟลเดอร์คือรหัส TIS-620 -> 161 = ก
CODE_RE = re.compile(r"^\d+$")
# "Copy of " ต้องตัดทิ้งก่อน ไม่งั้นได้ group ปลอม (มี 6 ไฟล์)
NAME_RE = re.compile(r"^(?:Copy of )?(\w{2}_\d{3})([st]g)_")

# คลาสที่รูปน้อยกว่านี้ ไม่กัน val (กันไปก็เหลือ train ไม่พอ)
MIN_IMAGES_FOR_VAL = 10


def tis620_char(code: int) -> str:
    return bytes([code]).decode("tis-620")


def scan(root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(root.iterdir()):
        if not (class_dir.is_dir() and CODE_RE.match(class_dir.name)):
            continue
        code = int(class_dir.name)
        for f in sorted(class_dir.glob("*.jpg")):
            m = NAME_RE.match(f.name)
            if not m:
                print(f"  ! ชื่อไฟล์ผิดรูปแบบ ข้าม: {f.name}")
                continue
            with Image.open(f) as im:
                w, h = im.size
            rows.append(
                {
                    "path": f.relative_to(root).as_posix(),
                    "code": code,
                    "char": tis620_char(code),
                    "group": f"{m.group(1)}{m.group(2)}",
                    "width": w,
                    "height": h,
                }
            )
    return pd.DataFrame(rows)


def assign_split(df: pd.DataFrame, val_frac: float, seed: int) -> pd.DataFrame:
    """กัน group ทั้งกลุ่มไว้เป็น val — ห้ามให้ group เดียวกันโผล่สองฝั่ง"""
    # คลาสหายาก (ฃ=1, ฑ=1, ...) อยู่ train เสมอ กันไปก็เหลือ train ไม่พอ
    counts = df["code"].value_counts()
    rare = set(counts[counts < MIN_IMAGES_FOR_VAL].index)
    eligible = ~df["code"].isin(rare)

    # เลือก group สะสมจนได้ ~val_frac ของ "จำนวนภาพ" ไม่ใช่จำนวน group
    # (เกณฑ์กำหนด 80/20 ที่จำนวนข้อมูล ส่วน group แบ่งทั้งก้อนเพื่อกัน leakage)
    sizes = df[eligible]["group"].value_counts()
    order = pd.Series(sizes.index).sample(frac=1.0, random_state=seed).tolist()
    target = val_frac * len(df)
    val_groups, acc = set(), 0
    for g in order:
        if acc >= target:
            break
        val_groups.add(g)
        acc += sizes[g]

    df["split"] = np.where(eligible & df["group"].isin(val_groups), "val", "train")

    # คลาสที่บังเอิญไม่มี train เลย (group ทั้งหมดตกไป val) -> ดึงกลับ
    for code, sub in df.groupby("code"):
        if (sub["split"] == "train").sum() == 0:
            df.loc[sub.index, "split"] = "train"

    # กันพลาดที่แพงที่สุดของงานนี้: ฟอนต์เดียวกัน + คลาสเดียวกัน โผล่ทั้ง train และ val
    # ถ้าหลุด val acc จะสวยหลอกแล้วไปตกตอนเจอ test จริง
    #
    # เช็กที่ระดับ (คลาส, group) ไม่ใช่ group เดี่ยว ๆ เพราะการดึงคลาสหายากกลับเข้า train
    # ทำให้ group หนึ่งมีได้ทั้งสองฝั่ง -- แต่คนละคลาสกัน ซึ่งไม่ใช่ leakage ของ label
    pairs = lambda sp: set(map(tuple, df[df.split == sp][["code", "group"]].to_numpy()))
    overlap = pairs("train") & pairs("val")
    assert not overlap, f"(คลาส, group) รั่วข้าม split: {sorted(overlap)[:5]}"
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="ThaiCharacter Dataset")
    ap.add_argument("--out", default="index.csv")
    ap.add_argument("--val-frac", type=float, default=0.20)  # เกณฑ์กำหนด 80/20
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    print(f"สแกน {root} ...")
    df = scan(root)
    df = assign_split(df, args.val_frac, args.seed)

    # label 0..N-1 เรียงตามรหัส TIS-620 เพื่อให้ index คงที่ทุกครั้ง
    codes = sorted(df["code"].unique())
    df["label"] = df["code"].map({c: i for i, c in enumerate(codes)})

    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    tr, va = (df["split"] == "train").sum(), (df["split"] == "val").sum()
    print(f"\nรวม {len(df):,} ภาพ | {len(codes)} คลาส | {df['group'].nunique()} groups")
    print(f"train {tr:,} ({tr / len(df):.1%}) | val {va:,} ({va / len(df):.1%})")
    print(f"ขนาดภาพ: w {df.width.min()}-{df.width.max()}  h {df.height.min()}-{df.height.max()}")
    print(f"w/h ratio: {(df.width / df.height).min():.2f} - {(df.width / df.height).max():.2f}")

    no_val = [tis620_char(c) for c in codes if ((df.code == c) & (df.split == "val")).sum() == 0]
    if no_val:
        print(f"\n! {len(no_val)} คลาสไม่มี val (รูปน้อยเกิน): {' '.join(no_val)}")
        print("  -> macro-F1 บน val จะไม่ครอบคลุมคลาสพวกนี้ ต้องพึ่ง synthetic data (Phase 5)")

    worst = Counter(dict(df["char"].value_counts())).most_common()
    print(f"\nคลาสใหญ่สุด: {worst[0][0]}={worst[0][1]:,}  เล็กสุด: {worst[-1][0]}={worst[-1][1]}")
    print(f"อัตราส่วน imbalance = {worst[0][1] / worst[-1][1]:,.0f} : 1")
    print(f"\nเขียน {args.out} แล้ว")


if __name__ == "__main__":
    main()
