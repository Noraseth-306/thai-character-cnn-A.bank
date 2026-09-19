"""เติมภาพสังเคราะห์เฉพาะคลาสที่ข้อมูลจริงน้อย -> index_mixed.csv

ต่างจาก pretrain ทั้งชุด (p5_*) ตรงที่ไม่แตะคลาสใหญ่เลย
คลาสใหญ่ยังเห็นแต่ข้อมูลจริง 100% จึงไม่มีทางเสีย distribution ฝั่งที่ test วัดจริง

val ไม่ถูกแตะต้อง -- มาจากข้อมูลจริงเท่านั้นเสมอ
path เก็บแบบอิงรากโปรเจกต์ เพราะสองแหล่งอยู่คนละโฟลเดอร์ (ใช้ --train-root .)
"""
import argparse
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--below", type=int, default=100, help="เติมคลาสที่มีภาพจริงน้อยกว่านี้")
    ap.add_argument("--cap", type=int, default=400, help="เติมจนมีรวมไม่เกินกี่ภาพ")
    ap.add_argument("--out", default="index_mixed.csv")
    args = ap.parse_args()

    real = pd.read_csv("index.csv", encoding="utf-8-sig")
    syn = pd.read_csv("synth_index.csv", encoding="utf-8-sig")

    tr = real[real.split == "train"].copy()
    tr["path"] = "ThaiCharacter Dataset/" + tr["path"]
    syn["path"] = "synth/" + syn["path"]

    n_train = tr.char.value_counts()
    need = {c: min(args.cap - n, args.cap) for c, n in n_train.items() if n < args.below}

    picked = []
    for ch, k in need.items():
        pool = syn[syn.char == ch]
        if pool.empty:
            print(f"  ! {ch}: ไม่มีภาพสังเคราะห์ ข้าม")
            continue
        picked.append(pool.sample(min(k, len(pool)), random_state=0))

    add = pd.concat(picked) if picked else syn.iloc[:0]
    add["split"] = "train"
    out = pd.concat([tr, add], ignore_index=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"คลาสที่เติม: {len(need)} คลาส (ภาพจริง < {args.below})")
    cmp = pd.DataFrame({"จริง": n_train, "เติม": add.char.value_counts()}).fillna(0).astype(int)
    cmp = cmp[cmp["เติม"] > 0].sort_values("จริง")
    print(cmp.head(15).to_string())
    print(f"\ntrain เดิม {len(tr):,} -> {len(out):,} ภาพ (+{len(add):,})")
    print(f"คลาสใหญ่ไม่ถูกแตะเลย | val ยังเป็นข้อมูลจริง 100%")


if __name__ == "__main__":
    main()
