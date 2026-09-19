"""สร้างรูปประกอบการนำเสนอลงโฟลเดอร์ figures/"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# matplotlib ไม่มีฟอนต์ไทยมาให้ ต้องยืมของ Windows ไม่งั้นได้สี่เหลี่ยมเปล่า
THAI = "C:/Windows/Fonts/leelawad.ttf"
font_manager.fontManager.addfont(THAI)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=THAI).get_name()
plt.rcParams["figure.dpi"] = 130

OUT = Path("figures")
ROOT = "ThaiCharacter Dataset"


def fig_distribution(df: pd.DataFrame) -> None:
    n = df.char.value_counts()
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.bar(range(len(n)), n.values, color="#3b6ea5")
    ax.set_yscale("log")
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.index, fontsize=9)
    ax.set_xlim(-0.8, len(n) - 0.2)
    ax.set_ylabel("จำนวนภาพ (log scale)")
    ax.set_title(f"การกระจายข้อมูล {len(n)} คลาส — ไม่สมดุลสูงสุด {n.max():,} : {n.min()}")
    for i, (c, v) in enumerate(zip(n.index, n.values)):
        if v <= 15 or v >= n.max() * 0.6:
            ax.text(i, v * 1.15, str(v), ha="center", fontsize=7)
    ax.axhline(n.median(), ls="--", lw=0.8, color="crimson")
    ax.text(len(n) - 1, n.median() * 1.2, f"มัธยฐาน {int(n.median())}", ha="right",
            fontsize=8, color="crimson")
    fig.tight_layout()
    fig.savefig(OUT / "01_class_distribution.png")
    plt.close(fig)


def fig_samples(df: pd.DataFrame, per_class: int = 6) -> None:
    """ตัวอย่างข้อมูลทุกคลาส วาดตามอัตราส่วนจริงเพื่อให้เห็นว่า w/h ต่างกันจริง"""
    chars = df.drop_duplicates("label").sort_values("label")
    ncol, cell = 12, 46
    nrow = -(-len(chars) // ncol)
    sheet = Image.new("L", (ncol * (per_class * 22 + 30), nrow * cell), 255)
    for i, (_, r) in enumerate(chars.iterrows()):
        sub = df[df.char == r["char"]]
        sub = sub.sample(min(per_class, len(sub)), random_state=0)
        x0, y0 = (i % ncol) * (per_class * 22 + 30), (i // ncol) * cell
        for k, p in enumerate(sub.path):
            im = Image.open(f"{ROOT}/{p}").convert("L")
            im.thumbnail((20, cell - 12), Image.NEAREST)
            sheet.paste(im, (x0 + 28 + k * 22, y0 + 6))
        sheet.paste(Image.new("L", (1, cell), 220), (x0, y0))
    sheet.save(OUT / "02_samples_raw.png")

    fig, ax = plt.subplots(figsize=(15, nrow * 0.52))
    ax.imshow(np.asarray(sheet), cmap="gray", aspect="auto")
    ax.axis("off")
    for i, (_, r) in enumerate(chars.iterrows()):
        ax.text((i % ncol) * (per_class * 22 + 30) + 12, (i // ncol) * cell + cell * 0.62,
                r["char"], fontsize=11, color="#b00")
    ax.set_title("ตัวอย่างข้อมูลแต่ละคลาส (แสดงตามสัดส่วนภาพจริง)")
    fig.tight_layout()
    fig.savefig(OUT / "02_samples.png")
    plt.close(fig)


def fig_challenges(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].hist(df.height, bins=60, range=(0, 60), color="#3b6ea5")  # ตัด outlier 0.09% ออกจากแกน
    axes[0].set_title("ความสูงภาพ (px)")
    axes[0].axvline(df.height.median(), color="crimson", ls="--")
    axes[0].set_xlabel(f"มัธยฐาน {int(df.height.median())} px — เล็กมาก")

    ar = df.width / df.height
    axes[1].hist(ar, bins=60, color="#3b6ea5", range=(0, 3))
    axes[1].set_title("อัตราส่วน กว้าง/สูง")
    axes[1].set_xlabel("ต่างกัน 0.18–5.80 จึงห้ามบีบเป็นจัตุรัส")

    n = df.char.value_counts().values
    axes[2].plot(np.arange(1, len(n) + 1), np.sort(n)[::-1], "o-", ms=3, color="#3b6ea5")
    axes[2].set_yscale("log")
    axes[2].set_title("ความไม่สมดุลของคลาส")
    axes[2].set_xlabel(f"{n.max():,} : {n.min()}  ({(n < 20).sum()} คลาสมีน้อยกว่า 20 ภาพ)")
    fig.suptitle("ความท้าทายของชุดข้อมูล", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_challenges.png")
    plt.close(fig)


def fig_curves() -> None:
    runs = sorted(Path("runs").glob("*/history.csv"))
    if not runs:
        print("  (ยังไม่มีผลเทรน ข้ามกราฟ curve)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for f in runs:
        h = pd.read_csv(f)
        lbl = f.parent.name
        axes[0].plot(h.epoch, h.train_loss, label=f"{lbl} train")
        axes[0].plot(h.epoch, h.val_loss, "--", label=f"{lbl} val")
        axes[1].plot(h.epoch, h.val_acc, label=lbl)
        axes[2].plot(h.epoch, h.val_macro_f1, label=lbl)
    for a, t in zip(axes, ["Loss", "Validation Accuracy", "Validation macro-F1"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "04_training_curves.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv("index.csv", encoding="utf-8-sig")
    fig_distribution(df)
    print("  01_class_distribution.png")
    fig_samples(df)
    print("  02_samples.png")
    fig_challenges(df)
    print("  03_challenges.png")
    fig_curves()
    print("  04_training_curves.png")
    print(f"\nเสร็จ -> {OUT}/")


if __name__ == "__main__":
    main()
