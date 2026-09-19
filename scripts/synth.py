"""Phase 5: สร้างข้อมูลสังเคราะห์จาก PrintAksorn ให้ "เหมือนของจริง"

ทำได้เพราะชุดข้อมูลนี้เป็น *ตัวพิมพ์* ไม่ใช่ลายมือ -- render ฟอนต์แล้วทำให้เสื่อม
ตามกระบวนการเดียวกับต้นฉบับ (สแกน -> ย่อ -> binarize -> JPEG) ได้เลย
วิธีนี้ใช้กับชุดลายมือ (Burapha-TH, ALICE-THI) ไม่ได้ เพราะลายมือสังเคราะห์ไม่ได้

หัวใจคือไม่เดาพารามิเตอร์ความเสื่อมเอง แต่ *สุ่มจากภาพจริงของคลาสเดียวกัน*:
  - ความสูงเป้าหมาย  <- สุ่มจากความสูงจริงของคลาสนั้น
  - ink fraction     <- สุ่มจากภาพจริง แล้วเลือก threshold ให้ได้ค่านั้นพอดี
distribution จึงตรงกันโดยการก่อสร้าง ไม่ต้องจูนมือ
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FALLBACK_GLYPH_SIZE = 128


def thai_system_fonts() -> list[str]:
    """ฟอนต์ไทยในเครื่อง ใช้กับคลาสที่ PrintAksorn ไม่มี (ฤ)

    ต้องเทียบกับ glyph ที่ไม่มีจริง ไม่งั้นฟอนต์ละตินที่วาดกล่อง tofu จะผ่านหมด
    """
    def render(ft, ch):
        im = Image.new("L", (FALLBACK_GLYPH_SIZE,) * 2, 255)
        ImageDraw.Draw(im).text((24, 16), ch, 0, font=ft)
        return np.asarray(im)

    out = []
    for f in sorted(Path("C:/Windows/Fonts").glob("*.tt[fc]")):
        try:
            ft = ImageFont.truetype(str(f), 64)
            a = render(ft, "ฤ")
            if a.min() > 200 or np.array_equal(a, render(ft, "\ue000")):
                continue
            out.append(str(f))
        except Exception:
            continue
    return out


def render_glyph(font_path: str, ch: str) -> np.ndarray:
    ft = ImageFont.truetype(font_path, 96)
    im = Image.new("L", (256, 256), 255)
    ImageDraw.Draw(im).text((48, 32), ch, 0, font=ft)
    return np.asarray(im)


def tight_crop(gray: np.ndarray) -> np.ndarray | None:
    """ภาพจริงถูก crop ชิดตัวอักษรพอดี (วัดได้ 1.000) -- ต้องทำให้เหมือนกัน"""
    ink = gray < 128
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    return gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def degrade(crop: np.ndarray, target_h: int, target_ink: float, rng) -> bytes | None:
    """ย่อ -> binarize ให้ได้ ink fraction ตามเป้า -> ใส่ JPEG artifact"""
    ch, cw = crop.shape
    tw = max(1, round(cw * target_h / ch))
    small = Image.fromarray(crop).resize((tw, target_h), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float32)

    # เลือก threshold จาก percentile -> ได้ ink fraction ตรงเป้าโดยไม่ต้องวนจูน
    thr = np.percentile(a, 100 * target_ink)
    binar = np.where(a <= thr, 0, 255).astype(np.uint8)  # ดำ = หมึก เหมือนต้นฉบับ
    if binar.min() == binar.max():
        return None

    buf = io.BytesIO()
    Image.fromarray(binar).save(buf, "JPEG", quality=int(rng.integers(60, 96)))
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index.csv")
    ap.add_argument("--real-root", default="ThaiCharacter Dataset")
    ap.add_argument("--pim", default="PrintAksorn_dataset")
    ap.add_argument("--out", default="synth")
    ap.add_argument("--index-out", default="synth_index.csv")
    ap.add_argument("--per-class", type=int, default=406)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    real = pd.read_csv(args.index, encoding="utf-8-sig")
    meta = pd.read_csv(f"{args.pim}/master_file.csv")

    # pool สถิติจริง: (ความสูง, ink) ต่อคลาส -- คลาสที่ตัวอย่างน้อยใช้ pool รวม
    print("วัดสถิติภาพจริง (ใช้เป็นเป้าของภาพสังเคราะห์) ...")
    sample = real.groupby("char", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 120), random_state=args.seed), include_groups=False
    )
    ink_by_char: dict[str, list[float]] = {}
    for ch, path in zip(real.loc[sample.index, "char"], sample["path"]):
        a = np.asarray(Image.open(f"{args.real_root}/{path}").convert("L"))
        ink_by_char.setdefault(ch, []).append(float((a < 128).mean()))
    global_ink = np.concatenate(list(ink_by_char.values()))
    global_h = real["height"].to_numpy()

    classes = real.drop_duplicates("label").sort_values("label")
    by_char = {c: g["filepath"].tolist() for c, g in meta.groupby("original_character")}
    fallback_fonts: list[str] = []

    out_root = Path(args.out)
    rows, skipped = [], 0
    for _, row in classes.iterrows():
        ch, code, label = row["char"], int(row["code"]), int(row["label"])
        (out_root / str(code)).mkdir(parents=True, exist_ok=True)

        srcs = by_char.get(ch, [])
        from_font = not srcs
        if from_font:
            if not fallback_fonts:
                print("  PrintAksorn ไม่มีคลาสนี้ -> หาฟอนต์ไทยในเครื่อง ...")
                fallback_fonts = thai_system_fonts()
            srcs = fallback_fonts
            print(f"  {ch} (code {code}): ใช้ฟอนต์ระบบ {len(srcs)} ตัว")

        # สุ่มเป้าจากภาพจริงของคลาสเดียวกัน ถ้ามีน้อยเกินไปก็ใช้ pool รวม
        hs = real.loc[real["char"] == ch, "height"].to_numpy()
        hs = hs if len(hs) >= 5 else global_h
        inks = np.asarray(ink_by_char.get(ch, []))
        inks = inks if len(inks) >= 5 else global_ink

        pick = rng.choice(len(srcs), size=args.per_class, replace=len(srcs) < args.per_class)
        for i, si in enumerate(pick):
            src = srcs[si]
            gray = render_glyph(src, ch) if from_font else np.asarray(
                Image.open(f"{args.pim}/{src}").convert("L")
            )
            crop = tight_crop(gray)
            if crop is None:
                skipped += 1
                continue
            h = int(np.clip(rng.choice(hs) * rng.uniform(0.92, 1.08), 6, 60))
            jpg = degrade(crop, h, float(rng.choice(inks)), rng)
            if jpg is None:
                skipped += 1
                continue

            name = Path(src).stem if from_font else Path(src).stem
            fp = out_root / str(code) / f"s{i:04d}_{name[:40]}.jpg"
            fp.write_bytes(jpg)
            with Image.open(fp) as im:
                w, hh = im.size
            rows.append(
                {"path": f"{code}/{fp.name}", "code": code, "char": ch,
                 "group": f"synth_{name[:30]}", "width": w, "height": hh,
                 "label": label, "split": "synth"}
            )
        print(f"  {ch} (code {code}): {args.per_class} ภาพ", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.index_out, index=False, encoding="utf-8-sig")
    print(f"\nสร้าง {len(df):,} ภาพ ({skipped} ข้าม) -> {out_root}/ + {args.index_out}")


if __name__ == "__main__":
    main()
