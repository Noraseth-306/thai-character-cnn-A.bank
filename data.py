"""Dataset + transforms.

ภาพในชุดนี้เป็น bilevel (ค่ากระจุกที่ 0-4 กับ 251-255) ที่ผ่าน JPEG มา
-> binarize ทิ้ง ringing artifact แล้ว invert ให้ ink=255, พื้นหลัง=0
   (พื้นหลัง=0 ทำให้ padding ด้วย 0 กลืนกับพื้นหลังพอดี)

ภาพเล็กมาก (3x6 ถึง 36x53 px) รวมกันแค่ ~27MB -> cache ลง RAM ทั้งหมด
ไม่ต้องยุ่งกับ DataLoader workers บน Windows เลย
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


def class_names(index_csv) -> list[str]:
    """label -> ตัวอักษร โดยอ่านจาก index.csv เต็ม

    ห้ามสร้างจาก subset (train/val) เด็ดขาด: 8 คลาสไม่มี val เลย
    ทำให้ index เลื่อนแล้วรายงานชื่อคลาสผิดแบบเงียบ ๆ
    """
    df = pd.read_csv(index_csv, encoding="utf-8-sig")
    return df.drop_duplicates("label").sort_values("label")["char"].tolist()


class PadToSquare(torch.nn.Module):
    """เติมขอบให้เป็นจัตุรัส *ก่อน* resize เพื่อคง aspect ratio

    w/h ต่อคลาสต่างกัน 0.51 (ๅ) ถึง 1.65 (ั) ถ้า resize เป็นจัตุรัสตรง ๆ
    ข้อมูลนี้หายหมด สระ/วรรณยุกต์จะปนกันทันที
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        s = max(h, w)
        left, top = (s - w) // 2, (s - h) // 2
        # pad ด้วย 0 = พื้นหลัง (เพราะ invert มาแล้ว)
        return torch.nn.functional.pad(img, (left, s - w - left, top, s - h - top), value=0)


class InkJitter(torch.nn.Module):
    """สุ่มทำเส้นหนา/บางลง 1 พิกเซล -- เลียนความเพี้ยนจริงของชุดข้อมูลนี้

    ภาพต้นฉบับเกิดจาก สแกน -> binarize ค่า threshold ที่ต่างกันเล็กน้อยทำให้
    เส้นหนาบางไม่เท่ากัน (วัดได้: ink fraction กระจาย 0.33-0.55)
    ต่างจาก affine (หมุน/เฉือน) ซึ่งวัดแล้วทำให้แย่ลง เพราะตัวอักษร ~20px
    ที่ต่างกันแค่จุดเดียว (ื/ึ, ั/้) เสียข้อมูลจำแนกไปตอน interpolate

    ทำที่ความละเอียดต้นฉบับก่อน resize เพราะ threshold จริงเกิดตอนนั้น
    """

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        r = torch.rand(()).item()
        if r >= 0.5:
            return img
        x = img.float().unsqueeze(0)
        if r < 0.25:  # dilate: เส้นหนาขึ้น
            x = torch.nn.functional.max_pool2d(x, 3, 1, 1)
        else:  # erode: เส้นบางลง
            x = -torch.nn.functional.max_pool2d(-x, 3, 1, 1)
        return x.squeeze(0).to(torch.uint8)


def build_transform(size: int, train: bool, aug: str) -> v2.Compose:
    # ห้าม RandomHorizontalFlip เด็ดขาดทุกโหมด:
    # อักษรไทยกลับด้านแล้วเปลี่ยนความหมาย (พ/ผ, ก/ภ) = สอนผิด
    pre = [InkJitter()] if train and aug == "domain" else []
    steps = pre + [PadToSquare(), v2.Resize((size, size), antialias=True)]
    if train and aug == "affine":
        steps.append(
            v2.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6, fill=0)
        )
    elif train and aug == "domain":
        # เลื่อนเล็กน้อยพอ เลียนความคลาดของการตัดกรอบตอน segment
        steps.append(v2.RandomAffine(degrees=0, translate=(0.04, 0.04), fill=0))
    steps += [v2.ToDtype(torch.float32, scale=True), v2.Normalize([0.5], [0.5])]
    return v2.Compose(steps)


class ThaiCharDataset(Dataset):
    def __init__(self, index_csv, root, split, size=48, aug="none"):
        df = pd.read_csv(index_csv, encoding="utf-8-sig")
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.labels = torch.tensor(self.df["label"].to_numpy(), dtype=torch.long)
        self.transform = build_transform(size, train=(split == "train"), aug=aug)

        root = str(root)
        self.images = []
        for p in self.df["path"]:
            a = np.asarray(Image.open(f"{root}/{p}").convert("L"))
            # binarize + invert: ink -> 255, bg -> 0
            self.images.append(np.where(a > 127, 0, 255).astype(np.uint8))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        img = torch.from_numpy(self.images[i]).unsqueeze(0)  # (1,H,W) uint8
        return self.transform(img), self.labels[i]

    @property
    def class_counts(self) -> np.ndarray:
        return np.bincount(self.df["label"].to_numpy())
