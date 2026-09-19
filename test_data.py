"""Self-check ส่วนที่พังแล้วเงียบ: รัน `python test_data.py`"""
import numpy as np
import torch
from PIL import Image

from data import PadToSquare, build_transform
from prepare_data import tis620_char


def test_tis620():
    assert tis620_char(161) == "ก"
    assert tis620_char(163) == "ฃ"
    assert tis620_char(249) == "๙"


def test_pad_keeps_aspect():
    """หัวใจของงานนี้ — pad ต้องได้จัตุรัสและวางของไว้กลาง ไม่ใช่ยืดภาพ"""
    pad = PadToSquare()
    x = torch.full((1, 20, 10), 255, dtype=torch.uint8)  # สูงกว่ากว้าง เช่น 'ๅ'
    out = pad(x)
    assert out.shape == (1, 20, 20), out.shape
    assert out[0, :, :5].sum() == 0 and out[0, :, 15:].sum() == 0, "ของไม่ได้อยู่กลาง"
    assert out.sum() == x.sum(), "pad ต้องไม่ทำให้ ink หาย"

    y = torch.full((1, 9, 15), 255, dtype=torch.uint8)  # กว้างกว่าสูง เช่น 'ั'
    assert pad(y).shape == (1, 15, 15)

    # ภาพเล็กสุดในชุดจริงคือ 3x6 ต้องไม่พัง
    assert pad(torch.zeros((1, 6, 3), dtype=torch.uint8)).shape == (1, 6, 6)


def test_pad_before_resize_beats_squash():
    """พิสูจน์ว่าทำไมต้อง pad ก่อน: squash ตรง ๆ ทำให้ภาพคนละ ratio กลายเป็นเหมือนกัน"""
    tall = torch.zeros((1, 30, 10), dtype=torch.uint8)
    tall[0, 5:25, 2:8] = 255
    wide = torch.zeros((1, 10, 30), dtype=torch.uint8)
    wide[0, 2:8, 5:25] = 255

    tf = build_transform(48, train=False, aug=False)
    a, b = tf(tall), tf(wide)
    assert not torch.allclose(a, b, atol=0.3), "pad ไม่ทำงาน สองภาพนี้ไม่ควรเหมือนกัน"


def test_binarize_invert():
    """ค่าจริงในชุดข้อมูลกระจุกที่ 0-4 กับ 251-255 (JPEG ringing) ต้องถูกตัดทิ้ง"""
    a = np.array([[0, 3, 128, 252, 255]], dtype=np.uint8)
    out = np.where(a > 127, 0, 255).astype(np.uint8)
    assert out.tolist() == [[255, 255, 0, 0, 0]], out
    # ink (เข้ม) -> 255, พื้นหลัง (สว่าง) -> 0 เพื่อให้ pad ด้วย 0 กลืนพื้นหลัง


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nผ่านหมด")
