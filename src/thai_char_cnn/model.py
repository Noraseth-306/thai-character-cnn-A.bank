"""Phase 1 baseline: custom CNN ตาม tutorial บท "การสร้างโมเดลแบบกำหนดเอง" (nn.Module)"""
import torch.nn as nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """48x48 -> 24 -> 12 -> 6 -> GAP

    ใช้ GAP แทน Flatten+Linear ใหญ่ ๆ: พารามิเตอร์น้อยกว่ามาก
    และรับ input ขนาดอื่นได้โดยไม่ต้องแก้โค้ด
    """

    def __init__(self, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(_block(1, 32), _block(32, 64), _block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---- Phase 2: Transfer Learning ----------------------------------------
# ไม่มี pretrained backbone สำหรับอักษรไทยให้ใช้ (สำรวจ HuggingFace แล้วมีแต่
# VLM ระดับ 3B และ CRNN ที่อ่านทั้งบรรทัด) จึงใช้ ImageNet เป็นจุดตั้งต้น
# แล้วเทียบกับการ pretrain ด้วยข้อมูลสังเคราะห์ของเราเอง

_ZOO = {
    "resnet18": ("resnet18", "ResNet18_Weights", "IMAGENET1K_V1", "fc"),
    "resnet34": ("resnet34", "ResNet34_Weights", "IMAGENET1K_V1", "fc"),
    "resnet50": ("resnet50", "ResNet50_Weights", "IMAGENET1K_V2", "fc"),
    "vgg19": ("vgg19", "VGG19_Weights", "IMAGENET1K_V1", "classifier"),
}


class Pretrained(nn.Module):
    """หุ้มโมเดล torchvision ให้รับภาพ grayscale 1 ช่อง

    ทำซ้ำเป็น 3 ช่องในโมเดล ไม่ใช่ใน transform เพราะถ้าทำตั้งแต่ dataloader
    จะกินแรมและแบนด์วิดท์ 3 เท่าโดยไม่ได้ข้อมูลเพิ่มเลย
    """

    def __init__(self, arch: str, n_classes: int):
        super().__init__()
        import torchvision.models as tvm

        fn, wcls, wname, head = _ZOO[arch]
        self.net = getattr(tvm, fn)(weights=getattr(getattr(tvm, wcls), wname))
        # เก็บ "ตัวโมดูล" ของ head ไว้ ไม่ใช่ชื่อ:
        # classifier ของ VGG เป็น Sequential ที่มี Linear 4096 อยู่ด้วย
        # ถ้าอ้างด้วยชื่อ ตอน freeze จะปลดล็อกไป 119M พารามิเตอร์แทนที่จะเป็น 295K
        if head == "fc":
            self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)
            self.head = self.net.fc
        else:
            self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, n_classes)
            self.head = self.net.classifier[6]

    def forward(self, x):
        return self.net(x.repeat(1, 3, 1, 1))

    def set_backbone_frozen(self, frozen: bool) -> None:
        for p in self.net.parameters():
            p.requires_grad = not frozen
        for p in self.head.parameters():
            p.requires_grad = True


def build_model(arch: str, n_classes: int, dropout: float = 0.3) -> nn.Module:
    if arch == "smallcnn":
        return SmallCNN(n_classes, dropout)
    if arch not in _ZOO:
        raise SystemExit(f"ไม่รู้จัก --arch {arch} (มี: smallcnn, {', '.join(_ZOO)})")
    return Pretrained(arch, n_classes)
