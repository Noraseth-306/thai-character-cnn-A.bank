# การจำแนกตัวอักษรไทยด้วยโครงข่ายประสาทแบบคอนโวลูชัน

โครงงานนี้พัฒนาโมเดลสำหรับจำแนกภาพตัวอักษรไทยแบบหนึ่งภาพต่อหนึ่งตัวอักษร จำนวน 72 คลาส โดยใช้ PyTorch เปรียบเทียบ Custom CNN กับโมเดล Transfer Learning ตระกูล ResNet และทดลองแก้ปัญหาจำนวนข้อมูลต่อคลาสไม่สมดุลด้วยข้อมูลตัวพิมพ์จาก PrintAksorn

ผลลัพธ์ที่ดีที่สุดด้าน Accuracy คือ ResNet-50 Transfer Learning ที่ **98.40%** บน Validation set ส่วนโมเดล ResNet-50 ที่เติมข้อมูลให้เฉพาะคลาสขาดแคลนได้ Accuracy **98.32%** และ Macro-F1 สูงสุดที่ **97.07%**

## สารบัญ

1. [โครงสร้างโครงการ](#โครงสร้างโครงการ)
2. [ชุดข้อมูล](#ชุดข้อมูล)
3. [การติดตั้ง](#การติดตั้ง)
4. [การรัน Pipeline คำสั่งเดียว](#การรัน-pipeline-คำสั่งเดียว)
5. [ขั้นตอนที่ 1 การสำรวจและทำความสะอาดข้อมูล](#ขั้นตอนที่-1-การสำรวจและทำความสะอาดข้อมูล)
6. [ขั้นตอนที่ 2 การแบ่ง Train และ Validation](#ขั้นตอนที่-2-การแบ่ง-train-และ-validation)
7. [ขั้นตอนที่ 3 การเตรียมภาพก่อนเข้าโมเดล](#ขั้นตอนที่-3-การเตรียมภาพก่อนเข้าโมเดล)
8. [ขั้นตอนที่ 4 โมเดลที่ใช้](#ขั้นตอนที่-4-โมเดลที่ใช้)
9. [ขั้นตอนที่ 5 การฝึกโมเดล](#ขั้นตอนที่-5-การฝึกโมเดล)
10. [ขั้นตอนที่ 6 การเติมข้อมูลให้คลาสขาดแคลน](#ขั้นตอนที่-6-การเติมข้อมูลให้คลาสขาดแคลน)
11. [ขั้นตอนที่ 7 การประเมินผล](#ขั้นตอนที่-7-การประเมินผล)
12. [ผลการทดลอง](#ผลการทดลอง)
13. [การเลือก Weight สำหรับใช้งาน](#การเลือก-weight-สำหรับใช้งาน)
14. [วิธีทำ Inference](#วิธีทำ-inference)
15. [การตรวจสอบ Overfitting](#การตรวจสอบ-overfitting)
16. [ข้อค้นพบจากการทดลอง](#ข้อค้นพบจากการทดลอง)
17. [วิธีทำซ้ำตั้งแต่ต้น](#วิธีทำซ้ำตั้งแต่ต้น)

## โครงสร้างโครงการ

```text
CNN-project/
├── README.md                     # เอกสารโครงการและวิธีใช้งานทั้งหมด
├── requirements.txt              # ไลบรารีที่ต้องติดตั้ง
├── scripts/                      # โปรแกรมแต่ละขั้น แยกเป็นไฟล์ Python
│   ├── train.py                  # ฝึกโมเดลและบันทึก Weight
│   ├── evaluate.py               # วัด Accuracy/Macro-F1 และวิเคราะห์ข้อผิดพลาด
│   ├── inference.py              # ทำนายภาพใหม่ด้วย Weight ที่ฝึกแล้ว
│   ├── pipeline.py               # เรียกทุกขั้นตอนแบบ end-to-end
│   ├── prepare_data.py           # Clean และแบ่ง Train/Validation
│   ├── synth.py                  # สร้างข้อมูลสังเคราะห์จาก PrintAksorn
│   └── mix_synth.py              # เติมข้อมูลเฉพาะคลาสที่มีภาพน้อย
├── src/thai_char_cnn/            # Dataset, preprocessing และโมเดล
├── tests/                        # ตรวจสอบ preprocessing
├── weights/                      # Weight หลักที่ฝึกเรียบร้อยแล้ว
│   └── resnet50_best.pt
├── results/                      # Metric และรายงานของ Weight หลัก
├── figures/                      # รูปผลลัพธ์สำหรับรายงาน
└── runs/                         # ผลทดลองในเครื่อง (ไม่อัปโหลดขึ้น Git)
```

หน้าแรกของ GitHub จึงแสดงเฉพาะเอกสารและโฟลเดอร์สำคัญ ส่วน Dataset, synthetic data, index และผลทดลองชั่วคราวถูกกันออกด้วย `.gitignore`

สำหรับ Inference ใช้ `scripts/inference.py`, โค้ดใน `src/thai_char_cnn/` และ `weights/resnet50_best.pt` โดย mapping ระหว่าง label กับอักขระไทยทั้ง 72 คลาสถูกบันทึกอยู่ใน Weight แล้ว

## ชุดข้อมูล

### ชุดข้อมูลหลัก

ชุดข้อมูลหลักอยู่ใน `ThaiCharacter Dataset` และจัดเก็บด้วยโครงสร้างดังนี้

```text
ThaiCharacter Dataset/
├── 161/        # ก
├── 162/        # ข
├── 163/        # ฃ
└── ...
```

ชื่อโฟลเดอร์เป็นรหัส TIS-620 ตัวอย่างเช่น

| รหัส | ตัวอักษร |
|---:|:---:|
| 161 | ก |
| 162 | ข |
| 163 | ฃ |
| 240 | ๐ |
| 249 | ๙ |

สถิติชุดข้อมูลหลักหลังกรองไฟล์ที่ไม่เกี่ยวข้อง

| รายการ | ค่า |
|---|---:|
| จำนวนภาพ | 62,707 |
| จำนวนคลาส | 72 |
| จำนวนกลุ่มเอกสารและรูปแบบ | 102 |
| คลาสใหญ่ที่สุด | า จำนวน 5,025 ภาพ |
| คลาสเล็กที่สุด | ฃ และ ฑ จำนวนคลาสละ 1 ภาพ |
| อัตราส่วนความไม่สมดุลสูงสุด | 5,025 ต่อ 1 |

ภาพต้นฉบับมีขนาดไม่คงที่ ตั้งแต่ภาพวรรณยุกต์ขนาดเล็กไม่กี่พิกเซลไปจนถึงภาพที่มีความสูงมากกว่า 50 พิกเซล ภาพส่วนใหญ่เป็น grayscale ที่ค่าพิกเซลกระจุกใกล้สีดำและสีขาว เนื่องจากผ่านกระบวนการสแกนและบีบอัด JPEG

### ชุดข้อมูล PrintAksorn

PrintAksorn เป็นข้อมูลตัวอักษรไทยที่สร้างจากฟอนต์ 406 รูปแบบ มี 95 อักขระ จำนวน 38,570 ภาพ ขนาด 224 x 224 พิกเซล ชุดนี้ครอบคลุม 71 จาก 72 คลาสของโครงการ โดยขาดเฉพาะ `ฤ`

โครงการนี้ไม่ได้นำ PrintAksorn ทั้งหมดมาผสมกับข้อมูลจริง แต่ใช้เฉพาะคลาสที่มีข้อมูลจริงใน Train ต่ำกว่า 100 ภาพ เพื่อลดผลกระทบต่อ distribution ของคลาสใหญ่

### ข้อมูล ALICE ที่ไม่ได้นำมาใช้

ได้ตรวจสอบชุดข้อมูลลายมือ ALICE เพิ่มเติมแล้ว พบว่าครอบคลุม 70 จาก 72 คลาสของโครงการ แต่โมเดลที่ฝึกจากข้อมูลตัวพิมพ์ทำนาย ALICE ได้ Accuracy ประมาณ 37.7% เท่านั้น แสดงว่ามี domain gap ระหว่างลายมือกับตัวพิมพ์สูง จึงไม่นำ ALICE มาผสมกับ Train หลัก

## การติดตั้ง

สภาพแวดล้อมที่ใช้พัฒนา

| Package | Version ที่ทดสอบ |
|---|---|
| Python | 3.13.7 |
| PyTorch | 2.14.0+cu126 |
| torchvision | 0.29.0+cu126 |
| pandas | 3.0.1 |
| NumPy | 2.4.2 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |

ติดตั้ง Package หลักสำหรับเครื่องที่มี NVIDIA GPU และรองรับ CUDA 12.6

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install pandas numpy pillow scikit-learn matplotlib
```

ตรวจสอบว่า PyTorch มองเห็น GPU

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

หากไม่มี GPU โค้ดจะเปลี่ยนไปใช้ CPU อัตโนมัติ แต่การฝึก ResNet-50 จะใช้เวลานานขึ้นมาก

## การรัน Pipeline คำสั่งเดียว

`scripts/pipeline.py` รวมขั้นตอนหลักทั้งหมดไว้ในคำสั่งเดียว โดยใช้โมเดล ResNet-50 และข้อมูลสังเคราะห์แบบเติมเฉพาะคลาสที่ขาดแคลน เส้นทางนี้จึงมีครบทั้ง Transfer Learning, Data Augmentation และแนวคิด Targeted Synthetic Top-up

```text
PREPARE
Clean data + group split ประมาณ 80:20
    ↓
SYNTH
ปรับ PrintAksorn ให้ใกล้ domain ภาพสแกนจริง
    ↓
MIX
เติมเฉพาะคลาสที่มีภาพจริงต่ำกว่า 100 ภาพ
    ↓
TRAIN
ResNet-50 ImageNet Transfer Learning + Fine-tuning
    ↓
EVALUATE
Accuracy + Macro-F1 + วิเคราะห์คู่คลาสที่สับสน
    ↓
SMOKE TEST
โหลด Weight และทำนายภาพจริงหนึ่งภาพ
```

### รัน Pipeline เต็ม

```powershell
python scripts/pipeline.py
```

ค่ามาตรฐานของ Pipeline

| Parameter | ค่า |
|---|---:|
| Seed | 42 |
| Train/Validation | ประมาณ 80:20 |
| Synthetic ต่อคลาส | 406 ภาพ |
| เกณฑ์คลาสข้อมูลน้อย | ต่ำกว่า 100 ภาพ |
| จำนวนสูงสุดหลังเติม | ประมาณ 400 ภาพต่อคลาส |
| Architecture | ResNet-50 |
| Input size | 96 x 96 |
| Batch size | 64 |
| Epochs | 15 |
| Freeze backbone | 3 epochs |
| Learning rate | 0.001 ก่อนลดระหว่าง Fine-tune |

Output หลักจะอยู่ที่

```text
index.csv
synth/
synth_index.csv
index_mixed.csv
runs/pipeline_mix_resnet50/best.pt
runs/pipeline_mix_resnet50/history.csv
runs/pipeline_mix_resnet50/report.json
pipeline_state.json
pipeline_summary.json
```

### ทดสอบ Pipeline แบบเร็ว

โหมด `--quick` ใช้ synthetic จำนวนน้อยและฝึกเพียง 1 epoch จุดประสงค์คือทดสอบว่าแต่ละขั้นเชื่อมต่อกันถูกต้อง ไม่ได้ใช้สร้าง Weight สำหรับส่งงาน

```powershell
python scripts/pipeline.py --quick
```

Quick pipeline ถูกทดสอบแล้วและผ่านครบทุกขั้น ตั้งแต่สร้าง index จนถึงโหลด Weight ทำนายภาพจริง

### ดูคำสั่งโดยไม่รัน

```powershell
python scripts/pipeline.py --dry-run
```

คำสั่งนี้เหมาะสำหรับตรวจ path และ parameter ก่อนเริ่มงานที่ใช้เวลานาน

### ทำงานต่อจาก Output เดิม

```powershell
python scripts/pipeline.py --resume
```

`--resume` จะข้ามขั้น Prepare, Synth, Mix หรือ Train เมื่อพบ Output ที่จำเป็นครบแล้ว ส่วน Evaluate และ Smoke Test จะรันใหม่เพื่อยืนยันว่า Weight ยังโหลดได้

### เริ่มหรือหยุดเฉพาะบางช่วง

```powershell
# เริ่มจาก Train โดยใช้ index ที่สร้างไว้แล้ว
python scripts/pipeline.py --start-at train

# ทำเฉพาะ Prepare ถึง Mix
python scripts/pipeline.py --stop-after mix

# ประเมิน Weight ที่มีอยู่แล้ว
python scripts/pipeline.py `
  --start-at evaluate `
  --model-dir runs/mix_resnet50
```

### Quality Gate

หลัง Train เสร็จ Pipeline จะอ่าน epoch เดียวกับ Weight ที่ถูกเลือกด้วย Macro-F1 และตรวจเกณฑ์เริ่มต้นดังนี้

- Validation Accuracy ต้องไม่น้อยกว่า 0.97
- Validation Macro-F1 ต้องไม่น้อยกว่า 0.95

หากไม่ผ่าน Pipeline จะจบด้วยสถานะผิดพลาดและไม่รายงานว่าโมเดลพร้อมใช้งาน สามารถเปลี่ยนเกณฑ์ได้ด้วย

```powershell
python scripts/pipeline.py --min-accuracy 0.98 --min-macro-f1 0.96
```

### Reproducibility

Pipeline ส่ง seed เดียวกันไปยังขั้น Prepare, Synth, Mix และ Train โดย `scripts/train.py` กำหนด seed ให้ Python, NumPy, PyTorch และ CUDA พร้อมปิด cuDNN benchmark และเปิด deterministic mode เท่าที่ backend รองรับ

ทุกขั้นบันทึกสถานะและระยะเวลาลง `pipeline_state.json` ส่วนผล Quality Gate บันทึกลง `pipeline_summary.json`

## ขั้นตอนที่ 1 การสำรวจและทำความสะอาดข้อมูล

ดำเนินการด้วย `scripts/prepare_data.py`

```powershell
python scripts/prepare_data.py
```

สคริปต์ทำงานตามลำดับดังนี้

1. อ่านเฉพาะโฟลเดอร์ที่ชื่อเป็นตัวเลขและเฉพาะไฟล์ `.jpg`
2. ไม่อ่านไฟล์ระบบ เช่น `.DS_Store`
3. ตรวจสอบชื่อไฟล์ด้วยรูปแบบที่กำหนดไว้
4. รองรับไฟล์ที่ขึ้นต้นด้วย `Copy of` โดยไม่สร้างกลุ่มข้อมูลปลอม
5. อ่านความกว้างและความสูงของทุกภาพ
6. แปลงรหัส TIS-620 จากชื่อโฟลเดอร์เป็นอักขระไทย
7. สร้าง label ต่อเนื่องตั้งแต่ 0 ถึง 71
8. สร้าง `index.csv` ที่ประกอบด้วยข้อมูลต่อไปนี้

| Field | ความหมาย |
|---|---|
| `path` | ตำแหน่งไฟล์เทียบกับโฟลเดอร์ Dataset |
| `code` | รหัส TIS-620 |
| `char` | ตัวอักษรไทย |
| `group` | กลุ่มเอกสารและรูปแบบจากชื่อไฟล์ |
| `width` | ความกว้างต้นฉบับ |
| `height` | ความสูงต้นฉบับ |
| `split` | `train` หรือ `val` |
| `label` | หมายเลขคลาส 0 ถึง 71 |

### เหตุผลที่ไม่ลบภาพขนาดเล็ก

พบภาพที่มีความกว้างหรือความสูงต่ำจำนวนประมาณ 2,435 ภาพ แต่เมื่อตรวจสอบพบว่าส่วนใหญ่เป็นสระและวรรณยุกต์ เช่น `ั ่ ้ ๊ ิ ุ ์` ซึ่งมีขนาดเล็กตามธรรมชาติ การลบด้วยเงื่อนไขขนาดจึงเท่ากับลบข้อมูลจริงของบางคลาส

ภาพที่มีขนาดหรืออัตราส่วนสุดโต่งจริงมีประมาณ 54 ภาพ หรือราว 0.09% ของข้อมูลทั้งหมด จำนวนนี้น้อยและบางภาพเป็นตัวอักษรแนวตั้งหรือแนวนอนที่ถูกต้อง จึงเก็บภาพทั้งหมดไว้แทนการกรองแบบเหมารวม

## ขั้นตอนที่ 2 การแบ่ง Train และ Validation

แบ่งข้อมูลเป็น

- Train 49,988 ภาพ หรือ 79.7%
- Validation 12,719 ภาพ หรือ 20.3%

สัดส่วนคลาดจาก 80:20 เล็กน้อยเนื่องจากแบ่งข้อมูลเป็นกลุ่ม ไม่ได้แยกทีละภาพ

### เหตุผลที่แบ่งตามกลุ่ม

ชื่อไฟล์สะท้อนแหล่งเอกสารและรูปแบบ เช่น `bc_001sg` หากสุ่มภาพทีละไฟล์ ตัวอักษรจากเอกสารหรือฟอนต์เดียวกันอาจอยู่ทั้ง Train และ Validation ทำให้โมเดลเห็นรูปแบบเดียวกันมาก่อนและได้ Accuracy สูงเกินจริง

โครงการจึงเลือกกลุ่มสะสมเข้า Validation จนได้ประมาณ 20% ของจำนวนภาพ พร้อมตรวจด้วย assertion ว่าคู่ `(class, group)` เดียวกันไม่ปรากฏข้าม split

คลาสที่มีภาพน้อยมากจะถูกเก็บไว้ใน Train เพื่อไม่ให้คลาสนั้นไม่มีตัวอย่างสำหรับเรียนรู้ ผลคือ Validation มี 65 จาก 72 คลาส ดังนั้น Accuracy และ Macro-F1 ที่รายงานไม่สามารถสะท้อน 7 คลาสที่ไม่มีตัวอย่างใน Validation ได้ครบถ้วน

## ขั้นตอนที่ 3 การเตรียมภาพก่อนเข้าโมเดล

ดำเนินการใน `src/thai_char_cnn/data.py` ทุกครั้งที่โหลดข้อมูล

### 3.1 แปลงเป็น Grayscale

ภาพทั้งหมดถูกอ่านเป็นภาพช่องสีเดียว เนื่องจากสีไม่ใช่ข้อมูลที่จำเป็นสำหรับการจำแนกตัวอักษร

### 3.2 Binarization

ใช้ threshold 127

```text
pixel > 127  -> background
pixel <= 127 -> ink
```

ขั้นตอนนี้ลดค่ารบกวนจาก JPEG ringing ซึ่งทำให้พิกเซลที่ควรเป็นดำหรือขาวมีค่าเบี่ยงเบนเล็กน้อย

### 3.3 กลับสี

กำหนดให้

- ตัวอักษรหรือหมึกมีค่า 255
- พื้นหลังมีค่า 0

การกำหนดพื้นหลังเป็น 0 ทำให้พื้นที่ที่เติมในขั้นตอน Pad มีค่าเดียวกับพื้นหลัง

### 3.4 Pad to Square

ภาพต้นฉบับมีอัตราส่วนกว้างต่อสูงต่างกันมาก หาก Resize เป็นสี่เหลี่ยมโดยตรง ตัวอักษรจะถูกยืดหรือบีบและสูญเสียรูปทรงสำคัญ

จึงดำเนินการดังนี้

```text
ภาพต้นฉบับ
    -> เติมพื้นหลังด้านที่สั้นกว่าให้เป็นสี่เหลี่ยม
    -> Resize เป็นขนาดที่โมเดลต้องการ
```

Custom CNN ใช้ 48 x 48 พิกเซล, ResNet-18 ใช้ 64 x 64 พิกเซล และ ResNet-50 ใช้ 96 x 96 พิกเซล การใช้ 96 แทน 224 ช่วยลดเวลาและหน่วยความจำ เนื่องจากภาพต้นฉบับมีรายละเอียดเพียงประมาณ 20 พิกเซลอยู่แล้ว

### 3.5 Normalize

แปลงเป็น float และ Normalize ด้วย mean 0.5 และ standard deviation 0.5 ทำให้ค่าพิกเซลอยู่ใกล้ช่วง -1 ถึง 1

### 3.6 Cache ภาพใน RAM

ภาพต้นฉบับมีขนาดเล็กและรวมกันใช้หน่วยความจำไม่มาก `ThaiCharDataset` จึงโหลดภาพเข้า RAM เพื่อไม่ต้องอ่านไฟล์จาก OneDrive ซ้ำในทุก epoch

## ขั้นตอนที่ 4 โมเดลที่ใช้

### 4.1 Custom CNN

โครงสร้างหลักใน `SmallCNN`

```text
Input 1 x 48 x 48
    -> Conv-BatchNorm-ReLU
    -> Conv-BatchNorm-ReLU
    -> MaxPool
    -> Conv Block 64 channels
    -> Conv Block 128 channels
    -> Adaptive Average Pooling
    -> Dropout
    -> Linear 128 -> 72 classes
```

จุดเด่น

- มีประมาณ 296,168 parameters
- ใช้ Batch Normalization ช่วยให้การฝึกเสถียร
- ใช้ Global Adaptive Average Pooling แทน Fully Connected Layer ขนาดใหญ่
- รับภาพหลายขนาดได้โดยไม่ต้องกำหนดจำนวน feature หลัง convolution แบบตายตัว

### 4.2 ResNet Transfer Learning

รองรับ ResNet-18, ResNet-34 และ ResNet-50 จาก `torchvision.models` โดยโหลด Weight ที่ฝึกจาก ImageNet

ภาพในโครงการเป็น grayscale 1 ช่อง แต่ ResNet รับภาพ RGB 3 ช่อง โค้ดจึงทำซ้ำช่องสีภายในโมเดลก่อนส่งเข้า backbone โดยไม่ทำสำเนาข้อมูลตั้งแต่ Dataset

Classification Head เดิมสำหรับ ImageNet 1,000 คลาสถูกแทนด้วย Linear Layer สำหรับ 72 คลาส

จุดเด่นของ ResNet คือ Residual Connection ซึ่งช่วยให้ gradient ไหลผ่านโครงข่ายลึกได้ดี ลดปัญหาโมเดลลึกแล้วฝึกยาก

## ขั้นตอนที่ 5 การฝึกโมเดล

ดำเนินการด้วย `scripts/train.py`

### Loss Function

ใช้ Cross Entropy Loss พร้อม `label_smoothing=0.1` เพื่อลดความมั่นใจเกินไปของโมเดลและช่วย regularization

### Optimizer

ใช้ AdamW โดยมี weight decay `1e-4`

### Learning Rate Scheduler

ใช้ Cosine Annealing เพื่อลด learning rate อย่างต่อเนื่องระหว่างการฝึก

### Mixed Precision

เมื่อมี CUDA จะใช้ Automatic Mixed Precision และ GradScaler ช่วยลดการใช้ VRAM และเพิ่มความเร็ว

### การบันทึก Weight

หลังทุก epoch จะคำนวณผลบน Validation set และบันทึก `best.pt` เมื่อ Macro-F1 ดีขึ้น ภายในไฟล์ประกอบด้วย

- `model`: น้ำหนักโมเดล
- `args`: ค่า parameter ที่ใช้ฝึก เช่น architecture และ image size
- `classes`: mapping label ไปเป็นอักขระทั้ง 72 คลาส

การฝัง `classes` และ `args` ทำให้ `scripts/inference.py` โหลด architecture, ขนาดภาพ และ mapping ได้จาก Weight โดยตรง

### การฝึก Transfer Learning สองระยะ

1. ระยะ Freeze backbone 3 epochsแรก เพื่อให้ Classification Head เรียนรู้ก่อน
2. ระยะ Fine-tune ปลดล็อก backbone ทั้งหมดและลด learning rate ลง 10 เท่า เพื่อไม่ให้ gradient ทำลาย Weight ที่ Pretrain มาอย่างรวดเร็ว

คำสั่งฝึก ResNet-50

```powershell
python scripts/train.py `
  --arch resnet50 `
  --size 96 `
  --bs 48 `
  --epochs 15 `
  --freeze-epochs 3 `
  --out runs/tl_resnet50
```

## ขั้นตอนที่ 6 การเติมข้อมูลให้คลาสขาดแคลน

ขั้นตอนนี้ใช้ PrintAksorn ไม่ใช้ ALICE

### 6.1 ปรับ PrintAksorn ให้ใกล้ข้อมูลจริง

ดำเนินการด้วย

```powershell
python scripts/synth.py
```

`scripts/synth.py` ทำงานดังนี้

1. อ่าน `master_file.csv` ของ PrintAksorn
2. เลือกเฉพาะตัวอักษรที่อยู่ใน 72 คลาสของโครงการ
3. Crop พื้นหลังให้ชิดตัวอักษร
4. สุ่มความสูงเป้าหมายจากความสูงจริงของคลาสเดียวกัน
5. สุ่มสัดส่วนพิกเซลหมึกจากภาพจริง
6. เลือก threshold ด้วย percentile เพื่อให้สัดส่วนหมึกใกล้เป้าหมาย
7. บันทึกเป็น JPEG คุณภาพ 60 ถึง 95 เพื่อจำลอง artifact ของข้อมูลจริง
8. สร้าง `synth_index.csv`

PrintAksorn ไม่มี `ฤ` จึง render เพิ่มจากฟอนต์ไทยใน Windows ที่ตรวจสอบแล้วว่ามี glyph จริง โดยเปรียบเทียบกับ missing-glyph box เพื่อไม่รับฟอนต์ที่แสดงเพียงสี่เหลี่ยมแทนตัวอักษร

ผลลัพธ์คือภาพที่ผ่านการปรับ domain จำนวน 28,572 ภาพ

### 6.2 เติมเฉพาะคลาสข้อมูลน้อย

ดำเนินการด้วย

```powershell
python scripts/mix_synth.py --below 100 --cap 400
```

หลักการ

- ตรวจจำนวนภาพจริงใน Train ของแต่ละคลาส
- เลือกเฉพาะคลาสที่มีน้อยกว่า 100 ภาพ
- เติมจนมีข้อมูลรวมไม่เกินประมาณ 400 ภาพต่อคลาส
- ไม่เติมคลาสใหญ่
- ไม่เปลี่ยน Validation set

ผลลัพธ์

| รายการ | ค่า |
|---|---:|
| คลาสที่เติม | 31 คลาส |
| ภาพที่เพิ่ม | 11,303 ภาพ |
| Train เดิม | 49,988 ภาพ |
| Train หลังเติม | 61,291 ภาพ |

ตัวอย่างจำนวนที่เติม

| ตัวอักษร | ภาพจริงใน Train | ภาพที่เติม |
|:---:|---:|---:|
| ฃ | 1 | 392 |
| ฑ | 1 | 390 |
| ฬ | 3 | 394 |
| ๗ | 4 | 390 |
| ฮ | 9 | 391 |
| ๖ | 10 | 390 |
| ฤ | 13 | 387 |
| ึ | 14 | 386 |

คำสั่งฝึก ResNet-50 ด้วยข้อมูลผสม

```powershell
python scripts/train.py `
  --train-index index_mixed.csv `
  --train-root . `
  --arch resnet50 `
  --size 96 `
  --bs 64 `
  --epochs 15 `
  --freeze-epochs 3 `
  --out runs/mix_resnet50
```

`--train-index` ใช้เลือกข้อมูล Train แบบผสม ส่วน Validation ยังอ่านจาก `index.csv` และ `ThaiCharacter Dataset` ตามเดิม จึงไม่มีภาพ PrintAksorn ปนในชุดประเมิน

## ขั้นตอนที่ 7 การประเมินผล

รายงานตัวชี้วัดสองค่า

### Accuracy

สัดส่วนภาพทั้งหมดที่ทำนายถูก เหมาะสำหรับดูประสิทธิภาพโดยรวม แต่ได้รับอิทธิพลสูงจากคลาสที่มีจำนวนมาก

### Macro-F1

คำนวณ F1 แยกแต่ละคลาสแล้วเฉลี่ยโดยให้น้ำหนักทุกคลาสเท่ากัน เหมาะกับข้อมูลที่ไม่สมดุล เพราะคลาสที่มี 1 ภาพและคลาสที่มี 5,000 ภาพมีน้ำหนักเท่ากันในการเฉลี่ย

โครงการเลือกบันทึก Weight ตาม Macro-F1 เพื่อไม่ให้โมเดลที่เก่งเฉพาะคลาสใหญ่ถูกเลือกเป็นโมเดลที่ดีที่สุด

สามารถวิเคราะห์คู่คลาสที่สับสนได้ด้วย

```powershell
python scripts/evaluate.py --ckpt weights/resnet50_best.pt
```

คู่ที่พบว่าสับสนบ่อย ได้แก่ `ๅ -> า`, `า -> ๅ`, `า -> ว`, `ั -> ้`, `บ -> น` และ `ด -> ต`

## ผลการทดลอง

| วิธี | Validation Accuracy | Macro-F1 | ข้อสรุป |
|---|---:|---:|---|
| ResNet-50 Transfer Learning | **98.40%** | 96.68% | Accuracy สูงสุด |
| ResNet-50 + Synthetic Top-up | 98.32% | **97.07%** | สมดุลรายคลาสดีที่สุด |
| ResNet-18 Transfer Learning | 98.31% | 95.78% | ใกล้ ResNet-50 แต่เร็วกว่า |
| Custom CNN Baseline | 98.29% | 95.82% | โมเดลเล็กแต่ Accuracy สูง |
| Synthetic Pretrain + Fine-tune | 98.27% | 96.24% | ช่วย Macro-F1 เล็กน้อย |
| Domain Augmentation | 97.64% | 85.37% | ทำลายรายละเอียดบางคลาส |
| Weighted Random Sampler | 97.35% | 96.32% | ช่วยคลาสเล็กแต่ Accuracy ลด |
| Class Weight แบบ 1/sqrt(n) | 97.26% | 94.22% | ใช้งานได้แต่ไม่ชนะ Baseline |
| Synthetic Pretrain เท่านั้น | 93.49% | 84.27% | ไม่เคยเห็นข้อมูลจริง |

### ผลของ Synthetic Top-up ต่อคลาสข้อมูลน้อย

- Recall เฉลี่ยของคลาสที่เติมเพิ่มจาก 94.48% เป็น 95.50%
- Recall เฉลี่ยของ 20 คลาสใหญ่เปลี่ยนจาก 98.65% เป็น 98.56%
- Macro-F1 เพิ่มจาก 96.68% เป็น 97.07%
- Accuracy ลดเพียง 0.08 percentage point

ผลดังกล่าวแสดงว่าการเติมเฉพาะคลาสที่ขาดช่วยคลาสเล็กโดยไม่สร้างความเสียหายเชิงระบบต่อคลาสใหญ่

### ความไม่แน่นอนของผล

Validation มี 12,719 ภาพ ช่วงความเชื่อมั่นโดยประมาณของ Accuracy แถว 98% อยู่ที่ประมาณ ±0.22 percentage point ความแตกต่างระหว่างโมเดลอันดับต้น ๆ จึงมีขนาดเล็กกว่าความไม่แน่นอนนี้ ไม่ควรสรุปว่าโมเดลใดเหนือกว่าอย่างมีนัยสำคัญจาก split เพียงครั้งเดียว

## การเลือก Weight สำหรับใช้งาน

### เน้น Accuracy

ใช้

```text
weights/resnet50_best.pt
```

- Validation Accuracy 98.40%
- Macro-F1 96.68%
- เหมาะเป็น Weight หลักสำหรับส่งงานที่จัดอันดับด้วย Accuracy

### เน้นความสมดุลทุกคลาส

ใช้

```text
runs/mix_resnet50/best.pt
```

- Validation Accuracy 98.32%
- Macro-F1 97.07%
- เหมาะเมื่อให้ความสำคัญกับคลาสข้อมูลน้อย

## วิธีทำ Inference

### ทำนายภาพเดียว

```powershell
python scripts/inference.py `
  --ckpt weights/resnet50_best.pt `
  --input "path/to/image.jpg"
```

ตัวอย่างผลลัพธ์

```text
bc_001sg_3_118.jpg -> ก (รหัส 161, ความมั่นใจ 0.909)
```

### ทำนายทั้งโฟลเดอร์

```powershell
python scripts/inference.py `
  --ckpt weights/resnet50_best.pt `
  --input "path/to/images" `
  --out result.csv
```

CSV ที่ได้ประกอบด้วย

| Column | ความหมาย |
|---|---|
| `filename` | ชื่อภาพ |
| `predicted_code` | รหัส TIS-620 ที่โมเดลทำนาย |
| `predicted_char` | ตัวอักษรไทยที่โมเดลทำนาย |
| `confidence` | ความน่าจะเป็นสูงสุดจาก Softmax |
| `true_code` | รหัสจริง หากชื่อโฟลเดอร์แม่เป็นตัวเลข |

หากภาพถูกจัดเป็น `<input>/<TIS-620 code>/*.jpg` สคริปต์จะคำนวณ Accuracy ให้โดยอัตโนมัติ

ขั้นตอน Inference ภายใน

1. โหลด Weight และอ่าน architecture, image size และ classes
2. เปิดภาพเป็น grayscale
3. Binarize และกลับสีให้เหมือนตอน Train
4. Pad เป็นสี่เหลี่ยมและ Resize
5. ส่งเข้าโมเดล
6. ใช้ Softmax คำนวณ confidence
7. แปลง label กลับเป็นอักขระไทยและรหัส TIS-620

## การตรวจสอบ Overfitting

สำหรับ ResNet-50 ที่ epoch 12

| Metric | Train | Validation |
|---|---:|---:|
| Accuracy | 99.82% | 98.40% |
| Loss | 0.7503 | 0.7980 |

ช่องว่าง Accuracy เท่ากับประมาณ 1.42 percentage points โมเดลเริ่มมีแนวโน้ม overfitting เล็กน้อยหลัง epoch 12 เนื่องจาก Training Accuracy เพิ่มต่อจนเกือบ 100% ขณะที่ Validation Loss ไม่ลดลงต่อ แต่ Validation Accuracy ไม่ได้ทรุดลงอย่างรุนแรง

Weight ที่บันทึกตาม Macro-F1 มาจาก epoch 12 ซึ่งอยู่ก่อน Validation Loss เริ่มเพิ่มชัดเจน จึงเป็นจุดหยุดที่เหมาะสม

## ข้อค้นพบจากการทดลอง

### Data Augmentation ไม่ได้ช่วยเสมอไป

Affine augmentation ลด Accuracy จาก 98.29% เป็นประมาณ 97.84% และ augmentation แบบทำเส้นหนา/บางลดเหลือ 97.64% ภาพตัวอักษรในชุดนี้มีขนาดเล็กมาก บางคลาสต่างกันเพียงหนึ่งจุดหรือไม่กี่พิกเซล การหมุน เฉือน หรือเปลี่ยนความหนาจึงอาจเปลี่ยนลักษณะจำแนกของคลาส

### Class Weight ที่แรงเกินไปทำให้โมเดลพัง

การใช้ weight แบบ `1/n` ทำให้คลาสที่มี 1 ภาพได้รับน้ำหนักมากกว่าคลาสใหญ่หลายพันเท่า ส่งผลให้โมเดลละเลยคลาสใหญ่และ Accuracy ลดลงอย่างรุนแรง การเปลี่ยนเป็น `1/sqrt(n)` ช่วยให้ฝึกได้ตามปกติ แต่ยังไม่ชนะ Baseline

โค้ดปัจจุบันป้องกันการเปิด Weighted Sampler และ Class Weight พร้อมกัน เพราะเป็นการแก้ imbalance ซ้ำสองชั้น

### โมเดลเล็กสามารถแข่งขันกับ ResNet-50 ได้

Custom CNN ที่มีประมาณ 296,000 parameters ได้ Accuracy 98.29% ใกล้กับ ResNet-50 ที่มีประมาณ 23.6 ล้าน parameters และได้ 98.40% สาเหตุคือข้อมูลเป็นภาพไบนารีขนาดเล็ก งานจึงไม่ต้องใช้ feature ด้านสีและ texture ที่ซับซ้อนเหมือน ImageNet

### ข้อมูลที่ตรง domain สำคัญกว่าจำนวนไฟล์

การเพิ่มข้อมูลจำนวนมากด้วย augmentation ไม่ได้ทำให้ผลดีขึ้น แต่การเติม PrintAksorn เฉพาะคลาสข้อมูลน้อยช่วย Macro-F1 เพราะแก้ปัญหาในตำแหน่งที่จำเป็นโดยไม่เปลี่ยน distribution ของคลาสใหญ่

## วิธีทำซ้ำตั้งแต่ต้น

### วิธี A ฝึก ResNet-50 ด้วยข้อมูลจริงเท่านั้น

```powershell
# 1. สร้าง index และแบ่งข้อมูลประมาณ 80:20
python scripts/prepare_data.py

# 2. ตรวจ preprocessing
python tests/test_data.py

# 3. ฝึก ResNet-50
python scripts/train.py `
  --arch resnet50 `
  --size 96 `
  --bs 48 `
  --epochs 15 `
  --freeze-epochs 3 `
  --out runs/tl_resnet50

# 4. วิเคราะห์ผล
python scripts/evaluate.py --ckpt weights/resnet50_best.pt

# 5. ทดลอง Inference
python scripts/inference.py `
  --ckpt weights/resnet50_best.pt `
  --input "ThaiCharacter Dataset/161"
```

### วิธี B ฝึก ResNet-50 พร้อมเติมคลาสข้อมูลน้อย

```powershell
# 1. สร้าง index ข้อมูลจริง
python scripts/prepare_data.py

# 2. ปรับ PrintAksorn ให้ใกล้ข้อมูลจริง
python scripts/synth.py

# 3. เติมเฉพาะคลาสที่มีข้อมูลจริงน้อยกว่า 100 ภาพ
python scripts/mix_synth.py --below 100 --cap 400

# 4. ฝึกด้วย Train แบบผสมและ Validation จริง
python scripts/train.py `
  --train-index index_mixed.csv `
  --train-root . `
  --arch resnet50 `
  --size 96 `
  --bs 64 `
  --epochs 15 `
  --freeze-epochs 3 `
  --out runs/mix_resnet50

# 5. ทำนายด้วย Weight ที่ได้
python scripts/inference.py `
  --ckpt runs/mix_resnet50/best.pt `
  --input "path/to/images" `
  --out result.csv
```

## สถานะงาน

- โค้ด Train พร้อมใช้งาน
- โค้ด Inference พร้อมใช้งาน
- Weight ฝึกเรียบร้อยแล้ว
- Mapping 72 คลาสฝังอยู่ใน Weight
- Accuracy สูงสุดที่วัดได้ 98.40%
- Macro-F1 สูงสุดที่วัดได้ 97.07%
- กำหนดส่ง 25 กันยายน 2569 ก่อนเวลา 16:00 น.

ผลของ Weight หลักอยู่ใน `results/resnet50_history.csv` และ `results/resnet50_report.json` ส่วนผลทดลองใหม่จะถูกสร้างใน `runs/<experiment>/`
