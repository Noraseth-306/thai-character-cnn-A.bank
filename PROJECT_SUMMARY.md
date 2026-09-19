# Thai Character Classification with CNN

เอกสารนี้สรุปสิ่งที่ดำเนินการไปแล้วในโครงงานจำแนกตัวอักษรไทยด้วย CNN รวมถึงการเตรียมข้อมูล การทดลองโมเดล ผลลัพธ์ และวิธีใช้งานโค้ด

## 1. ภาพรวมโครงงาน

- งาน: จำแนกภาพตัวอักษรไทยแบบภาพละ 1 ตัวอักษร
- จำนวนข้อมูลต้นฉบับ: **62,707 ภาพ**
- จำนวนคลาส: **72 คลาส**
- Framework: PyTorch
- อุปกรณ์ที่ใช้ฝึก: NVIDIA GeForce RTX 3050 Ti Laptop GPU
- การแบ่งข้อมูล:
  - Train: **49,988 ภาพ (79.7%)**
  - Validation: **12,719 ภาพ (20.3%)**
- การแบ่งข้อมูลทำตามกลุ่มเอกสาร/ฟอนต์ เพื่อลดโอกาสที่รูปแบบจากเอกสารเดียวกันจะรั่วไปอยู่ทั้ง Train และ Validation

> หมายเหตุ: ชุด 20% ถูกใช้สำหรับเลือก Weight และรายงานผล จึงเรียกว่า Validation set ไม่ใช่ Independent Test set

## 2. การเตรียมและทำความสะอาดข้อมูล

การเตรียมข้อมูลอยู่ใน `prepare_data.py` และ `data.py`

สิ่งที่ดำเนินการ:

1. สแกนเฉพาะโฟลเดอร์คลาสและไฟล์ภาพ `.jpg`
2. กรองไฟล์ระบบ เช่น `.DS_Store`
3. รองรับไฟล์ที่ขึ้นต้นด้วย `Copy of` โดยไม่สร้างกลุ่มข้อมูลใหม่ผิดพลาด
4. แปลงชื่อโฟลเดอร์ซึ่งเป็นรหัส TIS-620 ให้เป็นตัวอักษรไทย เช่น `161 = ก`
5. สร้าง label ต่อเนื่องตั้งแต่ 0 ถึง 71
6. แปลงภาพเป็น grayscale
7. Binarize ที่ threshold 127 เพื่อลด JPEG ringing และ noise ระหว่างค่าสีดำ/ขาว
8. กลับสีให้ตัวอักษรเป็น 255 และพื้นหลังเป็น 0
9. ใช้ `PadToSquare` เติมขอบก่อน Resize เพื่อรักษาอัตราส่วนของตัวอักษร
10. ไม่ลบภาพขนาดเล็ก เพราะส่วนใหญ่เป็นสระและวรรณยุกต์ที่มีขนาดเล็กตามธรรมชาติ เช่น `ั ่ ้ ิ ุ`

ข้อมูลมีความไม่สมดุลสูง โดยคลาสใหญ่ที่สุดมี 5,025 ภาพ แต่บางคลาสมีเพียง 1 ภาพ

## 3. โมเดลที่พัฒนา

### 3.1 Custom CNN Baseline

สร้าง CNN ขนาดเล็กด้วย PyTorch ประกอบด้วย:

- Convolution
- Batch Normalization
- ReLU
- Max Pooling
- Adaptive Average Pooling
- Dropout
- Fully Connected Layer สำหรับ 72 คลาส

โมเดลมีประมาณ **296,168 parameters** และได้ผลดังนี้:

- Validation Accuracy: **98.29%**
- Macro-F1: **95.82%**

### 3.2 Transfer Learning

ทดลองใช้โมเดลที่ฝึกบน ImageNet ได้แก่:

- ResNet-18
- ResNet-50

ขั้นตอนการฝึก:

1. เปลี่ยน Classification Head ให้รองรับ 72 คลาส
2. Freeze backbone ใน 3 epochs แรก
3. ฝึกเฉพาะ Classification Head ก่อน
4. Unfreeze backbone และลด learning rate ลง 10 เท่า
5. Fine-tune ทั้งโมเดลด้วยข้อมูลตัวอักษรไทย

ResNet-50 ให้ผล Accuracy สูงสุด:

- Validation Accuracy: **98.40%**
- Macro-F1: **96.68%**
- Weight: `runs/tl_resnet50/best.pt`

Training Accuracy ที่ epoch 12 เท่ากับ 99.82% ขณะที่ Validation Accuracy เท่ากับ 98.40% มีช่องว่างประมาณ 1.42% โมเดลเริ่มมีแนวโน้ม overfitting เล็กน้อยหลัง epoch 12 แต่ Validation Accuracy ไม่ได้ลดลงอย่างรุนแรง

## 4. การจัดการคลาสที่มีข้อมูลน้อย

ใช้ชุดข้อมูล PrintAksorn ซึ่งมีตัวอักษรไทยจากหลายฟอนต์ เพื่อเติมเฉพาะคลาสที่มีภาพจริงใน Train น้อยกว่า 100 ภาพ

ขั้นตอนอยู่ใน `synth.py` และ `mix_synth.py`

หลักการ:

1. เลือกเฉพาะคลาสที่มีข้อมูลจริงน้อยกว่า 100 ภาพ
2. เติมจนแต่ละคลาสมีข้อมูลรวมประมาณ 400 ภาพ
3. เติมทั้งหมด 31 คลาส จำนวน 11,303 ภาพ
4. ไม่เติมภาพสังเคราะห์ให้คลาสใหญ่
5. Validation set ยังคงเป็นข้อมูลจริง 100% ไม่มีภาพสังเคราะห์ปน

ตัวอย่าง:

| ตัวอักษร | ภาพจริงใน Train | ภาพที่เติม |
|---|---:|---:|
| ฃ | 1 | 392 |
| ฑ | 1 | 390 |
| ฬ | 3 | 394 |
| ๗ | 4 | 390 |
| ฮ | 9 | 391 |
| ๖ | 10 | 390 |
| ฤ | 13 | 387 |

ผลเมื่อใช้ร่วมกับ ResNet-50:

- Validation Accuracy: **98.32%**
- Macro-F1: **97.07%** ซึ่งสูงที่สุดจากการทดลองทั้งหมด
- Recall เฉลี่ยของคลาสที่เติมข้อมูลเพิ่มขึ้นจาก **94.48% เป็น 95.50%**
- Recall ของ 20 คลาสใหญ่เปลี่ยนเพียง **98.65% เป็น 98.56%**
- Weight: `runs/mix_resnet50/best.pt`

วิธีนี้ช่วยให้ผลลัพธ์ระหว่างคลาสสมดุลขึ้น โดยแทบไม่กระทบคลาสใหญ่

## 5. Data Augmentation ที่ทดลอง

ทดลองสองรูปแบบ:

1. Affine augmentation เช่น หมุน เลื่อน ย่อ-ขยาย และ shear
2. Domain-matched augmentation ด้วยการทำเส้นหนา/บางแบบ dilate/erode

ผลการทดลอง:

| วิธี | Validation Accuracy | Macro-F1 |
|---|---:|---:|
| ไม่ใช้ augmentation (Baseline) | 98.29% | 95.82% |
| Affine augmentation | 97.84% | 93.62% |
| Domain-matched augmentation | 97.64% | 85.37% |

Augmentation ทั้งสองแบบไม่ช่วยเพิ่มประสิทธิภาพ เนื่องจากภาพต้นฉบับมีขนาดเล็กมากและบางคลาสต่างกันเพียงไม่กี่พิกเซล การเปลี่ยนรูปหรือความหนาของเส้นจึงอาจทำลายลักษณะสำคัญของตัวอักษร

## 6. การทดลองแก้ Class Imbalance

ทดลองวิธีต่อไปนี้:

- Weighted Random Sampler
- Class-weighted Cross Entropy
- Synthetic top-up เฉพาะคลาสข้อมูลน้อย

ข้อค้นพบ:

- Weighted sampler ช่วย Macro-F1 แต่ทำให้ Accuracy ลดลง
- Class weight แบบ `1/n` รุนแรงเกินไปและทำให้โมเดลทิ้งคลาสใหญ่
- Class weight แบบ `1/sqrt(n)` ใช้งานได้ แต่ยังไม่ชนะ Baseline
- Synthetic top-up เฉพาะคลาสข้อมูลน้อยให้ผลดีที่สุดด้าน Macro-F1

## 7. สรุปผลการทดลอง

| โมเดล/วิธี | Validation Accuracy | Macro-F1 |
|---|---:|---:|
| ResNet-50 Transfer Learning | **98.40%** | 96.68% |
| ResNet-50 + Synthetic Top-up | 98.32% | **97.07%** |
| ResNet-18 Transfer Learning | 98.31% | 95.78% |
| Custom CNN Baseline | 98.29% | 95.82% |
| Synthetic Pretrain + Fine-tune | 98.27% | 96.24% |
| Weighted Random Sampler | 97.35% | 96.32% |

ความแตกต่างด้าน Accuracy ระหว่างโมเดลอันดับต้น ๆ มีขนาดเล็กกว่าช่วงความเชื่อมั่นโดยประมาณ ±0.22% จึงไม่ควรสรุปว่าต่างกันอย่างมีนัยสำคัญจากการแบ่งข้อมูลเพียงครั้งเดียว

## 8. โมเดลที่แนะนำให้ใช้งาน

เลือกตามวัตถุประสงค์:

- หากเน้น Accuracy สูงสุด: `runs/tl_resnet50/best.pt` — **98.40%**
- หากเน้นผลลัพธ์สมดุลทุกคลาส: `runs/mix_resnet50/best.pt` — Accuracy **98.32%**, Macro-F1 **97.07%**

สำหรับการส่งงานที่มีการจัดอันดับด้วย Accuracy แนะนำ `tl_resnet50/best.pt` เป็น Weight หลัก และเก็บ `mix_resnet50/best.pt` เป็น Weight สำรอง

## 9. ไฟล์สำคัญ

| ไฟล์ | หน้าที่ |
|---|---|
| `prepare_data.py` | สแกนข้อมูล สร้าง label และแบ่ง Train/Validation |
| `data.py` | Dataset, preprocessing และ augmentation |
| `model.py` | Custom CNN และโมเดล Transfer Learning |
| `train.py` | ฝึกโมเดลและบันทึก Weight ที่ดีที่สุด |
| `predict.py` | โหลด Weight และทำนายภาพใหม่ |
| `synth.py` | สร้างภาพสังเคราะห์ให้มีลักษณะใกล้ข้อมูลจริง |
| `mix_synth.py` | เติมภาพสังเคราะห์เฉพาะคลาสข้อมูลน้อย |
| `analyze.py` | วิเคราะห์ recall และคู่คลาสที่สับสน |
| `test_data.py` | ตรวจสอบ preprocessing และการรักษาอัตราส่วนภาพ |

## 10. วิธีใช้งาน

### 10.1 เตรียมดัชนีข้อมูลแบบ 80:20

```powershell
python prepare_data.py
```

### 10.2 ฝึก ResNet-50 ด้วย Transfer Learning

```powershell
python train.py --arch resnet50 --size 96 --bs 48 --epochs 15 --freeze-epochs 3 --out runs/tl_resnet50
```

### 10.3 สร้างและผสมข้อมูลสำหรับคลาสที่มีข้อมูลน้อย

```powershell
python synth.py
python mix_synth.py
```

### 10.4 ฝึก ResNet-50 ด้วยข้อมูลผสม

```powershell
python train.py --train-index index_mixed.csv --train-root . --arch resnet50 --size 96 --bs 64 --epochs 15 --freeze-epochs 3 --out runs/mix_resnet50
```

### 10.5 ทำนายภาพเดี่ยว

```powershell
python predict.py --ckpt runs/tl_resnet50/best.pt --input "path/to/image.jpg"
```

### 10.6 ทำนายภาพทั้งโฟลเดอร์และบันทึกเป็น CSV

```powershell
python predict.py --ckpt runs/tl_resnet50/best.pt --input "path/to/images" --out result.csv
```

ไฟล์ Weight ฝังรายชื่อทั้ง 72 คลาสไว้ภายในแล้ว จึงไม่ต้องใช้ `index.csv` ในขั้นตอน Inference

## 11. สถานะปัจจุบัน

- โค้ด Train พร้อมใช้งาน
- โค้ด Inference พร้อมใช้งานและทดสอบกับ Custom CNN และ ResNet-50 แล้ว
- มี Weight ที่ฝึกเรียบร้อยแล้ว
- Accuracy สูงสุดที่วัดได้: **98.40%**
- Macro-F1 สูงสุดที่วัดได้: **97.07%**
- กำหนดส่ง: **25 กันยายน 2569 ก่อนเวลา 16:00 น.**
