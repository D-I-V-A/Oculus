# Export YOLO Model To ONNX

Panduan lengkap untuk mengekspor model YOLO (YOLOv5 / YOLOv8 / YOLOv9 / YOLOv11) ke format ONNX agar bisa digunakan dalam pengembangan di berbagai platform baik dalam website,backend dan desktop.

Berikut cara untuk export model dalam algoritma Yolo

----

## Export YOLOv8 / YOLOv9 / YOLOv11 (Ultralytics)

### a. Instalisasai
```bash
pip install ultralytics
```

### b. Export ke ONNX
```bash
from ultralytics import YOLO

# Load model (YOLOv8 / YOLOv9 / YOLOv11 .pt)
model = YOLO("<path file best.pt>")

# Export to ONNX
model.export(format="onnx",opset=21)
```

| Parameter       | Fungsi                                              |
| --------------- | --------------------------------------------------- |
| `opset=18`      | Menentukan opset ONNX (direkomendasikan opset ≥ 18) |


atau export juga bisa seperti ini

```bash
model = YOLO("<path file best.pt>")

# Export to ONNX
model.export(format="onnx", opset=21, dynamic=True, simplify=True)
```


| Parameter       | Fungsi                                              |
| --------------- | --------------------------------------------------- |
| `opset=18`      | Menentukan opset ONNX (direkomendasikan opset ≥ 18) |
| `dynamic=True`  | Mengaktifkan dynamic input shape                    |
| `simplify=True` | Menyederhanakan graph ONNX otomatis                 |



### Output

```bash
best.onnx
```