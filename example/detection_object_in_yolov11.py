import os

import cv2

from Oculus.Yolo.V11 import YOLOv11Detector

detector = YOLOv11Detector(
    model_path="assets/models/yolo11s.onnx", label_yaml="assets/labels/coco8.yaml"
)

image_path = "assets\images\Gu66kEFb0AEalKW.jpeg"


if not os.path.exists(image_path):
    raise FileNotFoundError(f"Gambar tidak ditemukan di: {image_path}")

boxes, scores, classes = detector.detect(image_path)
print(boxes, scores, classes)
