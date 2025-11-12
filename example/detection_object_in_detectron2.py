from Oculus.FasterRCNN.detectron2 import Detectron2Detector

detector = Detectron2Detector(
    onnx_model_path="assets\models\detectron2_fasterrcnn.onnx",
    coco_json_path="assets\labels\coco.json",
    conf_thresh=0.5,
)
boxes, scores, class_ids = detector.detect("assets\images\Gu66kEFb0AEalKW.jpeg")
print(f"boxes result:{boxes}")
print(f"scores result:{scores}")
print(f"class_ids result:{class_ids}")
