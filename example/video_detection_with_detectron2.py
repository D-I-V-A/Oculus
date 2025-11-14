from Oculus.FasterRCNN.detectron2 import Detectron2Detector
from Oculus.utils import Detectron2VideoProcess
detector = Detectron2Detector(
    onnx_model_path=r"assets\models\detectron2_fasterrcnn.onnx",
    coco_json_path=r"assets\labels\coco.json",
    conf_thresh=0.5,
)

video_processor = Detectron2VideoProcess(detector)
video_processor.process_webcam(camera_id=0)