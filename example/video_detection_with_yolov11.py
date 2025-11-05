from Oculus.Yolo.V11 import YOLOv11Detector
from Oculus.utils import YOLOVideoProcess
# VIDEO_FILE = "assets/videos/example.mp4"  
video_path = "assets/videos/example.mp4"
output_path = "output_video.mp4"
detector = YOLOv11Detector(
    "assets\model\yolo11s.onnx",
    label_yaml="assets\label\coco8.yaml",
)
video_processor = YOLOVideoProcess(detector)
<<<<<<< HEAD
# video_processor.process_video(video_path, output_path)
video_processor.process_video(video_path, output_path, 
                       use_threading=True)  
=======
video_processor.process_webcam()  
>>>>>>> 8db80b8 (feat:adding feature video processing and realtime-webcam)
