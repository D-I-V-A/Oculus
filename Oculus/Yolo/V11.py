import numpy as np
import onnxruntime as ort
import yaml
import cv2
import psutil
from typing import List, Tuple
import os

class YOLOv11Detector:
    def __init__(self, model_path: str, label_yaml: str,
                 conf_thresh: float = 0.25, iou_thresh: float = 0.7,
                 optimize: bool = True,apply_letterbox:bool=True):
        """
        YOLOv11 ONNX Runtime Detector (Vanilla Parser)
        ---------------------------------------------
        Args:
            model_path (str): path file model .onnx
            label_yaml (str): path file yaml berisi daftar kelas
            conf_thresh (float): ambang minimal confidence score
            iou_thresh (float): ambang minimal IoU untuk NMS
            optimize (bool): aktifkan ONNX SessionOptions ringan
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        # MODEL_PATH = Path(model_path)
        # check label jika ada
        if os.path.exists(label_yaml):
            if label_yaml.endswith('.yaml'):
                with open(label_yaml,'r',encoding="utf-8") as f:
                    self.classes = yaml.safe_load(f)['names']
            else:
                raise ValueError("File nya harus berupa yaml ya !")
        else:
            raise FileNotFoundError("File Yaml tidak ditemukan")
        # Load ONNX model
        opts = ort.SessionOptions()
        if optimize:
            logical_cores = psutil.cpu_count(logical=True)
            opts.intra_op_num_threads = max(1,logical_cores//2)
            # this for task-level parallelism
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if os.path.exists(model_path) and model_path.endswith('.onnx'):
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
        else:
            raise FileNotFoundError("File ONNX tidak ditemukan")
        # Ambil dimensi input (biasanya [1,3,640,640])
        input_shape = self.session.get_inputs()[0].shape
        self.INPUT_H = input_shape[2]
        self.INPUT_W = input_shape[3]

    def __letterbox(self, image:np.ndarray, new_shape=(640, 640),
                   color=(114, 114, 114))->Tuple[np.ndarray,Tuple,Tuple]:
        shape = image.shape[:2]  # [h, w]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = (r, r)
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img_padded, ratio, (dw, dh)

    def __preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple, Tuple]:
        img_letterbox, ratio, dwdh = self.__letterbox(image, (self.INPUT_H, self.INPUT_W))
        img = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img, ratio, dwdh

    def __nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        ==== this section ====
        Pure NumPy NMS """
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
    # this postprocess section of Yolo
    def __postprocess(self, outputs, orig_h, orig_w, ratio, dwdh):
        preds = outputs[0].transpose(0, 2, 1)[0]
        boxes_xywh, scores_all = preds[:, :4], preds[:, 4:]

        confidences = scores_all.max(axis=1)
        class_ids = scores_all.argmax(axis=1)
        mask = confidences > self.conf_thresh
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return [], [], []

        # xywh → xyxy
        boxes = np.copy(boxes_xywh)
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # pastikan tetap np.ndarray (float32)
        boxes = boxes.astype(np.float32)

        # scale to original
        boxes[:, [0, 2]] -= dwdh[0]
        boxes[:, [1, 3]] -= dwdh[1]
        boxes[:, :4] /= ratio[0]

        # clip ke batas gambar
        boxes = np.clip(boxes, 0, max(orig_w, orig_h))

        # 🔧 FIX: pastikan sebelum NMS tetap array, bukan list
        idxs = self.__nms(boxes, confidences, self.iou_thresh)

        # ambil hasil akhir, baru konversi ke list
        boxes = boxes[idxs].astype(int).tolist()
        scores = confidences[idxs].tolist()
        class_ids = class_ids[idxs].tolist()

        return boxes, scores, class_ids

    def detect(self, file_path: str):
        if os.path.exists(file_path):
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                image = cv2.imread(file_path)
                blob, ratio, dwdh = self.__preprocess(image)
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
                boxes, scores, class_ids = self.__postprocess(outputs, image.shape[0], image.shape[1], ratio, dwdh)
                return boxes, scores, class_ids
            # jika ini berupa video
            elif file_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise ValueError("Gagal membuka file video")
                while True:
                    ret,frame = cap.read()
                    if not ret :
                        break
                    blob, ratio, dwdh = self.__preprocess(frame)
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
                    boxes, scores, class_ids = self.__postprocess(
                        outputs, frame.shape[0], frame.shape[1], ratio, dwdh
                    )

            else:
                raise ValueError("Format File tidak dapat dikenali")
        else:
            raise FileNotFoundError("File path tidak dapat di temukan")
        
