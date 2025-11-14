import onnxruntime as ort
import cv2
import numpy as np
import os
import time
import psutil
import yaml

from typing import Tuple

class V8Detector:
    def __init__(self,onnx_model_path:str,class_path:str,
                 conf_thres:float=.7,
                 iou_thres:float=0.25,
                 optimize:bool=False):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # check model nya ada gk

        if not os.path.exists(onnx_model_path):
            raise ValueError("Model ONNX nya gk ada ")
        if not os.path.exists(class_path):
            raise ValueError("class file nya tidak ada")
        with open(class_path,'r',encoding="utf-8") as f:
            data = yaml.safe_load(f)
            # check apakah ini tipe data dict
            if isinstance(data, dict) and "names" in data:
                self.class_names = data["names"]
            else:
                self.class_names = data
        opts = ort.SessionOptions()
        if optimize:
            logical_core = psutil.cpu_count(logical=True)
            opts.intra_op_num_threads = max(1,logical_core//2)
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.INPUT_H = input_shape[2]
        self.INPUT_W = input_shape[3]

    def __letterbox(self, img: np.ndarray, new_shape=(640, 640)):
        color = (114, 114, 114)
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)
        return img_padded, (r, r), (dw, dh)
    

    def __inference(self, input_tensor: np.ndarray):
        start = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        print(f"Inference Time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return outputs
    

    def __preprocess(self, image: np.ndarray) -> Tuple:
        img_letterbox, ratio, dwdh = self.__letterbox(
            image, (self.INPUT_H, self.INPUT_W)
        )
        img = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0
        return img, ratio, dwdh
    
    def __compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area
        return intersection_area / np.maximum(union_area, 1e-6)


    def __nms(self, boxes, scores, iou_thres):
        # sorted list of prediction score
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []
        while sorted_indices.size > 0:
            curr = sorted_indices[0]
            keep_boxes.append(curr)
            if sorted_indices.size == 1:
                break
            ious = self.__compute_iou(boxes[curr, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_thres)[0]
            sorted_indices = sorted_indices[keep_indices + 1]
        return keep_boxes
    
    def __xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
        
    def __postprocess(self, outputs, orig_h, orig_w, ratio, dwdh):
        preds = outputs[0]

        # --- Handle various ONNX output shapes ---
        if preds.ndim == 3:
            # (1,84,6300) or (1,84,8400)
            if preds.shape[1] == 84:
                preds = preds.transpose(0, 2, 1)
            preds = preds[0]
        elif preds.ndim == 2 and preds.shape[0] == 84:
            preds = preds.transpose(1, 0)

        print("DEBUG shape:", preds.shape)

        boxes = preds[:, :4]
        scores_all = preds[:, 4:]
        scores = np.max(scores_all, axis=1)
        class_ids = np.argmax(scores_all, axis=1)

        mask = scores > self.conf_threshold
        boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return [], [], []

        boxes = self.__xywh2xyxy(boxes)
        boxes[:, [0, 2]] -= dwdh[0]
        boxes[:, [1, 3]] -= dwdh[1]
        boxes[:, :4] /= ratio[0]

        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

        keep = self.__nms(boxes, scores, self.iou_threshold)
        boxes, scores, class_ids = boxes[keep].astype(int), scores[keep], class_ids[keep]

        return boxes, scores, class_ids
    
    def detect(self, img_input):
        # Bisa menerima path string atau ndarray langsung
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"File gambar tidak ditemukan: {img_input}")
            if img_input.endswith((".jpg", ".jpeg", ".png")):
                image = cv2.imread(img_input)
                orig_h, orig_w = image.shape[:2]
                img, ratio, dwdh = self.__preprocess(image)
                outputs = self.__inference(img)
                boxes, scores, class_ids = self.__postprocess(outputs, orig_h, orig_w, ratio, dwdh)
                return boxes, scores, class_ids
            elif img_input.endswith((".mp4", ".avi", ".mov", ".mkv")):
                # coming soon
                pass
        else:
            raise ValueError("parameter harus bertipe data str")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    detector = V8Detector(
        "assets/models/yolov8m.onnx",
        "assets/labels/coco8.yaml",
    )

    image = cv2.imread("assets/images/Gu66kEFb0AEalKW.jpeg")
    boxes, scores, class_ids = detector.detect(image)

    for box, score, cls in zip(boxes, scores, class_ids):
        cls_id = int(cls)
        label_name = detector.class_names.get(cls_id, f"id:{cls_id}")
        label = f"{label_name} {score:.2f}"
        cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), 2)
        cv2.putText(image, label, (box[0], box[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()