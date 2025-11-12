import onnxruntime as ort
import cv2
import numpy as np
import os
import time
import psutil
import yaml


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

    def __letterbox(self, img: np.ndarray, new_shape=(480, 480)):
        color = (114,114,114)
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        img_resized = cv2.resize(img,new_unpad,interpolation=cv2.INTER_LINEAR)
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
    

    def __preprocess(self, image: np.ndarray):
        img_letterbox, ratio, dwdh = self.__letterbox(
            image, (self.INPUT_H, self.INPUT_W)
        )
        img = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
        return img, ratio, dwdh
    
    def __nms(self, boxes, scores, iou_thres):
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return keep
    
    def __postprocess(self, outputs, orig_h, orig_w, ratio, dwdh):
            preds = outputs[0]
            if preds.shape[1] > preds.shape[2]:
                preds = preds.transpose(0, 2, 1)
            preds = preds[0]

            boxes = preds[:, :4]
            scores_all = preds[:, 4:]
            scores = np.max(scores_all, axis=1)
            class_ids = np.argmax(scores_all, axis=1)

            mask = scores > self.conf_threshold
            boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

            if len(boxes) == 0:
                return [], [], []

            boxes_xyxy = np.copy(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

            boxes_xyxy[:, [0, 2]] -= dwdh[0]
            boxes_xyxy[:, [1, 3]] -= dwdh[1]
            boxes_xyxy[:, :4] /= ratio[0]
            boxes_xyxy = np.clip(boxes_xyxy, 0, max(orig_h, orig_w))

            keep = self.__nms(boxes_xyxy, scores, self.iou_threshold)
            boxes_xyxy = boxes_xyxy[keep].astype(int)
            scores = scores[keep]
            class_ids = class_ids[keep]

            return boxes_xyxy, scores, class_ids
    
    def detect(self, img_input):
        # Bisa menerima path string atau ndarray langsung
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"File gambar tidak ditemukan: {img_input}")
            image = cv2.imread(img_input)
        elif isinstance(img_input, np.ndarray):
            image = img_input
        else:
            raise TypeError("Input harus berupa path string atau numpy.ndarray")
        
        if image is None:
            raise ValueError("Gagal membaca gambar")

        orig_h, orig_w = image.shape[:2]
        img, ratio, dwdh = self.__preprocess(image)
        outputs = self.__inference(img)
        boxes, scores, class_ids = self.__postprocess(outputs, orig_h, orig_w, ratio, dwdh)
        return boxes, scores, class_ids


if __name__ == "__main__":
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
        cv2.imshow("Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()