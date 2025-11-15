import onnxruntime as ort
import numpy as np
import json
import os
import cv2
import time
from typing import Union,Tuple

class SSD300_VGG16_Detection:
    def __init__(self,onnx_file_path:str,label_file_path:str,
                 conf_thresh: float = 0.4,nms_thresh: float = 0.45,
                 input_size:Union[int,Tuple[int]]=300,device:str='cpu'):
        # check kalau label file path
        if not os.path.exists(label_file_path) and label_file_path.endswith('.json'):
            raise FileNotFoundError("File label nya tidak ditemukan")
        if not os.path.exists(onnx_file_path):
            raise FileNotFoundError(f"File ONNX tidak ditemukan")
        
        with open(label_file_path,'r',encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data,dict) and "categories" in data:
                cats = data['categories']
            else:
                cats = data
                # for SSD,class index = array index, not id
            self.classes = [c['name'] for c in sorted(cats,key=lambda x:x.get("id",0))]
            
            # setup coenffetions threshold
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size

        # oyy
        self.opt = ort.SessionOptions()
        self.sess = ort.InferenceSession(onnx_file_path, 
                                         sess_options=self.opt,
                                         providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.priors = self.__generate_priors()
    def __preprocess(self,img_arr:np.ndarray):
        H, W = img_arr.shape[:2]
        r = self.input_size / max(H, W)
        new_w, new_h = int(W * r), int(H * r)
        resized = cv2.resize(img_arr, (new_w, new_h))

        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        top = (self.input_size - new_h) // 2
        left = (self.input_size - new_w) // 2
        canvas[top:top+new_h, left:left+new_w, :] = resized

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[None]
        return img, r, top, left
    
    def __softmax(self,x:np.ndarray):
        x = x - np.max(x,axis=1,keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-9)
    def __generate_priors(self):
        fmap_dims = [38, 19, 10, 5, 3, 1]
        aspect_ratios = [
            [1, 2, 0.5],
            [1, 2, 3, 0.5, 1/3],
            [1, 2, 3, 0.5, 1/3],
            [1, 2, 3, 0.5, 1/3],
            [1, 2, 0.5],
            [1, 2, 0.5]
        ]
        min_scale, max_scale = 0.2, 0.9
        num_layers = len(fmap_dims)
        scales = [min_scale + (max_scale - min_scale) * k / (num_layers - 1) for k in range(num_layers)]
        priors = []
        for k, fmap in enumerate(fmap_dims):
            sk = scales[k]
            sk_next = scales[k+1] if k+1 < num_layers else 1.0
            for i in range(fmap):
                for j in range(fmap):
                    cx = (j + 0.5) / fmap
                    cy = (i + 0.5) / fmap
                    # square
                    priors.append([cx, cy, sk, sk])
                    # scale prime
                    s_prime = np.sqrt(sk * sk_next)
                    priors.append([cx, cy, s_prime, s_prime])
                    # aspect ratio boxes
                    for ar in aspect_ratios[k]:
                        if abs(ar - 1) < 1e-6:
                            continue
                        w = sk * np.sqrt(ar)
                        h = sk / np.sqrt(ar)
                        priors.append([cx, cy, w, h])
        priors = np.array(priors, dtype=np.float32)
        priors = np.clip(priors, 0, 1)
        assert priors.shape[0] == 8732, f"Expected 8732 priors, got {priors.shape[0]}"
        return priors

    def __decode_boxes(self,loc):
        pri = self.priors
        center_variance = 0.1
        size_variance = 0.2
        cx = pri[:, 0] + loc[:, 0] * center_variance * pri[:, 2]
        cy = pri[:, 1] + loc[:, 1] * center_variance * pri[:, 3]
        w = pri[:, 2] * np.exp(loc[:, 2] * size_variance)
        h = pri[:, 3] * np.exp(loc[:, 3] * size_variance)
        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2
        return np.stack([xmin, ymin, xmax, ymax], axis=1) * self.input_size

    def __nms(self, boxes, scores):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes.T
        area = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (area[i] + area[order[1:]] - inter + 1e-9)
            order = order[np.where(iou <= self.nms_thresh)[0] + 1]
        return keep

    def __inference(self, inp):
        start = time.perf_counter()
        outputs = self.sess.run(None, {self.input_name: inp})
        loc_out = outputs[0]  # (200, 4)
        conf_out = outputs[1] # (200,)

        # VGNet single-class detection: ubah ke 2D (background vs object)
        if conf_out.ndim == 1:
            conf_out = np.stack([1 - conf_out, conf_out], axis=1)

        loc = loc_out
        conf = self.__softmax(conf_out)
        print(f"Inference Time: {(time.perf_counter() - start) * 1000:.2f} ms")
        return conf, loc

    def detect(self, img_path: str) -> list:
        img = cv2.imread(img_path)
        H, W = img.shape[:2]
        
        inp, r, top, left = self.__preprocess(img)
        conf, loc = self.__inference(inp)
        
        # kalau boxes sudah sesuai koordinat asli, bisa langsung pakai
        
        # scale boxes ke ukuran gambar jika perlu
        boxes = loc.copy()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r

        results = []
        num_classes = conf.shape[1]
        for cls in range(num_classes):
            scores = conf[:, cls]
            inds = np.where(scores > self.conf_thresh)[0]
            if len(inds) == 0: continue
            keep = self.__nms(boxes[inds],scores[inds])
            for k in keep:
                i = inds[k]
                results.append({
                    "box": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "class_id": cls,
                    "label": self.classes[cls] if cls < len(self.classes) else f"class_{cls}"
                })
        return sorted(results, key=lambda x: x["score"], reverse=True)
        
if __name__ == "__main__":
    model_path = "assets/models/ssd300_Vgg16.onnx"
    label_path = "assets/labels/coco.json"
    img_path = "assets/images/Gu66kEFb0AEalKW.jpeg"

    detector = SSD300_VGG16_Detection(model_path, label_path, conf_thresh=0.72,nms_thresh=0.01)
    detections = detector.detect(img_path)
    # print("Detections:", detections)
    img = cv2.imread(img_path)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f'{det["label"]}:{det["score"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()