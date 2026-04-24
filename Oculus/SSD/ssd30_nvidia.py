import onnxruntime as ort
import numpy as np
import json
import os
import cv2
from typing import Union,Tuple

class SSD300_Nvidia_Detection:
    def __init__(self,onnx_model_path:str,class_json_file:str,
                 conf_thresh: float = 0.4,nms_thresh: float = 0.45,
                 input_size:int =300):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")
        if not os.path.exists(class_json_file):
            raise FileNotFoundError(f"class json not found: {class_json_file}")
    
        # load model json
        with open(class_json_file,"r",encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and "categories" in data:
                cats = data["categories"]
            else:
                cats = data
            self.classes = [c["name"] for c in sorted(cats, key=lambda x: x.get("id", 0))]
        
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
    
        # onnx session
        self.opt = ort.SessionOptions()
        self.sess = ort.InferenceSession(
            onnx_model_path,self.opt,providers=['CPUExecutionProvider']
        )
        self.input_name = self.sess.get_inputs()[0].name

        # priors
        self.priors = self.__generate_correct_ssd_priors()
        self.center_variance = 0.1
        self.size_variance = 0.2
    
    def __preprocess(self, img_arr:np.ndarray):
        img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = img.transpose(2, 0, 1)[None]
        return img
    
    def __inference(self,inp):
        outputs = self.sess.run(None,{self.input_name:inp})
        loc_out,conf_out = None,None
        for o in outputs:
            if o.shape[1] == 4:
                loc_out = o
            elif o.shape[1] > 4:
                conf_out = o

        if loc_out is None or conf_out is None:
            raise RuntimeError("Cannot determine which ONNX output is loc/conf")
        loc = loc_out[0].transpose(1, 0)      # (8732,4)
        conf = conf_out[0].transpose(1, 0)    # (8732,81)

        conf = self.__softmax(conf)
        return conf, loc
    @staticmethod
    def __softmax(x):
        x = x - np.max(x,axis=1,keepdims=True)
        e = np.exp(x)
        return e / (np.sum(e, axis=1, keepdims=True) + 1e-9)
    def __generate_correct_ssd_priors(self):
        fmap_dims = [38, 19, 10, 5, 3, 1] 
        aspect_ratios = [ [1, 2, 0.5],
                          [1, 2, 3, 0.5, 1/3],
                          [1, 2, 3, 0.5, 1/3], 
                          [1, 2, 3, 0.5, 1/3], 
                          [1, 2, 0.5], 
                          [1, 2, 0.5]
                         ]
        min_scale,max_scale = 0.2,0.9
        num_layers = len(fmap_dims)
        scales = [ min_scale + (max_scale - min_scale) * k / (num_layers - 1) for k in range(num_layers) ]
        priors = []
        for k,fmap in enumerate(fmap_dims):
            sk = scales[k]
            sk_next = scales[k+1] if k+1 < num_layers else 1.0
            for i in range(fmap):
                for j in range(fmap):
                    cx = (j + 0.5) / fmap
                    cy = (i + 0.5) / fmap

                    # 1. Square box
                    priors.append([cx,cy,sk,sk])
                    # sk 
                    s_prime = np.sqrt(sk * sk_next)
                    priors.append([cx, cy, s_prime, s_prime])

                    # ratio boxes
                    for ar in aspect_ratios[k]:
                        if abs(ar-1) < 1e-6:
                            continue
                        w = sk * np.sqrt(ar)
                        h = sk / np.sqrt(ar)
                        priors.append([cx, cy, w, h])
        priors = np.array(priors,dtype=np.float32)
        priors = np.clip(priors,0,1)
        # warning 
        assert priors.shape[0] == 8732, f"Expected 8732 priors, got {priors.shape[0]}"

        return priors
    def __decode_boxes(self,loc):
        pri = self.priors

        cx = pri[:, 0] + loc[:, 0] * self.center_variance * pri[:, 2]
        cy = pri[:, 1] + loc[:, 1] * self.center_variance * pri[:, 3]
        w = pri[:, 2] * np.exp(loc[:, 2] * self.size_variance)
        h = pri[:, 3] * np.exp(loc[:, 3] * self.size_variance)

        xmin = (cx - w / 2) * self.input_size
        ymin = (cy - h / 2) * self.input_size
        xmax = (cx + w / 2) * self.input_size
        ymax = (cy + h / 2) * self.input_size

        return np.stack([xmin, ymin, xmax, ymax], axis=1)
    
    def __nms(self,boxes,scores):
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

            remain = np.where(iou <= self.nms_thresh)[0]
            order = order[remain + 1]

        return keep
    
    def __postprocess(self,conf,boxes):
        results = []
        _, num_classes = conf.shape

        for cls in range(1, num_classes):
            scores = conf[:, cls]
            inds = np.where(scores > self.conf_thresh)[0]
            if len(inds) == 0:
                continue

            keep = self.__nms(boxes[inds], scores[inds])
            for k in keep:
                i = inds[k]
                bbox = boxes[i].tolist()

                results.append({
                    "box": bbox,
                    "score": float(scores[i]),
                    "class_id": cls,
                    "label": self.classes[cls] if cls < len(self.classes) else f"class_{cls}"
                })

        return sorted(results, key=lambda x: x["score"], reverse=True)
    

    def detect(self,image_path:str):
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
        
        H, W = img.shape[:2]

        inp = self.__preprocess(img)
        conf, loc = self.__inference(inp)
        boxes = self.__decode_boxes(loc)

        boxes[:, [0, 2]] *= (W / self.input_size)
        boxes[:, [1, 3]] *= (H / self.input_size)

        return self.__postprocess(conf, boxes)


if __name__ == "__main__":
    model_path = "assets/models/nvidia_ssd.onnx"
    label_path = "assets/labels/coco.json"
    img_path = "assets/images/Gu66kEFb0AEalKW.jpeg" 
    det = SSD300_Nvidia_Detection(model_path, label_path, conf_thresh=0.5) 
    results = det.detect(img_path) 
    print("Detections:", results) 
# det.visualize(img_path, results)