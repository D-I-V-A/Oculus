import json
import os

import cv2
import numpy as np
import onnxruntime as ort
import psutil


class Detectron2Detector:
    def __init__(
        self,
        onnx_model_path: str,
        coco_json_path: str,
        conf_thresh: float = 0.5,
        optimization: bool = True,
    ):
        """ """
        self.classes = None
        self.conf_thresh = conf_thresh
        self.opts = ort.SessionOptions()
        # check file json with coco label
        if os.path.exists(coco_json_path):
            with open(coco_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)["categories"]
                self.classes = [c["name"] for c in sorted(data, key=lambda x: x["id"])]
        if optimization:
            logical_cores = psutil.cpu_count(logical=True)
            self.opts.intra_op_num_threads = max(1, logical_cores // 2)
            self.opts.inter_op_num_threads = 1
            self.opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            # and input in inferencSession

        # check model nya
        if os.path.exists(onnx_model_path):
            self.session = ort.InferenceSession(
                onnx_model_path,
                sess_options=self.opts,
                providers=["CPUExecutionProvider"],
            )
        else:
            raise FileNotFoundError("File ONNX tidak ditemukan")

        input_shape = self.session.get_inputs()[0].shape
        self.batch_size, self.channels, self.INPUT_H, self.INPUT_W = input_shape

    def __letterbox(
        self, image: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)
    ):
        """Resize image dengan aspect ratio tetap + padding"""
        if image is None:
            raise ValueError("Image is None, check the file path")

        shape = image.shape[:2]  # H, W
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, r, (dw, dh)

    def __preprocess(self, image: np.ndarray):
        """Resize + convert HWC BGR -> NCHW float32"""
        img, r, dwdh = self.__letterbox(image, (self.INPUT_H, self.INPUT_W))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # add batch
        return img, r, dwdh

    def __regional_proposal(self, image_arr: np.ndarray):
        blob, ratio, dwdh = self.__preprocess(image_arr)
        input_name = self.session.get_inputs()[0].name
        output_names = [o.name for o in self.session.get_outputs()]

        # Jalankan model
        boxes, scores, class_ids = self.session.run(output_names, {input_name: blob})

        return boxes, scores, class_ids, ratio, dwdh

    def __post_rpn(self, image_arr: np.ndarray, boxes, scores, class_ids, ratio, dwdh):
        mask = np.array(scores) >= self.conf_thresh
        boxes = np.array(boxes)[mask]
        scores = np.array(scores)[mask]
        class_ids = np.array(class_ids)[mask]

        # Kembalikan ke ukuran asli
        boxes[:, [0, 2]] -= dwdh[0]
        boxes[:, [1, 3]] -= dwdh[1]
        boxes /= ratio

        h, w = image_arr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h - 1)
        boxes = boxes.astype(int).tolist()

        return boxes, scores.tolist(), class_ids.tolist()

    def detect(self, file_path: str):
        if os.path.exists(file_path):
            if file_path.endswith((".jpg", ".jpeg", ".png")):
                image = cv2.imread(file_path)
                boxes, scores, class_ids, ratio, dwdh = self.__regional_proposal(image)
                boxes, scores, class_ids = self.__post_rpn(
                    image, boxes, scores, class_ids, ratio, dwdh
                )
                return boxes, scores, class_ids

            if file_path.endswith((".mp4", ".avi", ".mov", ".mkv")):
                cap = cv2.VideoCapture(file_path)
                results = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    boxes, scores, class_ids, ratio, dwdh = self.__regional_proposal(
                        frame
                    )
                    boxes, scores, class_ids = self.__post_rpn(
                        frame, boxes, scores, class_ids, ratio, dwdh
                    )
                    results.append((boxes, scores, class_ids))
                cap.release()
                return results
        else:
            raise FileNotFoundError(
                "Format file tidak dikenali! Gunakan .jpg, .png, atau .mp4."
            )
