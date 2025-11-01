import cv2
import numpy as np
import time
import queue
import threading
from typing import Callable, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..Yolo.V11 import YOLOv11Detector
from ..FasterRCNN.detectron2 import Detectron2Detector

class VideoProcess(ABC):
    """Abstract base class for video processing with threading support"""
    
    def __init__(self, session, max_queue_size: int = 16):
        self.session = session
        self.max_queue_size = max_queue_size
        
        # Threading components
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        
        # Processing stats
        self.frame_count = 0
        self.start_time = None
        
        # Must be set by subclass
        self.input_size = None

    @abstractmethod
    def _preprocess(self, frame: np.ndarray):
        """Preprocess frame - must be implemented by subclass"""
        pass

    @abstractmethod
    def _postprocess(self, frame: np.ndarray, outputs, *args):
        """Postprocess frame with model outputs - must be implemented by subclass"""
        pass

    def process_video(self, input_path: str, output_path: str, batch_size: int = 8, 
                     show_preview: bool = False, use_threading: bool = True):
        """Main video processing method"""
        
        if use_threading:
            return self._process_video_threaded(input_path, output_path, show_preview)
        else:
            return self._process_video_single(input_path, output_path, batch_size, show_preview)

    def _process_video_single(self, input_path: str, output_path: str, batch_size: int = 8, show_preview: bool = False):
        """Single-threaded video processing"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
        
        print(f"Processing video (Single-threaded): {input_path}")
        print(f"Resolution: {orig_width}x{orig_height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frame_count = 0
        start_time = time.time()
        batch_count = 0
        
        try:
            while True:
                batch_frames = []
                batch_data = []
                
                # Collect batch
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    batch_frames.append(frame)
                    
                    # Preprocess
                    preprocess_result = self._preprocess(frame)
                    if isinstance(preprocess_result, tuple):
                        blob, *preprocess_args = preprocess_result
                    else:
                        blob = preprocess_result
                        preprocess_args = []
                    
                    if blob.ndim == 3:
                        blob = blob[None, :, :, :]
                    
                    batch_data.append((blob, preprocess_args))
                
                if not batch_frames:
                    break
                
                # Process batch
                for i, frame in enumerate(batch_frames):
                    blob, preprocess_args = batch_data[i]
                    outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
                    result_frame = self._postprocess(frame, outputs, *preprocess_args)
                    
                    out.write(result_frame)
                    frame_count += 1
                    
                    if show_preview:
                        display_frame = self._create_display_frame(
                            result_frame, frame_count, start_time, orig_width, orig_height
                        )
                        cv2.imshow(f"{self.__class__.__name__} - Detection", display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in [ord('q'), 27]:
                            print("Processing interrupted by user")
                            raise KeyboardInterrupt
                
                batch_count += 1
                if batch_count % 10 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {current_fps:.1f}")
                        
        finally:
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"Processing completed in {total_time:.2f} seconds")
            if total_time > 0:
                print(f"Average FPS: {frame_count/total_time:.2f}")
            print(f"Output saved to: {output_path}")

    def _process_video_threaded(self, input_path: str, output_path: str, show_preview: bool = False):
        """Dual-threaded video processing"""
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))
        
        print(f"Processing video (Dual-threaded): {input_path}")
        print(f"Resolution: {orig_width}x{orig_height}, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # Start processing thread
        processor_thread = threading.Thread(target=self._processing_worker)
        processor_thread.daemon = True
        processor_thread.start()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Wait if queue is full
                if self.input_queue.qsize() >= self.max_queue_size:
                    while self.input_queue.qsize() > self.max_queue_size // 2:
                        time.sleep(0.001)
                
                self.input_queue.put((frame, orig_width, orig_height))
                
                # Process output frames
                while not self.output_queue.empty():
                    result_frame = self.output_queue.get()
                    out.write(result_frame)
                    self.frame_count += 1
                    
                    if show_preview:
                        display_frame = self._create_display_frame(
                            result_frame, self.frame_count, self.start_time, orig_width, orig_height
                        )
                        cv2.imshow(f"{self.__class__.__name__} - Detection", display_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in [ord('q'), 27]:
                            print("Processing interrupted by user")
                            raise KeyboardInterrupt
                
                # Progress reporting
                if self.frame_count % 100 == 0:
                    elapsed = time.time() - self.start_time
                    current_fps = self.frame_count / elapsed if elapsed > 0 else 0
                    progress = (self.frame_count / total_frames * 100) if total_frames > 0 else 0
                    print(f"Processed {self.frame_count}/{total_frames} frames ({progress:.1f}%) - FPS: {current_fps:.1f}")
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        finally:
            self.stop_event.set()
            
            # Process remaining frames
            while not self.output_queue.empty():
                result_frame = self.output_queue.get()
                out.write(result_frame)
                self.frame_count += 1
            
            processor_thread.join(timeout=5.0)
            
            cap.release()
            out.release()
            if show_preview:
                cv2.destroyAllWindows()
            
            total_time = time.time() - self.start_time
            print(f"Processing completed in {total_time:.2f} seconds")
            if total_time > 0:
                print(f"Average FPS: {self.frame_count/total_time:.2f}")
            print(f"Output saved to: {output_path}")

    def _processing_worker(self):
        """Worker thread for processing frames"""
        while not self.stop_event.is_set():
            try:
                frame, orig_width, orig_height = self.input_queue.get(timeout=0.1)
                
                # Preprocess
                preprocess_result = self._preprocess(frame)
                if isinstance(preprocess_result, tuple):
                    blob, *preprocess_args = preprocess_result
                else:
                    blob = preprocess_result
                    preprocess_args = []
                
                if blob.ndim == 3:
                    blob = blob[None, :, :, :]
                
                # Run inference
                outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
                
                # Postprocess
                result_frame = self._postprocess(frame, outputs, *preprocess_args)
                
                # Put result in output queue
                while self.output_queue.qsize() >= self.max_queue_size and not self.stop_event.is_set():
                    time.sleep(0.001)
                
                if not self.stop_event.is_set():
                    self.output_queue.put(result_frame)
                    
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing worker: {e}")
                break

    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame"""
        preprocess_result = self._preprocess(frame)
        if isinstance(preprocess_result, tuple):
            blob, *preprocess_args = preprocess_result
        else:
            blob = preprocess_result
            preprocess_args = []
        
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        result_frame = self._postprocess(frame, outputs, *preprocess_args)
        return result_frame

    def _create_display_frame(self, frame: np.ndarray, frame_count: int, start_time: float, 
                            orig_width: int, orig_height: int) -> np.ndarray:
        """Create a display frame with info overlay"""
        max_display_size = 800
        h, w = frame.shape[:2]
        
        if max(h, w) > max_display_size:
            scale = max_display_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame.copy()
        
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        input_size_str = f"{self.input_size[0]}x{self.input_size[1]}" if self.input_size else "Not set"
        
        info_lines = [
            f"Frame: {frame_count}",
            f"FPS: {fps:.1f}",
            f"Original: {orig_width}x{orig_height}",
            f"Model Input: {input_size_str}",
            "Press 'q' to quit"
        ]
        
        for i, line in enumerate(info_lines):
            y_position = 30 + i * 25
            cv2.putText(display_frame, line, (10, y_position), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return display_frame

    def process_realtime(self, input_path: str, show_preview: bool = True):
        """Real-time video processing"""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {input_path}")
        
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"{self.__class__.__name__} Real-time Processing")
        print(f"Original: {orig_width}x{orig_height}")
        if self.input_size:
            print(f"Model Input: {self.input_size[0]}x{self.input_size[1]}")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frame = self._process_single_frame(frame)
                frame_count += 1
                
                display_frame = self._create_display_frame(
                    processed_frame, frame_count, start_time, orig_width, orig_height
                )
                cv2.imshow(f'{self.__class__.__name__} - Detection', display_frame)
                
                # Calculate dynamic delay to maintain FPS
                elapsed = time.time() - start_time
                expected_time = frame_count / fps if fps > 0 else 0
                actual_delay = max(1, int((expected_time - elapsed) * 1000)) if elapsed < expected_time else 1
                
                key = cv2.waitKey(actual_delay) & 0xFF
                if key == ord('q'):
                    break
                    
                if frame_count % 100 == 0:
                    current_fps = frame_count / elapsed
                    print(f"Processed {frame_count} frames - FPS: {current_fps:.1f}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"Processed {frame_count} frames in {total_time:.2f}s")
            if total_time > 0:
                print(f"Average FPS: {frame_count/total_time:.2f}")

    def process_webcam(self, camera_id: int = 0, show_preview: bool = True):
        """Webcam processing"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam: {camera_id}")
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS,15)
        
        frame_skip = 2
        frame_counter = 0
        
        print(f"{self.__class__.__name__} Webcam Processing")
        frame_count = 0
        start_time = time.time()
        last_frame = None
        
        try:
            while True:
                ret,frame = cap.read()
                if not ret:
                    break
                frame_counter += 1
                if frame_counter % frame_skip != 0:
                    # Use last processed frame for display
                    if last_frame is not None and show_preview:
                        cv2.imshow(f'{self.__class__.__name__} Webcam', last_frame)
                    continue
                try:
                    processed_frame = self._process_single_frame(frame)
                    last_frame = processed_frame
                    frame_count += 1
                    
                    if show_preview:
                        cv2.imshow(f'{self.__class__.__name__} Webcam', processed_frame)
                except Exception as e:
                    print(f"Processing error: {e}")
                    if show_preview:
                        cv2.imshow(f'{self.__class__.__name__} Webcam', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
                if frame_count % 30 == 0:  # Report less frequently
                    elapsed = time.time() - start_time
                    current_fps = frame_count / elapsed
                    print(f"Frames: {frame_count}, FPS: {current_fps:.1f}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            print(f"Processed {frame_count} frames in {total_time:.2f}s")
            if total_time > 0:
                print(f"Average FPS: {frame_count/total_time:.2f}")

class YOLOVideoProcess(VideoProcess):
    """YOLO-specific video processor"""
    
    def __init__(self, detector: YOLOv11Detector, max_queue_size: int = 16):
        self.detector = detector
        super().__init__(session=detector.session, max_queue_size=max_queue_size)
        
        # Set input size for base class
        self.input_size = (640, 640)
        self._override_detector_input_size()

    def _override_detector_input_size(self):
        """Override detector's input size to ensure 640x640 processing"""
        self.detector.INPUT_W = 640
        self.detector.INPUT_H = 640

    def _preprocess(self, frame: np.ndarray):
        """YOLO preprocessing"""
        self._override_detector_input_size()
        blob, ratio, dwdh = self.detector._YOLOv11Detector__preprocess(frame)
        orig_h, orig_w = frame.shape[:2]
        return blob, ratio, dwdh, orig_h, orig_w

    def _postprocess(self, frame: np.ndarray, outputs, ratio, dwdh, orig_h, orig_w):
        """YOLO postprocessing and drawing"""
        boxes, scores, class_ids = self.detector._YOLOv11Detector__postprocess(
            outputs, orig_h, orig_w, ratio, dwdh
        )

        for box, score, cls_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.detector.classes[cls_id]}: {score:.2f}"
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                         (x1 + label_width, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame


class Detectron2VideoProcess(VideoProcess):
    def __init__(self,detector:Detectron2Detector,max_queue_size:int=8):
        self.detector = detector
        super().__init__(session=detector.session, max_queue_size=max_queue_size)

        # set input size for base class
        self.input_size = (detector.input_height,detector.input_weight)
    def _preprocess(self, frame):
        blob, ratio, dwdh = self.detector._Detectron2Detector__preprocess(frame)
        return blob, ratio, dwdh
    def _postprocess(self, frame: np.ndarray, outputs: List[np.ndarray], 
                    ratio: float, dwdh: Tuple[float, float]) -> np.ndarray:
        try:
            if len(outputs) == 3:
                boxes, scores, class_ids = outputs
            else:
                boxes, scores, class_ids = outputs[0], outputs[1], outputs[2] if len(outputs) > 2 else None
            
            # Filter dengan confidence threshold
            if scores.size > 0:
                mask = scores >= self.detector.conf_thresh
                boxes = boxes[mask]
                scores = scores[mask]
                class_ids = class_ids[mask] if class_ids is not None else np.zeros_like(scores)
                
                # Adjust boxes to original coordinates
                if len(boxes) > 0:
                    boxes[:, [0, 2]] -= dwdh[0]
                    boxes[:, [1, 3]] -= dwdh[1]
                    boxes /= ratio
                    
                    # Clip boxes
                    h, w = frame.shape[:2]
                    np.clip(boxes[:, [0, 2]], 0, w, out=boxes[:, [0, 2]])
                    np.clip(boxes[:, [1, 3]], 0, h, out=boxes[:, [1, 3]])
                    
                    # Draw detections (optimized)
                    frame = self._draw_detections(frame, boxes, scores, class_ids)
            
            return frame
            
        except Exception as e:
            print(f"Postprocessing error: {e}")
            return frame
 
    
    def _draw_detections(self, frame: np.ndarray, boxes: np.ndarray, 
                        scores: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            boxes: Bounding boxes in [x1, y1, x2, y2] format
            scores: Confidence scores
            class_ids: Class IDs
            
        Returns:
            Frame with drawn detections
        """
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].astype(int)
            score = scores[i]
            cls_id = int(class_ids[i])
            
            # Get class name
            class_name = self.detector.classes[cls_id] if (self.detector.classes and 
                                                          cls_id < len(self.detector.classes)) else f"Class_{cls_id}"
            
            # Simple drawing without background for speed
            color = (0, 255, 0)  # Fixed color for speed
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)  # Thinner lines
            
            # Simple label without background
            label = f"{class_name}:{score:.1f}"
            cv2.putText(frame, label, (x1, max(y1-5, 15)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        if self.detector.classes and class_id < len(self.detector.classes):
            return self.detector.classes[class_id]
        return f"Class_{class_id}"

    def _get_color(self,class_id:int)->Tuple[int, int, int]:
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]

    def process_single_image(self, image_path: str, output_path: Optional[str] = None, 
                           show_result: bool = True) -> np.ndarray:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Process image
        processed_image = self._process_single_frame(image)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, processed_image)
            print(f"Result saved to: {output_path}")
        
        # Display if requested
        if show_result:
            # Resize for display if too large
            h, w = processed_image.shape[:2]
            max_display_size = 1200
            if max(h, w) > max_display_size:
                scale = max_display_size / max(h, w)
                display_image = cv2.resize(processed_image, 
                                         (int(w * scale), int(h * scale)))
            else:
                display_image = processed_image
            
            cv2.imshow('Detectron2 Detection', display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return processed_image
    def get_detection_info(self, image_path: str) -> Tuple[List, List, List]:
        """
        Get detection information without drawing
        
        Args:
            image_path: Path to input image
            
        Returns:
            Tuple of (boxes, scores, class_ids)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Use detector's detect method directly
        return self.detector.detect(image_path)

    def benchmark_performance(self, input_path: str, num_iterations: int = 100) -> dict:
        """
        Benchmark detection performance
        
        Args:
            input_path: Path to video or image for benchmarking
            num_iterations: Number of iterations for benchmark
            
        Returns:
            Dictionary with performance metrics
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        print(f"Running benchmark with {num_iterations} iterations...")
        
        times = []
        frame_count = 0
        
        try:
            while frame_count < num_iterations:
                ret, frame = cap.read()
                if not ret:
                    # Loop video for benchmarking
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                start_time = time.time()
                self._process_single_frame(frame)
                end_time = time.time()
                
                times.append(end_time - start_time)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count}/{num_iterations} frames")
        
        finally:
            cap.release()
        
        # Calculate statistics
        times = np.array(times)
        stats = {
            'total_frames': len(times),
            'total_time': np.sum(times),
            'average_time': np.mean(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'fps': 1.0 / np.mean(times),
            'std_time': np.std(times)
        }
        
        print("\n=== Benchmark Results ===")
        print(f"Total frames processed: {stats['total_frames']}")
        print(f"Total time: {stats['total_time']:.2f}s")
        print(f"Average time per frame: {stats['average_time']*1000:.2f}ms")
        print(f"FPS: {stats['fps']:.2f}")
        print(f"Min/Max time: {stats['min_time']*1000:.2f}ms / {stats['max_time']*1000:.2f}ms")
        print(f"Standard deviation: {stats['std_time']*1000:.2f}ms")
        
        return stats