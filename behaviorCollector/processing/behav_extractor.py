import cv2
import os
import warnings
from .behav_container import BehavCollector, EVENT, STATE
from tqdm import tqdm


FPS_WRITE = 10
PADDING_MS = 1000  # export window padding before/after behavior


class BehavExtractor:
    def __init__(self, bcollector: BehavCollector):
        self.bcollector = bcollector
        self.video_capture = [
            cv2.VideoCapture(path) for path in bcollector.video_path if path is not None
        ]
        
    def extract_epochs(self, path_dir: str, tqdm_fn=None, selections=None):
        if any(os.scandir(path_dir)):
            warnings.warn(f"Directory {path_dir} is not empty")
            
        if tqdm_fn is None:
            tqdm_fn = tqdm
        
        for behav_idx, b in enumerate(self.bcollector.behav_set):
            if not b.time_ms:
                continue

            selected_indices = list(range(len(b.time_ms)))
            if selections is not None:
                selected = selections.get(behav_idx)
                # None -> select all epochs for this behavior
                if selected is None:
                    selected_indices = list(range(len(b.time_ms)))
                else:
                    selected_indices = sorted(selected)
                if not selected_indices:
                    continue

            bar = tqdm_fn(total=len(selected_indices), desc=f"Extracting {b.name} epochs")
            for n in selected_indices:
                try:
                    if b.type == STATE:
                        start_ms = b.time_ms[n][0]
                        end_ms = b.time_ms[n][1]
                        
                        # name_start time_end time (video_id)
                        prefix = os.path.join(path_dir, f"{b.name}_{start_ms//1000}_{end_ms//1000}")
                        self.extract_single_epoch(prefix, start_ms, end_ms)
                    elif b.type == EVENT:
                        start_ms = b.time_ms[n]
                        prefix = os.path.join(path_dir, f"{b.name}_{start_ms//1000}")
                        self.extract_single_event(prefix, start_ms)
                except Exception as e:
                    warnings.warn(f"Failed to extract epoch {n} for behavior {b.name}: {e}")
                bar.update()
            bar.close()
        
        return True
    
    def _get_video_duration_ms(self, cap):
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0:
            return None
        return (total_frames / fps) * 1000

    def _draw_behavior_border(self, frame):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 2)
        return frame

    def extract_single_epoch(self, prefix_video, start_ms: int, end_ms: int):
        padded_start = max(0, start_ms - PADDING_MS)
        padded_end = end_ms + PADDING_MS

        for n, cap in enumerate(self.video_capture):
            if not cap.isOpened():
                raise ValueError("Video capture cannot be opened")

            duration_ms = self._get_video_duration_ms(cap)
            start_clip = padded_start
            end_clip = padded_end
            if duration_ms is not None:
                start_clip = max(0, min(start_clip, duration_ms))
                end_clip = max(start_clip, min(end_clip, duration_ms))
            
            writter = cv2.VideoWriter(
                f"{prefix_video}({n}).avi",
                cv2.VideoWriter_fourcc(*'XVID'),
                FPS_WRITE,
                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )
            
            cap.set(cv2.CAP_PROP_POS_MSEC, start_clip)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if current_ms > end_clip:
                    break
                if start_ms <= current_ms <= end_ms:
                    frame = self._draw_behavior_border(frame)
                writter.write(frame)
            writter.release()

    def extract_single_event(self, perfix_event, start_ms: int):
        for n, cap in enumerate(self.video_capture):
            if not cap.isOpened():
                raise ValueError("Video capture cannot be opened")
            
            cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at {perfix_event} ms")
            
            cv2.imwrite(f"{perfix_event}({n}).jpg", frame)
