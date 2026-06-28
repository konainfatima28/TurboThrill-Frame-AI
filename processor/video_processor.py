import os
import time
import cv2
from typing import Dict, Any, List, Callable
from detectors.spark_detector import SparkDetector
from utils.duplicate_remover import DuplicateRemover
from exporters.data_exporter import DataExporter

class VideoProcessor:
    """
    Coordinates frame parsing operations across individual video files, 
    orchestrating the computer vision detector, deduplicator, and export layers.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = SparkDetector(config)
        self.deduplicator = DuplicateRemover(config["deduplication"]["similarity_threshold"])
        self.sampling_rate = config["processing"]["frame_sampling_rate"]
        self.output_format = config["processing"]["output_format"]

    def process_video(self, video_path: str, output_base_dir: str, log_callback: Callable[[str], None]) -> int:
        if not os.path.exists(video_path):
            log_callback(f"[ERROR] Video file not found: {video_path}")
            return 0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log_callback(f"[ERROR] Failed to open video file: {video_path}")
            return 0

        video_name = os.path.basename(video_path)
        log_callback(f"[INFO] Initializing processing for: {video_name}")
        
        output_dir = DataExporter.initialize_output_directory(output_base_dir, video_name)
        self.deduplicator.reset()

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else 30.0
        
        frame_idx = 0
        saved_count = 0
        records: List[Dict[str, Any]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % self.sampling_rate != 0:
                continue

            start_time = time.time()
            is_spark, meta = self.detector.process_frame(frame)
            processing_time = time.time() - start_time

            if is_spark:
                # Deduplication check to prevent duplicate frames of the same spark
                if not self.deduplicator.is_duplicate(frame):
                    saved_path = DataExporter.save_frame(frame, output_dir, frame_idx, self.output_format)
                    h, w, _ = frame.shape
                    
                    record = {
                        "video_name": video_name,
                        "frame_number": frame_idx,
                        "timestamp_seconds": round(frame_idx / fps, 3),
                        "spark_confidence": meta["spark_confidence"],
                        "spark_pixel_count": meta["spark_pixel_count"],
                        "spark_area": meta["spark_area"],
                        "resolution": f"{w}x{h}",
                        "processing_time_seconds": round(processing_time, 5),
                        "saved_path": os.path.basename(saved_path)
                    }
                    records.append(record)
                    saved_count += 1
                    log_callback(f"[DETECTED] Saved spark frame {frame_idx} (Confidence: {meta['spark_confidence']})")

        cap.release()
        DataExporter.write_metadata_summary(output_dir, records)
        log_callback(f"[COMPLETED] Processed {video_name}. Extracted {saved_count} unique spark frames.")
        return saved_count