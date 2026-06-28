import os
import json
import csv
import cv2
import numpy as np
from typing import Dict, Any, List

class DataExporter:
    """
    Handles file output operations, writing non-lossy source image frames,
    consolidated performance CSV files, and complete individual frame JSON runs.
    """
    @staticmethod
    def initialize_output_directory(base_dir: str, video_name: str) -> str:
        safe_name = os.path.splitext(os.path.basename(video_name))[0]
        output_path = os.path.join(base_dir, safe_name)
        os.makedirs(output_path, exist_ok=True)
        return output_path

    @staticmethod
    def save_frame(frame: np.ndarray, output_dir: str, frame_num: int, fmt: str = "jpg") -> str:
        filename = f"spark_{frame_num:04d}.{fmt.lower()}"
        full_path = os.path.join(output_dir, filename)
        
        if fmt.lower() == "png":
            # Maximum lossless PNG compression strategy
            cv2.imwrite(full_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # High-quality baseline JPEG export (100% Quality)
            cv2.imwrite(full_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return full_path

    @staticmethod
    def write_metadata_summary(output_dir: str, records: List[Dict[str, Any]]) -> None:
        # Write clean metadata.json file
        json_path = os.path.join(output_dir, "metadata.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        # Write flat report.csv layout
        csv_path = os.path.join(output_dir, "report.csv")
        if not records:
            return
            
        headers = records[0].keys()
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(records)