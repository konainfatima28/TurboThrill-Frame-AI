import cv2
import numpy as np
from typing import Dict, Any, Tuple

class SparkDetector:
    """
    Advanced Computer Vision detector optimized with hard structural geometry constraints
    to extract authentic kinetic road sparks while completely dropping title cards, 
    promotional text glyphs, and high-exposure studio intros.
    """
    def __init__(self, config: Dict[str, Any]):
        cfg = config["spark_detection"]
        self.hsv_lower_1 = np.array(cfg["hsv_lower_1"], dtype=np.uint8)
        self.hsv_upper_1 = np.array(cfg["hsv_upper_1"], dtype=np.uint8)
        self.hsv_lower_2 = np.array(cfg["hsv_lower_2"], dtype=np.uint8)
        self.hsv_upper_2 = np.array(cfg["hsv_upper_2"], dtype=np.uint8)
        self.min_spark_area = cfg["min_spark_area"]
        self.max_spark_area = cfg["max_spark_area"]
        self.min_solidity = cfg["min_solidity"]
        self.brightness_threshold = cfg["brightness_threshold"]

        # Morphological kernel for linking micro-sparks within a single stream
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyzes a single frame to detect road sparks.
        Returns a tuple of (is_spark_detected, metadata_dict).
        """
        if frame is None:
            return False, {}

        # 1. Global Intensity Check: Instantly drop highly blown-out promotional splash intros
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_gray = np.mean(gray)
        if mean_gray > 135.0:
            return False, {"spark_confidence": 0.0, "spark_pixel_count": 0, "spark_area": 0.0, "spark_clusters": 0}

        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 3. Dual-band thresholding for orange/yellow/white transitions
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)
        combined_mask = cv2.bitwise_or(mask1, mask2)
        
        # 4. Brightness mask extraction (Isolate the high-intensity core)
        v_channel = hsv[:, :, 2]
        _, bright_mask = cv2.threshold(v_channel, max(self.brightness_threshold, 230), 255, cv2.THRESH_BINARY)
        spark_mask = cv2.bitwise_and(combined_mask, bright_mask)

        # Clean up noise
        spark_mask = cv2.morphologyEx(spark_mask, cv2.MORPH_CLOSE, self.kernel)

        # 5. Connected components analysis
        contours, _ = cv2.findContours(spark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_spark_area = 0.0
        total_spark_pixels = 0
        valid_spark_count = 0
        max_confidence = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            
            # HARD CEILING: Discard elements larger than 250px regardless of UI slider value.
            # This completely blocks large font strings and massive background graphics layers.
            if area < self.min_spark_area or area > min(self.max_spark_area, 250):
                continue

            # Structural shape validation
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            
            # TEXT DROP RULE: Digital font characters are highly solid objects (>0.80).
            # Real friction sparks are jagged, chaotic, and fragmented (lower solidity profiles).
            if solidity < self.min_solidity or solidity > 0.76:
                continue

            # BOUNDING BOX DIMENSION FILTER
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text sentences spread wide horizontally or climb vertically uniform.
            # Real spark particles are tiny scattered specs. Discard massive text-width bounds.
            if w > 45 or h > 45:
                continue

            # Local Matrix Variance Filter (Ensures internal pixel variance instead of flat font colors)
            roi_v = v_channel[y:y+h, x:x+w]
            if roi_v.size < 4:
                continue
            _, std_dev = cv2.meanStdDev(roi_v)
            if std_dev[0][0] < 20.0:
                continue

            # Valid spark particle confirmed
            mean_brightness = np.mean(roi_v) if roi_v.size > 0 else 0
            confidence = min((mean_brightness / 255.0) * (solidity * 1.2), 1.0)
            if confidence > max_confidence:
                max_confidence = confidence

            total_spark_area += area
            total_spark_pixels += cv2.countNonZero(spark_mask[y:y+h, x:x+w])
            valid_spark_count += 1

        is_detected = valid_spark_count > 0

        metadata = {
            "spark_confidence": round(max_confidence, 4) if is_detected else 0.0,
            "spark_pixel_count": int(total_spark_pixels),
            "spark_area": float(total_spark_area),
            "spark_clusters": valid_spark_count
        }

        return is_detected, metadata