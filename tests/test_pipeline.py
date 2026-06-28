import os
import numpy as np
import pytest
from detectors.spark_detector import SparkDetector
from utils.duplicate_remover import DuplicateRemover

@pytest.fixture
def mock_config():
    return {
        "spark_detection": {
            "hsv_lower_1": [0, 80, 200],
            "hsv_upper_1": [30, 255, 255],
            "hsv_lower_2": [150, 80, 200],
            "hsv_upper_2": [180, 255, 255],
            "min_spark_area": 2,
            "max_spark_area": 1000,
            "min_solidity": 0.2,
            "brightness_threshold": 200
        },
        "deduplication": {
            "similarity_threshold": 0.90
        },
        "processing": {
            "frame_sampling_rate": 1,
            "output_format": "jpg"
        }
    }

def test_spark_detector_blank_frame(mock_config):
    detector = SparkDetector(mock_config)
    # Generate pure black frame matrix
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    is_detected, meta = detector.process_frame(blank_frame)
    assert is_detected is False
    assert meta["spark_pixel_count"] == 0

def test_spark_detector_positive_spark(mock_config):
    detector = SparkDetector(mock_config)
    # Generate blank black template matrix frame
    spark_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a simulated intense orange-yellow spark spot core
    spark_frame[200:205, 300:305] = [20, 200, 255] # BGR structure matching HSV targets
    is_detected, meta = detector.process_frame(spark_frame)
    assert is_detected is True
    assert meta["spark_pixel_count"] > 0

def test_duplicate_remover(mock_config):
    remover = DuplicateRemover(similarity_threshold=0.95)
    frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
    frame_b = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # First frame evaluation configuration setup
    assert remover.is_duplicate(frame_a) is False
    # Verifying identical frames evaluate as duplicate matches
    assert remover.is_duplicate(frame_b) is True