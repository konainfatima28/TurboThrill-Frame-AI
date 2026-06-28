import cv2
from PIL import Image
import imagehash
import numpy as np
from typing import Optional

class DuplicateRemover:
    """
    Uses perceptual hashing to compute image similarity thresholds, 
    preventing sequential near-duplicate frames from cluttering output data.
    """
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.last_hash: Optional[imagehash.ImageHash] = None

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """
        Determines whether the incoming frame is perceptually identical to the last matched frame.
        """
        if frame is None:
            return True
            
        # Convert BGR CV2 frame to PIL Image for ImageHash compatibility
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        current_hash = imagehash.average_hash(pil_img)

        if self.last_hash is None:
            self.last_hash = current_hash
            return False

        # Calculate normalized Hamming distance similarity
        hamming_distance = current_hash - self.last_hash
        hash_size = current_hash.hash.size
        similarity = 1.0 - (hamming_distance / hash_size)

        if similarity >= self.similarity_threshold:
            return True
        else:
            self.last_hash = current_hash
            return False

    def reset(self) -> None:
        """Resets the internal tracking hash cache between different video files."""
        self.last_hash = None