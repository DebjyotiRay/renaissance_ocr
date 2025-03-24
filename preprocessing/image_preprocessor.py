"""
Image preprocessing module for Renaissance documents.
"""

import cv2
import numpy as np
from skimage import filters, morphology, restoration
from PIL import Image
import logging
from typing import Tuple, Optional, List, Dict, Any, Union

from configs.config import PreprocessingConfig

logger = logging.getLogger(__name__)

class RenaissanceImageProcessor:
    """
    Specialized image preprocessor for Renaissance-era documents with
    techniques optimized for historical document enhancement.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize the image processor with configuration.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        logger.info("Initialized Renaissance Image Processor")
    
    def preprocess(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply a full preprocessing pipeline to the input image.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image if it's a path
        if isinstance(image, str):
            logger.info(f"Loading image from {image}")
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image)
        else:
            img = image.copy()
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Apply noise reduction if configured
        if self.config.apply_noise_reduction:
            gray = self._reduce_noise(gray)
        
        # Apply CLAHE enhancement if configured
        if self.config.apply_clahe:
            gray = self._apply_clahe(gray)
        
        # Apply binarization if configured
        if self.config.apply_binarization:
            processed = self._binarize(gray)
        else:
            processed = gray
            
        logger.info("Image preprocessing completed")
        return processed
    
    def _reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction techniques specific to historical documents.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Noise-reduced image
        """
        # Apply non-local means denoising - effective but computationally intensive
        denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
        
        # For severely degraded documents, try bilateral filter for edge preservation
        # denoised = cv2.bilateralFilter(denoised, 9, 75, 75)
        
        logger.debug("Applied noise reduction")
        return denoised
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization.
        Particularly effective for historical documents with uneven lighting.
        
        Args:
            image: Input grayscale image
            
        Returns:
            CLAHE-enhanced image
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size
        )
        enhanced = clahe.apply(image)
        logger.debug("Applied CLAHE enhancement")
        return enhanced
    
    def _binarize(self, image: np.ndarray) -> np.ndarray:
        """
        Binarize the image using techniques suitable for historical documents.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Binarized image
        """
        # Sauvola thresholding is particularly effective for historical documents
        # with variable backgrounds and degradation
        window_size = 25
        thresh_sauvola = filters.threshold_sauvola(image, window_size=window_size)
        binary = image > thresh_sauvola
        
        # Convert to uint8 (0 and 255)
        binary = binary.astype(np.uint8) * 255
        
        # Optional: Apply morphological operations to clean up the result
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        logger.debug("Applied binarization")
        return binary
    
    def detect_skew(self, image: np.ndarray) -> float:
        """
        Detect the skew angle of the document.
        
        Args:
            image: Input image (grayscale or binarized)
            
        Returns:
            Estimated skew angle in degrees
        """
        # Ensure image is binarized
        if self.config.apply_binarization and np.max(image) > 1:
            binary = image
        else:
            thresh_sauvola = filters.threshold_sauvola(image, window_size=25)
            binary = (image > thresh_sauvola).astype(np.uint8) * 255
        
        # Apply edge detection
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        
        if lines is not None:
            angles = []
            for line in lines:
                rho, theta = line[0]
                # Filter horizontal and vertical lines
                if 0.1 < theta < 1.5 or 1.7 < theta < 3.1:
                    angle = np.degrees(theta - np.pi/2)
                    angles.append(angle)
            
            if angles:
                # Return the median angle
                return np.median(angles)
        
        logger.debug("No skew detected")
        return 0.0
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct the skew of the document.
        
        Args:
            image: Input image
            
        Returns:
            Deskewed image
        """
        angle = self.detect_skew(image)
        
        if abs(angle) > 0.5:  # Only correct if skew is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            
            # Calculate rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=(255, 255, 255))
            
            logger.debug(f"Corrected skew by {angle} degrees")
            return rotated
        
        return image
    
    def enhance_text_regions(self, image: np.ndarray) -> np.ndarray:
        """
        Apply specialized enhancements for text regions in historical documents.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Enhanced image
        """
        # Apply unsharp masking to enhance text edges
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        enhanced = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        
        # Adjust the contrast to make text more prominent
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        logger.debug("Applied text region enhancement")
        return enhanced

    def segment_page(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Segment the page into different regions (text, margins, decorations).
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary containing different regions and metadata
        """
        # Create a copy of the image for visualization
        visualization = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()
        
        # Ensure we're working with a binary image
        if self.config.apply_binarization and np.max(image) > 1:
            binary = image
        else:
            # Apply binarization
            binary = self._binarize(image)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to eliminate noise
        min_width, min_height = self.config.min_region_size
        valid_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > min_width and h > min_height:
                valid_contours.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': cv2.contourArea(contour)
                })
        
        # Sort by area (descending)
        valid_contours.sort(key=lambda c: c['area'], reverse=True)
        
        # Take the top N contours (or fewer if there aren't enough)
        top_contours = valid_contours[:min(self.config.max_regions, len(valid_contours))]
        
        # Extract regions and create visualization
        regions = []
        for i, contour_data in enumerate(top_contours):
            contour = contour_data['contour']
            x, y, w, h = contour_data['bbox']
            
            # Extract the region from the original image
            region = image[y:y+h, x:x+w]
            
            regions.append({
                'id': i,
                'bbox': (x, y, w, h),
                'region': region,
                'area': contour_data['area']
            })
            
            # Draw the contour on the visualization
            color = (0, 255, 0)  # Green for text regions
            cv2.rectangle(visualization, (x, y), (x+w, y+h), color, 2)
            cv2.putText(visualization, f"R{i}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        result = {
            'regions': regions,
            'visualization': visualization,
            'binary': binary
        }
        
        logger.info(f"Segmented page into {len(regions)} regions")
        return result