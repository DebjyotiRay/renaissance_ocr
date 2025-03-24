"""
Layout Validator Agent for validating and refining layout detection in Renaissance documents.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from agents.base_agent import BaseAgent
from configs.config import LayoutValidatorConfig

logger = logging.getLogger(__name__)

class LayoutValidatorAgent(BaseAgent):
    """
    Agent responsible for validating and refining the detected layout of document regions.
    """
    
    def __init__(self, config: LayoutValidatorConfig):
        """
        Initialize the Layout Validator agent.
        
        Args:
            config: Layout validator configuration
        """
        super().__init__("LayoutValidator", config.__dict__)
        self.log_info("Layout Validator agent initialized")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and refine detected document regions.
        
        Args:
            inputs: Dictionary containing:
                - 'image': Full document image
                - 'regions': List of region dictionaries with 'bbox' and 'region' keys
                
        Returns:
            Dictionary with validated and refined regions
        """
        # Validate inputs
        required_keys = ['image', 'regions']
        if not self.validate_inputs(inputs, required_keys):
            return {'error': 'Invalid inputs', 'validated_regions': []}
        
        regions = inputs['regions']
        image = inputs['image']
        
        self.log_info(f"Validating {len(regions)} regions")
        
        # Apply validation methods based on configuration
        validated_regions = regions.copy()
        
        for method in self.config["validation_methods"]:
            if method == "overlap_removal":
                validated_regions = self._remove_overlapping_regions(validated_regions)
            elif method == "alignment_correction":
                validated_regions = self._correct_region_alignment(validated_regions, image)
            elif method == "size_filtering":
                validated_regions = self._filter_by_size(validated_regions)
        
        # Create a visualization
        visualization = self._create_visualization(inputs['image'], validated_regions)
        
        self.log_info(f"Validation completed. {len(validated_regions)} regions after validation.")
        
        result = {
            'validated_regions': validated_regions,
            'visualization': visualization
        }
        
        return result
    
    def _remove_overlapping_regions(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove or merge overlapping regions based on IoU threshold.
        
        Args:
            regions: List of region dictionaries
            
        Returns:
            List of regions with overlaps resolved
        """
        if not regions:
            return []
        
        # Sort regions by area (largest first)
        sorted_regions = sorted(regions, key=lambda r: r['bbox'][2] * r['bbox'][3], reverse=True)
        
        validated_regions = [sorted_regions[0]]  # Start with the largest region
        
        for region in sorted_regions[1:]:
            # Check overlap with all validated regions
            overlaps = [self._calculate_iou(region['bbox'], vr['bbox']) 
                       for vr in validated_regions]
            
            max_overlap = max(overlaps) if overlaps else 0
            
            # If no significant overlap, add the region
            if max_overlap < self.config["overlap_threshold"]:
                validated_regions.append(region)
        
        self.log_debug(f"Overlap removal: {len(regions)} → {len(validated_regions)} regions")
        return validated_regions
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IoU value (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate coordinates of the intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _correct_region_alignment(self, regions: List[Dict[str, Any]], 
                                 image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Correct region alignment based on text lines and page structure.
        
        Args:
            regions: List of region dictionaries
            image: Full document image
            
        Returns:
            List of regions with corrected alignment
        """
        corrected_regions = []
        
        for region in regions:
            x, y, w, h = region['bbox']
            
            # Adjust region to better align with text lines
            # This is a simplified version - real implementation would be more sophisticated
            
            # For demonstration, just add a small padding
            padding = 5
            new_x = max(0, x - padding)
            new_y = max(0, y - padding)
            new_w = min(image.shape[1] - new_x, w + 2 * padding)
            new_h = min(image.shape[0] - new_y, h + 2 * padding)
            
            # Create new region with corrected bbox
            corrected_region = region.copy()
            corrected_region['bbox'] = (new_x, new_y, new_w, new_h)
            
            # Update the region image
            if len(image.shape) == 3:
                corrected_region['region'] = image[new_y:new_y+new_h, new_x:new_x+new_w, :]
            else:
                corrected_region['region'] = image[new_y:new_y+new_h, new_x:new_x+new_w]
            
            corrected_regions.append(corrected_region)
        
        self.log_debug("Region alignment correction applied")
        return corrected_regions
    
    def _filter_by_size(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter regions by size to remove noise and artifacts.
        
        Args:
            regions: List of region dictionaries
            
        Returns:
            Filtered list of regions
        """
        filtered_regions = []
        
        for region in regions:
            w, h = region['bbox'][2], region['bbox'][3]
            
            # Filter out very small regions that are likely noise
            min_width, min_height = 10, 10
            if w > min_width and h > min_height:
                filtered_regions.append(region)
        
        self.log_debug(f"Size filtering: {len(regions)} → {len(filtered_regions)} regions")
        return filtered_regions
    
    def _create_visualization(self, image: np.ndarray, 
                             regions: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create a visualization of the validated regions.
        
        Args:
            image: Full document image
            regions: List of validated region dictionaries
            
        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        if len(image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw each region
        for i, region in enumerate(regions):
            x, y, w, h = region['bbox']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for valid regions
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Add region ID
            cv2.putText(vis_image, f"R{i}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image