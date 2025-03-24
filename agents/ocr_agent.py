"""
OCR Agent that extracts text from image regions using the vision-language models.
"""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import cv2
from PIL import Image

from agents.base_agent import BaseAgent
from models.vision_encoder import QuantizedVisionEncoder
from models.language_model import QuantizedLanguageModel
from models.vl_connector import VisionLanguageConnector
from configs.config import OCRAgentConfig

logger = logging.getLogger(__name__)

class OCRAgent(BaseAgent):
    """
    Agent responsible for extracting text from document images.
    """
    
    def __init__(
        self,
        config: OCRAgentConfig,
        vision_encoder: QuantizedVisionEncoder,
        language_model: QuantizedLanguageModel,
        vl_connector: VisionLanguageConnector
    ):
        """
        Initialize the OCR agent.
        
        Args:
            config: OCR agent configuration
            vision_encoder: Vision encoder model
            language_model: Language model
            vl_connector: Vision-language connector
        """
        super().__init__("OCRAgent", config.__dict__)
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.vl_connector = vl_connector
        
        self.log_info("OCR Agent initialized")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process document regions to extract text.
        
        Args:
            inputs: Dictionary containing:
                - 'image': Full document image
                - 'regions': List of region dictionaries with 'bbox' and 'region' keys
                
        Returns:
            Dictionary with extracted text for each region
        """
        # Validate inputs
        required_keys = ['image', 'regions']
        if not self.validate_inputs(inputs, required_keys):
            return {'error': 'Invalid inputs', 'text_regions': []}
        
        self.log_info(f"Processing {len(inputs['regions'])} regions")
        
        # Extract text from each region
        text_regions = []
        
        for i, region in enumerate(inputs['regions']):
            region_id = region.get('id', i)
            region_image = region['region']
            bbox = region['bbox']
            
            self.log_debug(f"Processing region {region_id} with bbox {bbox}")
            
            # Extract text from the region
            extracted_text = self.extract_text_from_region(region_image)
            
            # Add to results
            region_result = {
                'id': region_id,
                'bbox': bbox,
                'text': extracted_text,
                'confidence': self._estimate_confidence(extracted_text)
            }
            
            text_regions.append(region_result)
        
        self.log_info(f"Text extraction completed for {len(text_regions)} regions")
        
        # Combine results
        result = {
            'text_regions': text_regions,
            'full_text': self._combine_region_texts(text_regions)
        }
        
        return result
    
    def extract_text_from_region(self, region_image: np.ndarray) -> str:
        """
        Extract text from a single region using the vision-language models.
        
        Args:
            region_image: Image of the region
            
        Returns:
            Extracted text
        """
        # Convert to PIL if needed
        if isinstance(region_image, np.ndarray):
            if len(region_image.shape) == 2:  # Grayscale
                region_image = Image.fromarray(region_image).convert('RGB')
            else:
                region_image = Image.fromarray(region_image)
        
        # Use the VL connector to process the image
        prompt = "Transcribe exactly all the text in this historical document segment:"
        extracted_text = self.vl_connector.process_image_to_text(
            region_image, 
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more deterministic outputs
            top_p=0.95
        )
        
        # Clean up the extracted text
        clean_text = self._clean_extracted_text(extracted_text, prompt)
        
        return clean_text
    
    def _clean_extracted_text(self, text: str, prompt: str) -> str:
        """
        Clean up the extracted text by removing the prompt and any irrelevant text.
        
        Args:
            text: Raw extracted text
            prompt: Prompt used for extraction
            
        Returns:
            Cleaned text
        """
        # Remove the prompt if it appears in the output
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        # Remove any common prefixes that the model might generate
        prefixes_to_remove = [
            "Here's the transcription:",
            "Transcription:",
            "Text:",
            "The text reads:",
            "I see the following text:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text
    
    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate the confidence score for the extracted text.
        
        Args:
            text: Extracted text
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # This is a simplified confidence estimation
        # A real implementation would be more sophisticated
        
        # Longer texts generally have more content to be confident about
        length_factor = min(len(text) / 100, 1.0)
        
        # Check for uncertainty markers in the text
        uncertainty_markers = ["[?]", "[unclear]", "(?)", "[illegible]"]
        uncertainty_count = sum(text.count(marker) for marker in uncertainty_markers)
        uncertainty_factor = max(0, 1.0 - (uncertainty_count * 0.1))
        
        # Combined score
        confidence = length_factor * uncertainty_factor
        
        # Impose a minimum confidence threshold
        return max(self.config["confidence_threshold"], confidence)
    
    def _combine_region_texts(self, text_regions: List[Dict[str, Any]]) -> str:
        """
        Combine texts from multiple regions into a coherent document text.
        
        Args:
            text_regions: List of text region results
            
        Returns:
            Combined text
        """
        # Simple combination for now - in a real implementation, 
        # we'd use the bounding boxes to determine the reading order
        
        # Sort regions by vertical position (top to bottom)
        sorted_regions = sorted(text_regions, key=lambda r: r['bbox'][1])
        
        # Combine texts
        full_text = "\n\n".join(region['text'] for region in sorted_regions)
        
        return full_text