"""
Vision encoder model for Renaissance document processing.
"""

import torch
import logging
from transformers import CLIPVisionModel, CLIPImageProcessor
from typing import Union, Optional, Dict, List, Tuple, Any
from PIL import Image
import numpy as np

from configs.config import VisionEncoderConfig

logger = logging.getLogger(__name__)

class QuantizedVisionEncoder:
    """
    CLIP Vision Encoder model with 8-bit quantization for memory efficiency.
    """
    
    def __init__(self, config: VisionEncoderConfig):
        """
        Initialize the quantized vision encoder.
        
        Args:
            config: Configuration for the vision encoder
        """
        self.config = config
        logger.info(f"Loading vision encoder: {config.model_name}")
        
        # Load the vision encoder with quantization
        self.model = CLIPVisionModel.from_pretrained(
            config.model_name,
            load_in_8bit=config.load_in_8bit,
            device_map=config.device_map,
            cache_dir=config.cache_dir,
        )
        
        # Load the image processor
        self.processor = CLIPImageProcessor.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )
        
        logger.info("Vision encoder loaded successfully")
    
    def preprocess_image(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, torch.Tensor]:
        """
        Preprocess an image for the vision encoder.
        
        Args:
            image: Image to process (file path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed image as a dictionary with 'pixel_values' tensor
        """
        if isinstance(image, str):
            logger.debug(f"Loading image from {image}")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        # Apply CLIP preprocessing
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        return inputs
    
    def encode(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Encode an image to obtain visual features.
        
        Args:
            image: Image to encode
            
        Returns:
            Visual features tensor
        """
        inputs = self.preprocess_image(image)
        
        # Process with the vision encoder
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Return the last hidden state
        return outputs.last_hidden_state
    
    def get_pooled_output(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Get the pooled output for an image.
        
        Args:
            image: Image to encode
            
        Returns:
            Pooled output tensor
        """
        inputs = self.preprocess_image(image)
        
        # Process with the vision encoder
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Return the pooled output
        return outputs.pooler_output
    
    def encode_batch(self, images: List[Union[str, np.ndarray, Image.Image]]) -> torch.Tensor:
        """
        Encode a batch of images.
        
        Args:
            images: List of images to encode
            
        Returns:
            Batch of visual features
        """
        # Convert all images to PIL
        pil_images = []
        for img in images:
            if isinstance(img, str):
                pil_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert('RGB'))
            else:
                pil_images.append(img)
        
        # Process batch
        inputs = self.processor(images=pil_images, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Process with the vision encoder
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state