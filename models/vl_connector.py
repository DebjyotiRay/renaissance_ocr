"""
Vision-Language Connector module that bridges the vision encoder with the language model.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Dict, Tuple

from configs.config import ConnectorConfig
from models.vision_encoder import QuantizedVisionEncoder
from models.language_model import QuantizedLanguageModel

logger = logging.getLogger(__name__)

class VisionLanguageConnector:
    """
    Connector that projects vision features into the language model's embedding space.
    """
    
    def __init__(
        self, 
        vision_encoder: QuantizedVisionEncoder,
        language_model: QuantizedLanguageModel,
        config: ConnectorConfig
    ):
        """
        Initialize the vision-language connector.
        
        Args:
            vision_encoder: The vision encoder model
            language_model: The language model
            config: Connector configuration
        """
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        
        # Get hidden sizes from models
        vision_hidden_size = self.vision_encoder.model.config.hidden_size
        language_hidden_size = self.language_model.model.config.hidden_size
        
        logger.info(f"Creating connector: vision_hidden={vision_hidden_size}, language_hidden={language_hidden_size}")
        
        # Create a linear projection layer to map vision features to language space
        dtype = torch.float16 if config.precision == "float16" else torch.float32
        
        # The connector is a simple linear layer for projecting vision features
        self.projection = nn.Linear(vision_hidden_size, language_hidden_size).to(dtype)
        
        # Get the device from the language model to ensure compatibility
        device = next(language_model.model.parameters()).device
        self.projection = self.projection.to(device)
        
        logger.info(f"Vision-Language connector initialized with {config.precision} precision")
    
    def connect(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Project vision features into the language model's embedding space.
        
        Args:
            vision_features: Features from the vision encoder
            
        Returns:
            Projected features compatible with the language model
        """
        # Ensure correct device
        device = self.projection.weight.device
        vision_features = vision_features.to(device)
        
        # Apply the projection
        projected_features = self.projection(vision_features)
        
        return projected_features
    
    def process_image_to_text(
        self, 
        image, 
        prompt: str = "Transcribe the text in this historical document:",
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Process an image through the vision-language pipeline to generate text.
        
        Args:
            image: The input image
            prompt: Text prompt to guide the generation
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text based on the image
        """
        # Extract visual features
        vision_features = self.vision_encoder.encode(image)
        
        # Project to language space
        projected_features = self.connect(vision_features)
        
        # This is where the actual connection would happen
        # For realistic implementation, this would involve embedding the
        # visual features into the language model's inputs
        # Here we're using a simplified approach
        
        # The proper implementation would depend on the specific architecture of 
        # the language model being used (e.g., Qwen2-VL has specific methods for this)
        
        # Simplified approach: use the language model with the prompted text
        # In a real implementation, we'd inject the visual features
        result = self.language_model.generate(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        return result
    
    def save(self, path: str):
        """
        Save the connector weights.
        
        Args:
            path: Path to save the connector
        """
        logger.info(f"Saving connector to {path}")
        torch.save(self.projection.state_dict(), path)
    
    def load(self, path: str):
        """
        Load connector weights.
        
        Args:
            path: Path to the saved connector
        """
        logger.info(f"Loading connector from {path}")
        state_dict = torch.load(path, map_location="cpu")
        self.projection.load_state_dict(state_dict)
        
        # Move to the correct device after loading
        device = next(self.language_model.model.parameters()).device
        self.projection = self.projection.to(device)