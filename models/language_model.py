"""
Language model for OCR with 4-bit quantization.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from configs.config import LanguageModelConfig, DoRAConfig

logger = logging.getLogger(__name__)

class QuantizedLanguageModel:
    """
    Quantized language model with 4-bit weights for Renaissance document OCR.
    """
    
    def __init__(self, language_config: LanguageModelConfig, dora_config: Optional[DoRAConfig] = None):
        """
        Initialize the quantized language model.
        
        Args:
            language_config: Configuration for the language model
            dora_config: Optional DoRA configuration for fine-tuning
        """
        self.config = language_config
        self.dora_config = dora_config
        
        logger.info(f"Loading language model: {language_config.model_name}")
        
        # Create BitsAndBytes configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=language_config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, language_config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=language_config.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=language_config.bnb_4bit_quant_type
        )
        
        # Load the model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            language_config.model_name,
            quantization_config=bnb_config,
            device_map=language_config.device_map,
            cache_dir=language_config.cache_dir,
        )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_config.model_name,
            cache_dir=language_config.cache_dir,
        )
        
        # Apply DoRA if provided
        if dora_config is not None:
            self._apply_dora(dora_config)
        
        logger.info("Language model loaded successfully")
    
    def _apply_dora(self, dora_config: DoRAConfig):
        """
        Apply DoRA (Decomposed Rank Adaptation) for fine-tuning.
        
        Args:
            dora_config: DoRA configuration
        """
        logger.info("Applying DoRA for parameter-efficient fine-tuning")
        
        # Convert DoRA config to LoRA config (for now, as DoRA is not directly available)
        # When proper DoRA implementation is available in PEFT, this should be updated
        lora_config = LoraConfig(
            r=dora_config.r,
            lora_alpha=dora_config.lora_alpha,
            target_modules=dora_config.target_modules,
            lora_dropout=dora_config.lora_dropout,
            bias=dora_config.bias,
            task_type="CAUSAL_LM"
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA adaptation
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("DoRA applied successfully")
    
    def generate(self, 
                prompt: str, 
                max_length: Optional[int] = None,
                temperature: float = 0.7,
                top_p: float = 0.9,
                repetition_penalty: float = 1.1) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Penalty for repetition
            
        Returns:
            Generated text
        """
        if max_length is None:
            max_length = self.config.max_context_length
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def process_visual_features(self, 
                               visual_features: torch.Tensor, 
                               prompt: str) -> str:
        """
        Process visual features with the language model.
        
        Args:
            visual_features: Visual features from the vision encoder
            prompt: Text prompt to guide generation
            
        Returns:
            Generated text based on visual features
        """
        # This is a simplified placeholder
        # For actual VL models, this would involve proper feature fusion
        # Here we're simulating this process
        
        # Construct a prompt that would typically be used with the visual features
        full_prompt = f"Describe the text in this document: {prompt}"
        
        return self.generate(full_prompt)
    
    def save_model(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        logger.info(f"Saving model to {path}")
        
        # Only save the LoRA/DoRA adapter weights if we're using PEFT
        if self.dora_config is not None:
            self.model.save_pretrained(path)
            logger.info("Saved PEFT adapter weights")
        else:
            # For a full model, this would be very large, so typically
            # we'd only save the adapters in a fine-tuned scenario
            self.model.save_pretrained(path)
            logger.info("Saved full model weights")
        
        # Save the tokenizer
        self.tokenizer.save_pretrained(path)
        logger.info("Saved tokenizer")
    
    def load_adapter(self, path: str):
        """
        Load a trained adapter from disk.
        
        Args:
            path: Path to the saved adapter
        """
        if self.dora_config is None:
            logger.warning("No DoRA configuration set, but trying to load adapter")
        
        logger.info(f"Loading adapter from {path}")
        self.model.load_adapter(path)
        logger.info("Adapter loaded successfully")