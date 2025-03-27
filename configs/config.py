"""
Configuration settings for the Renaissance OCR system.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple

@dataclass
class VisionEncoderConfig:
    model_name: str = "openai/clip-vit-large-patch14"
    load_in_8bit: bool = True
    device_map: str = "auto"
    max_resolution: Tuple[int, int] = (896, 896)
    cache_dir: Optional[str] = None

@dataclass
class LanguageModelConfig:
    model_name: str = "Qwen/Qwen2-VL-7B"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    device_map: str = "auto"
    cache_dir: Optional[str] = None
    max_context_length: int = 4096

@dataclass
class ConnectorConfig:
    precision: str = "float16"  # "float16" or "float32"

@dataclass
class DoRAConfig:
    r: int = 16
    lora_alpha: int = 8
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    lora_dropout: float = 0.1
    bias: str = "none"

@dataclass
class PreprocessingConfig:
    dpi: int = 600
    apply_binarization: bool = True
    apply_clahe: bool = True
    apply_noise_reduction: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
@dataclass
class OCRAgentConfig:
    confidence_threshold: float = 0.7
    min_region_size: Tuple[int, int] = (50, 20)
    max_regions: int = 50

@dataclass
class LayoutValidatorConfig:
    validation_methods: List[str] = field(default_factory=lambda: [
        "overlap_removal", "alignment_correction", "size_filtering"
    ])
    overlap_threshold: float = 0.3

@dataclass
class SpellingAgentConfig:
    language: str = "spanish"  # or other historical languages
    historical_period: str = "renaissance"
    use_contextual_correction: bool = True
    max_edit_distance: int = 3
    confidence_threshold: float = 0.8
    custom_dictionary_path: Optional[str] = None

@dataclass
class AgentOrchestratorConfig:
    max_turns: int = 7
    debug_mode: bool = True

@dataclass
class EvaluationConfig:
    calculate_cer: bool = True
    calculate_wer: bool = True
    calculate_layout_f1: bool = True
    calculate_historical_accuracy: bool = True
    reference_dir: Optional[str] = None

@dataclass
class SystemConfig:
    project_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: Optional[str] = None
    raw_data_dir: Optional[str] = None
    processed_data_dir: Optional[str] = None
    output_dir: Optional[str] = None
    models_dir: Optional[str] = None
    use_gpu: bool = True
    
    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = os.path.join(self.project_dir, "data")
        if self.raw_data_dir is None:
            self.raw_data_dir = os.path.join(self.data_dir, "raw")
        if self.processed_data_dir is None:
            self.processed_data_dir = os.path.join(self.data_dir, "processed")
        if self.output_dir is None:
            self.output_dir = os.path.join(self.project_dir, "output")
        if self.models_dir is None:
            self.models_dir = os.path.join(self.project_dir, "models")
        
        # Create directories if they don't exist
        for dir_path in [self.data_dir, self.raw_data_dir, self.processed_data_dir, 
                        self.output_dir, self.models_dir]:
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    language_model: LanguageModelConfig = field(default_factory=LanguageModelConfig)
    connector: ConnectorConfig = field(default_factory=ConnectorConfig)
    dora: DoRAConfig = field(default_factory=DoRAConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    ocr_agent: OCRAgentConfig = field(default_factory=OCRAgentConfig)
    layout_validator: LayoutValidatorConfig = field(default_factory=LayoutValidatorConfig)
    spelling_agent: SpellingAgentConfig = field(default_factory=SpellingAgentConfig)
    orchestrator: AgentOrchestratorConfig = field(default_factory=AgentOrchestratorConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

# Default configuration
DEFAULT_CONFIG = Config()