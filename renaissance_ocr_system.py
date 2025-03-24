"""
Renaissance OCR System - Main entry point for the OCR pipeline.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import cv2
from PIL import Image
import json

from configs.config import Config
from preprocessing.image_processor import RenaissanceImageProcessor
from models.vision_encoder import QuantizedVisionEncoder
from models.language_model import QuantizedLanguageModel
from models.vl_connector import VisionLanguageConnector
from agents.ocr_agent import OCRAgent
from agents.layout_validator_agent import LayoutValidatorAgent
from agents.historical_spelling_agent import HistoricalSpellingAgent
from agents.orchestrator import AgentOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('renaissance_ocr.log')
    ]
)

logger = logging.getLogger(__name__)

class RenaissanceOCRSystem:
    """
    Main class for the Renaissance OCR system with quantized models and multi-agent workflow.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the Renaissance OCR system.
        
        Args:
            config: Optional configuration object, will use default if None
        """
        self.config = config or Config()
        self.preprocessor = None
        self.vision_encoder = None
        self.language_model = None
        self.vl_connector = None
        self.orchestrator = None
        
        logger.info("Initializing Renaissance OCR System")
        
        # Initialize all components
        self._initialize_components()
        
        logger.info("Renaissance OCR System initialized successfully")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize preprocessor
            logger.info("Initializing image preprocessor")
            self.preprocessor = RenaissanceImageProcessor(self.config.preprocessing)
            
            # Initialize vision encoder
            logger.info("Initializing vision encoder")
            self.vision_encoder = QuantizedVisionEncoder(self.config.vision_encoder)
            
            # Initialize language model
            logger.info("Initializing language model")
            self.language_model = QuantizedLanguageModel(
                self.config.language_model,
                self.config.dora
            )
            
            # Initialize vision-language connector
            logger.info("Initializing vision-language connector")
            self.vl_connector = VisionLanguageConnector(
                self.vision_encoder,
                self.language_model,
                self.config.connector
            )
            
            # Initialize agents
            logger.info("Initializing agents")
            ocr_agent = OCRAgent(
                self.config.ocr_agent,
                self.vision_encoder,
                self.language_model,
                self.vl_connector
            )
            
            layout_validator = LayoutValidatorAgent(self.config.layout_validator)
            spelling_agent = HistoricalSpellingAgent(self.config.spelling_agent)
            
            # Initialize orchestrator
            logger.info("Initializing agent orchestrator")
            self.orchestrator = AgentOrchestrator(
                ocr_agent,
                layout_validator,
                spelling_agent,
                self.config.orchestrator
            )
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def process_document(self, document_path: str) -> Dict[str, Any]:
        """
        Process a document through the OCR pipeline.
        
        Args:
            document_path: Path to the document image
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing document: {document_path}")
            
            # Load and preprocess the image
            logger.info("Loading and preprocessing document")
            image = cv2.imread(document_path)
            if image is None:
                raise ValueError(f"Failed to load image from {document_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing
            preprocessed_image = self.preprocessor.preprocess(image_rgb)
            
            # Segment the page into regions
            logger.info("Segmenting document into regions")
            segmentation_result = self.preprocessor.segment_page(preprocessed_image)
            
            # Process with agent orchestrator
            logger.info("Processing with agent orchestrator")
            result = self.orchestrator.process_document(image_rgb, segmentation_result)
            
            # Add metadata
            result["metadata"] = {
                "filename": os.path.basename(document_path),
                "total_time": time.time() - start_time,
                "image_size": image.shape[:2],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Document processing completed in {time.time() - start_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            
            # Return error result
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "filename": os.path.basename(document_path),
                    "total_time": time.time() - start_time,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
    
    def process_batch(self, document_paths: List[str], output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a batch of documents.
        
        Args:
            document_paths: List of paths to document images
            output_dir: Optional directory to save results
            
        Returns:
            List of processing results
        """
        logger.info(f"Processing batch of {len(document_paths)} documents")
        
        results = []
        
        for i, path in enumerate(document_paths):
            logger.info(f"Processing document {i+1}/{len(document_paths)}: {path}")
            
            try:
                # Process the document
                result = self.process_document(path)
                results.append(result)
                
                # Save result if output_dir is provided
                if output_dir:
                    self._save_result(result, path, output_dir)
                
            except Exception as e:
                logger.error(f"Error processing document {path}: {str(e)}")
                results.append({
                    "status": "error",
                    "error": str(e),
                    "path": path
                })
        
        logger.info(f"Batch processing completed. Processed {len(results)} documents.")
        
        return results
    
    def _save_result(self, result: Dict[str, Any], document_path: str, output_dir: str):
        """
        Save processing result to output directory.
        
        Args:
            result: Processing result
            document_path: Path to the document
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Base filename without extension
        base_name = os.path.splitext(os.path.basename(document_path))[0]
        
        # Save JSON result
        json_path = os.path.join(output_dir, f"{base_name}_result.json")
        
        # Copy result and remove any non-serializable items
        json_result = result.copy()
        if "visualization" in json_result.get("layout", {}):
            del json_result["layout"]["visualization"]
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
        
        # Save text result
        if "text" in result:
            text_path = os.path.join(output_dir, f"{base_name}_text.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(result["text"])
        
        # Save visualization if available
        if "layout" in result and "visualization" in result["layout"]:
            vis_path = os.path.join(output_dir, f"{base_name}_visualization.jpg")
            cv2.imwrite(vis_path, result["layout"]["visualization"])
        
        logger.info(f"Results saved to {output_dir}")
    
    def fine_tune(self, 
                 training_data_dir: str, 
                 output_model_dir: str,
                 num_epochs: int = 3,
                 batch_size: int = 4,
                 learning_rate: float = 2e-5):
        """
        Fine-tune the language model on Renaissance documents.
        
        Args:
            training_data_dir: Directory containing training data
            output_model_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
        """
        logger.info(f"Starting fine-tuning process with {num_epochs} epochs")
        
        # This is a placeholder for the fine-tuning process
        # In a real implementation, we would:
        # 1. Load and preprocess training data
        # 2. Create a training dataset
        # 3. Set up the training loop
        # 4. Fine-tune the model
        # 5. Save the fine-tuned model
        
        logger.info("Fine-tuning not fully implemented in this version")
        logger.info("Saving current model as a placeholder")
        
        # Create output directory
        os.makedirs(output_model_dir, exist_ok=True)
        
        # Save current model
        self.language_model.save_model(output_model_dir)
        
        # Save connector
        self.vl_connector.save(os.path.join(output_model_dir, "connector.pt"))
        
        logger.info(f"Model saved to {output_model_dir}")
    
    def load_fine_tuned_model(self, model_dir: str):
        """
        Load a fine-tuned model.
        
        Args:
            model_dir: Directory containing the fine-tuned model
        """
        logger.info(f"Loading fine-tuned model from {model_dir}")
        
        try:
            # Load language model
            self.language_model.load_adapter(model_dir)
            
            # Load connector
            connector_path = os.path.join(model_dir, "connector.pt")
            if os.path.exists(connector_path):
                self.vl_connector.load(connector_path)
            
            logger.info("Fine-tuned model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise
            
def main():
    """
    Main entry point for running the Renaissance OCR system.
    """
    import argparse
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Renaissance OCR System")
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Process document command
    process_parser = subparsers.add_parser("process", help="Process a document")
    process_parser.add_argument("--input", "-i", required=True, help="Input document path")
    process_parser.add_argument("--output", "-o", default="output", help="Output directory")
    
    # Process batch command
    batch_parser = subparsers.add_parser("batch", help="Process a batch of documents")
    batch_parser.add_argument("--input-dir", "-i", required=True, help="Input directory containing documents")
    batch_parser.add_argument("--output", "-o", default="output", help="Output directory")
    batch_parser.add_argument("--pattern", "-p", default="*.jpg,*.png,*.tif,*.tiff,*.jpeg", 
                             help="File patterns to match (comma-separated)")
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune the model")
    finetune_parser.add_argument("--data", "-d", required=True, help="Training data directory")
    finetune_parser.add_argument("--output", "-o", default="fine_tuned_model", help="Output model directory")
    finetune_parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    finetune_parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    finetune_parser.add_argument("--learning-rate", "-lr", type=float, default=2e-5, help="Learning rate")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create OCR system
    system = RenaissanceOCRSystem()
    
    if args.command == "process":
        # Process a single document
        os.makedirs(args.output, exist_ok=True)
        result = system.process_document(args.input)
        system._save_result(result, args.input, args.output)
        
        print(f"Document processed. Results saved to {args.output}")
        
    elif args.command == "batch":
        # Process a batch of documents
        import glob
        
        os.makedirs(args.output, exist_ok=True)
        
        # Get all files matching the patterns
        patterns = args.pattern.split(",")
        document_paths = []
        
        for pattern in patterns:
            pattern_paths = glob.glob(os.path.join(args.input_dir, pattern.strip()))
            document_paths.extend(pattern_paths)
        
        print(f"Found {len(document_paths)} documents to process")
        
        # Process batch
        results = system.process_batch(document_paths, args.output)
        
        print(f"Batch processing completed. Results saved to {args.output}")
        
    elif args.command == "finetune":
        # Fine-tune the model
        system.fine_tune(
            args.data,
            args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"Fine-tuning completed. Model saved to {args.output}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()