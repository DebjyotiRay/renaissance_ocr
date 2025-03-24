#!/usr/bin/env python3
"""
Command-line interface for Renaissance OCR system.
"""

import argparse
import os
import sys
import glob
import logging
import json
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('renaissance_ocr_cli.log')
    ]
)

logger = logging.getLogger(__name__)

from renaissance_ocr_system import RenaissanceOCRSystem
from configs.config import Config
from utils.pdf_processor import PDFProcessor
from utils.visualizer import OCRVisualizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Renaissance OCR System CLI")
    
    # Main arguments
    parser.add_argument('--input', '-i', required=True, help='Input file or directory')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process documents')
    process_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    process_parser.add_argument('--format', choices=['json', 'text', 'html', 'all'], default='all', 
                              help='Output format')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate OCR results')
    eval_parser.add_argument('--ground-truth', '-g', required=True, help='Ground truth text file or directory')
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser('finetune', help='Fine-tune the model')
    finetune_parser.add_argument('--train-data', '-d', required=True, help='Training data directory')
    finetune_parser.add_argument('--model-output', '-m', default='fine_tuned_model', 
                                help='Output model directory')
    finetune_parser.add_argument('--epochs', '-e', type=int, default=3, help='Number of training epochs')
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from file or use default.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration object
    """
    if config_path and os.path.exists(config_path):
        try:
            # Load config from JSON file
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Convert to Config object
            # This is a simplified approach - a real implementation would
            # need to properly map the JSON structure to the Config class
            config = Config()
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.warning("Using default configuration")
    
    # Use default configuration
    return Config()

def find_documents(input_path: str) -> List[str]:
    """
    Find documents to process.
    
    Args:
        input_path: Input file or directory
        
    Returns:
        List of document paths
    """
    if os.path.isfile(input_path):
        # Single file
        return [input_path]
    elif os.path.isdir(input_path):
        # Directory - find all image and PDF files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.pdf']
        documents = []
        
        for ext in extensions:
            documents.extend(glob.glob(os.path.join(input_path, ext)))
            # Also search in subdirectories
            documents.extend(glob.glob(os.path.join(input_path, '**', ext), recursive=True))
        
        return sorted(documents)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

def process_documents(documents: List[str], ocr_system: RenaissanceOCRSystem, 
                     output_dir: str, visualize: bool, output_format: str) -> List[Dict[str, Any]]:
    """
    Process a list of documents.
    
    Args:
        documents: List of document paths
        ocr_system: OCR system
        output_dir: Output directory
        visualize: Whether to generate visualizations
        output_format: Output format
        
    Returns:
        List of processing results
    """
    results = []
    
    for i, doc_path in enumerate(documents):
        logger.info(f"Processing document {i+1}/{len(documents)}: {doc_path}")
        
        try:
            # Handle PDFs
            if doc_path.lower().endswith('.pdf'):
                logger.info(f"Processing PDF: {doc_path}")
                
                # Extract pages from PDF
                pdf_processor = PDFProcessor()
                image_paths = pdf_processor.extract_images_from_pdf(doc_path)
                
                # Process each page
                pdf_results = []
                
                for j, img_path in enumerate(image_paths):
                    logger.info(f"Processing PDF page {j+1}/{len(image_paths)}")
                    
                    # Process the page
                    result = ocr_system.process_document(img_path)
                    pdf_results.append(result)
                    
                    # Save page result
                    page_dir = os.path.join(output_dir, os.path.basename(doc_path).replace('.pdf', ''), f"page_{j+1:03d}")
                    os.makedirs(page_dir, exist_ok=True)
                    
                    # Save outputs based on format
                    save_outputs(result, img_path, page_dir, visualize, output_format)
                
                # Create combined PDF result
                pdf_result = {
                    'filename': os.path.basename(doc_path),
                    'num_pages': len(image_paths),
                    'pages': pdf_results,
                    'combined_text': "\n\n".join(r.get('text', '') for r in pdf_results)
                }
                
                results.append(pdf_result)
                
                # Create a combined result file
                combined_dir = os.path.join(output_dir, os.path.basename(doc_path).replace('.pdf', ''))
                with open(os.path.join(combined_dir, 'combined_text.txt'), 'w', encoding='utf-8') as f:
                    f.write(pdf_result['combined_text'])
                
            else:
                # Process single image document
                result = ocr_system.process_document(doc_path)
                results.append(result)
                
                # Create output directory
                doc_output_dir = os.path.join(output_dir, os.path.basename(doc_path).split('.')[0])
                os.makedirs(doc_output_dir, exist_ok=True)
                
                # Save outputs based on format
                save_outputs(result, doc_path, doc_output_dir, visualize, output_format)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {str(e)}")
            results.append({
                'filename': os.path.basename(doc_path),
                'status': 'error',
                'error': str(e)
            })
    
    return results