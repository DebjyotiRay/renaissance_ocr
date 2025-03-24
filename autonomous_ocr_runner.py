#!/usr/bin/env python3
"""
Autonomous runner for Renaissance OCR system.
This script automatically processes all documents in a fully automated workflow.
"""

import os
import sys
import argparse
import logging
import glob
import shutil
import json
import time
import subprocess
from typing import List, Dict, Tuple, Optional, Any
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('autonomous_ocr.log')
    ]
)

logger = logging.getLogger(__name__)

class AutonomousOCRRunner:
    """Main class for running the Renaissance OCR system autonomously."""
    
    def __init__(self, base_dir: str):
        """
        Initialize the autonomous runner.
        
        Args:
            base_dir: Base directory for the Renaissance OCR system
        """
        self.base_dir = os.path.abspath(base_dir)
        
        # Define directory structure
        self.dirs = {
            'data': os.path.join(self.base_dir, 'data'),
            'data_raw': os.path.join(self.base_dir, 'data', 'raw'),
            'data_raw_printed': os.path.join(self.base_dir, 'data', 'raw', 'printed'),
            'data_raw_handwritten': os.path.join(self.base_dir, 'data', 'raw', 'handwritten'),
            'data_processed': os.path.join(self.base_dir, 'data', 'processed'),
            'data_processed_printed': os.path.join(self.base_dir, 'data', 'processed', 'printed'),
            'data_processed_handwritten': os.path.join(self.base_dir, 'data', 'processed', 'handwritten'),
            'output': os.path.join(self.base_dir, 'output'),
            'output_printed': os.path.join(self.base_dir, 'output', 'printed'),
            'output_handwritten': os.path.join(self.base_dir, 'output', 'handwritten'),
            'models': os.path.join(self.base_dir, 'models'),
            'results': os.path.join(self.base_dir, 'results'),
        }
        
        # Ensure system path includes the base directory
        if self.base_dir not in sys.path:
            sys.path.insert(0, self.base_dir)
    
    def setup_environment(self) -> None:
        """Create necessary directory structure."""
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        
        logger.info(f"Directory structure created in {self.base_dir}")
    
    def organize_data(self, test_data_dir: str) -> Tuple[List[str], List[str]]:
        """
        Organize test data into the system directory structure.
        
        Args:
            test_data_dir: Directory containing test data
            
        Returns:
            Tuple containing lists of paths to processed printed and handwritten files
        """
        handwritten_dir = os.path.join(test_data_dir, "handwritten test only")
        
        # Process handwritten files
        handwritten_files = []
        if os.path.isdir(handwritten_dir):
            # Find all PDF files in the handwritten directory
            for file_path in glob.glob(os.path.join(handwritten_dir, "*.pdf")):
                basename = os.path.basename(file_path)
                dest_path = os.path.join(self.dirs['data_raw_handwritten'], basename)
                
                # Copy file
                shutil.copy2(file_path, dest_path)
                handwritten_files.append(dest_path)
                logger.info(f"Copied handwritten file: {basename}")
        else:
            logger.warning(f"Handwritten directory not found: {handwritten_dir}")
        
        # Process other files (printed documents)
        printed_files = []
        for file_path in glob.glob(os.path.join(test_data_dir, "*.pdf")):
            # Skip files in the handwritten directory
            if handwritten_dir in file_path:
                continue
            
            basename = os.path.basename(file_path)
            dest_path = os.path.join(self.dirs['data_raw_printed'], basename)
            
            # Copy file
            shutil.copy2(file_path, dest_path)
            printed_files.append(dest_path)
            logger.info(f"Copied printed file: {basename}")
        
        logger.info(f"Organized {len(printed_files)} printed files and {len(handwritten_files)} handwritten files")
        return printed_files, handwritten_files
    
    def preprocess_documents(self) -> Tuple[List[str], List[str]]:
        """
        Preprocess all documents using the Renaissance image processor.
        
        Returns:
            Tuple containing lists of paths to processed printed and handwritten files
        """
        try:
            # Import only when needed to avoid import errors if the environment isn't set up
            from preprocessing.image_processor import RenaissanceImageProcessor
            from configs.config import Config, PreprocessingConfig
            import cv2
            from utils.pdf_processor import PDFProcessor
            
            logger.info("Starting document preprocessing")
            
            # Initialize preprocessor with default config
            config = PreprocessingConfig()
            processor = RenaissanceImageProcessor(config)
            
            # Initialize specialized processor for handwritten documents
            handwritten_config = PreprocessingConfig()
            handwritten_config.apply_clahe = True
            handwritten_config.clahe_clip_limit = 3.0
            handwritten_config.apply_noise_reduction = True
            handwritten_processor = RenaissanceImageProcessor(handwritten_config)
            
            # Initialize PDF processor
            pdf_processor = PDFProcessor()
            
            # Collect all processed document paths
            processed_printed = []
            processed_handwritten = []
            
            # Process printed documents
            for pdf_path in glob.glob(os.path.join(self.dirs['data_raw_printed'], "*.pdf")):
                basename = os.path.basename(pdf_path)
                logger.info(f"Processing printed document: {basename}")
                
                # Extract images from PDF
                output_dir = os.path.join(self.dirs['data_processed_printed'], os.path.splitext(basename)[0])
                os.makedirs(output_dir, exist_ok=True)
                
                image_paths = pdf_processor.extract_images_from_pdf(pdf_path, output_dir)
                
                # Process each page image
                for img_path in image_paths:
                    img_basename = os.path.basename(img_path)
                    processed_path = os.path.join(output_dir, f"processed_{img_basename}")
                    
                    try:
                        # Read and preprocess image
                        image = cv2.imread(img_path)
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                            
                        processed_image = processor.preprocess(image)
                        cv2.imwrite(processed_path, processed_image)
                        processed_printed.append(processed_path)
                    except Exception as e:
                        logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            
            # Process handwritten documents
            for pdf_path in glob.glob(os.path.join(self.dirs['data_raw_handwritten'], "*.pdf")):
                basename = os.path.basename(pdf_path)
                logger.info(f"Processing handwritten document: {basename}")
                
                # Extract images from PDF
                output_dir = os.path.join(self.dirs['data_processed_handwritten'], os.path.splitext(basename)[0])
                os.makedirs(output_dir, exist_ok=True)
                
                image_paths = pdf_processor.extract_images_from_pdf(pdf_path, output_dir)
                
                # Process each page image with handwritten-specific preprocessing
                for img_path in image_paths:
                    img_basename = os.path.basename(img_path)
                    processed_path = os.path.join(output_dir, f"processed_{img_basename}")
                    
                    try:
                        # Read and preprocess image
                        image = cv2.imread(img_path)
                        if image is None:
                            logger.warning(f"Could not read image: {img_path}")
                            continue
                            
                        processed_image = handwritten_processor.preprocess(image)
                        cv2.imwrite(processed_path, processed_image)
                        processed_handwritten.append(processed_path)
                    except Exception as e:
                        logger.error(f"Error preprocessing image {img_path}: {str(e)}")
            
            logger.info(f"Preprocessing completed: {len(processed_printed)} printed pages and {len(processed_handwritten)} handwritten pages")
            return processed_printed, processed_handwritten
            
        except ImportError as e:
            logger.error(f"Import error during preprocessing: {str(e)}")
            logger.error("Make sure the Renaissance OCR system is properly installed")
            return [], []
        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            return [], []
    
    def run_ocr_processing(self) -> bool:
        """
        Run the OCR processing on all preprocessed documents.
        
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            # Import the Renaissance OCR system
            from renaissance_ocr_system import RenaissanceOCRSystem
            from configs.config import Config
            
            logger.info("Starting OCR processing")
            
            # Initialize system with default configuration for printed documents
            config = Config()
            system = RenaissanceOCRSystem(config)
            
            # Process printed documents
            printed_docs = glob.glob(os.path.join(self.dirs['data_processed_printed'], "**/*processed_*.jpg"), recursive=True)
            if printed_docs:
                logger.info(f"Processing {len(printed_docs)} printed document pages")
                system.process_batch(
                    document_paths=printed_docs,
                    output_dir=self.dirs['output_printed']
                )
                logger.info("Printed document processing completed")
            else:
                logger.warning("No processed printed documents found")
            
            # Configure system for handwritten documents
            handwritten_config = Config()
            handwritten_config.preprocessing.apply_clahe = True
            handwritten_config.preprocessing.clahe_clip_limit = 3.0
            handwritten_config.spelling_agent.language = "spanish"
            handwritten_config.spelling_agent.historical_period = "renaissance"
            
            handwritten_system = RenaissanceOCRSystem(handwritten_config)
            
            # Process handwritten documents
            handwritten_docs = glob.glob(os.path.join(self.dirs['data_processed_handwritten'], "**/*processed_*.jpg"), recursive=True)
            if handwritten_docs:
                logger.info(f"Processing {len(handwritten_docs)} handwritten document pages")
                handwritten_system.process_batch(
                    document_paths=handwritten_docs,
                    output_dir=self.dirs['output_handwritten']
                )
                logger.info("Handwritten document processing completed")
            else:
                logger.warning("No processed handwritten documents found")
            
            logger.info("OCR processing completed successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Import error during OCR processing: {str(e)}")
            logger.error("Make sure the Renaissance OCR system is properly installed")
            return False
        except Exception as e:
            logger.error(f"Error during OCR processing: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def collate_results(self) -> None:
        """Collate all results into a central location."""
        os.makedirs(self.dirs['results'], exist_ok=True)
        
        # Collect all result files
        result_files = []
        result_files.extend(glob.glob(os.path.join(self.dirs['output_printed'], "**/*_result.json"), recursive=True))
        result_files.extend(glob.glob(os.path.join(self.dirs['output_printed'], "**/*_text.txt"), recursive=True))
        result_files.extend(glob.glob(os.path.join(self.dirs['output_handwritten'], "**/*_result.json"), recursive=True))
        result_files.extend(glob.glob(os.path.join(self.dirs['output_handwritten'], "**/*_text.txt"), recursive=True))
        
        # Copy files to results directory
        for file_path in result_files:
            dest_path = os.path.join(self.dirs['results'], os.path.basename(file_path))
            shutil.copy2(file_path, dest_path)
        
        logger.info(f"Collated {len(result_files)} result files to {self.dirs['results']}")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self) -> None:
        """Create a summary report of all processed documents."""
        summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_documents": {
                "printed": len(glob.glob(os.path.join(self.dirs['data_raw_printed'], "*.pdf"))),
                "handwritten": len(glob.glob(os.path.join(self.dirs['data_raw_handwritten'], "*.pdf"))),
            },
            "total_pages": {
                "printed": len(glob.glob(os.path.join(self.dirs['data_processed_printed'], "**/*processed_*.jpg"), recursive=True)),
                "handwritten": len(glob.glob(os.path.join(self.dirs['data_processed_handwritten'], "**/*processed_*.jpg"), recursive=True)),
            },
            "results": {
                "printed": len(glob.glob(os.path.join(self.dirs['output_printed'], "**/*_result.json"), recursive=True)),
                "handwritten": len(glob.glob(os.path.join(self.dirs['output_handwritten'], "**/*_result.json"), recursive=True)),
            }
        }
        
        # Write summary report
        with open(os.path.join(self.dirs['results'], "summary_report.json"), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Create a text version of the summary report
        with open(os.path.join(self.dirs['results'], "summary_report.txt"), 'w') as f:
            f.write("Renaissance OCR Processing Summary\n")
            f.write("================================\n\n")
            f.write(f"Generated: {summary['timestamp']}\n\n")
            f.write("Document Count:\n")
            f.write(f"  Printed documents: {summary['total_documents']['printed']}\n")
            f.write(f"  Handwritten documents: {summary['total_documents']['handwritten']}\n\n")
            f.write("Page Count:\n")
            f.write(f"  Printed pages: {summary['total_pages']['printed']}\n")
            f.write(f"  Handwritten pages: {summary['total_pages']['handwritten']}\n\n")
            f.write("Result Files:\n")
            f.write(f"  Printed document results: {summary['results']['printed']}\n")
            f.write(f"  Handwritten document results: {summary['results']['handwritten']}\n\n")
            f.write("Results are available in the 'results' directory.\n")
        
        logger.info("Created summary reports")
    
    def run_full_pipeline(self, test_data_dir: str) -> bool:
        """
        Run the complete OCR pipeline from data organization to result collation.
        
        Args:
            test_data_dir: Directory containing test data
            
        Returns:
            True if processing was successful, False otherwise
        """
        try:
            logger.info(f"Starting full OCR pipeline for data in {test_data_dir}")
            
            # Step 1: Setup the environment
            self.setup_environment()
            
            # Step 2: Organize data
            self.organize_data(test_data_dir)
            
            # Step 3: Preprocess documents
            self.preprocess_documents()
            
            # Step 4: Run OCR processing
            success = self.run_ocr_processing()
            
            # Step 5: Collate results
            if success:
                self.collate_results()
                logger.info("Full OCR pipeline completed successfully")
            else:
                logger.error("OCR processing failed, results may be incomplete")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error running full pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Autonomous Renaissance OCR System Runner")
    parser.add_argument("--test-data", required=True, help="Directory containing test data")
    parser.add_argument("--base-dir", default=".", help="Base directory for the Renaissance OCR system")
    args = parser.parse_args()
    
    try:
        # Create and run the autonomous system
        runner = AutonomousOCRRunner(args.base_dir)
        success = runner.run_full_pipeline(args.test_data)
        
        if success:
            logger.info("Autonomous OCR processing completed successfully")
            return 0
        else:
            logger.error("Autonomous OCR processing failed")
            return 1
            
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())