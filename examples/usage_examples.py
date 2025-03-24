"""
Examples of how to use the Renaissance OCR system.
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renaissance_ocr_system import RenaissanceOCRSystem
from configs.config import Config
from utils.visualizer import OCRVisualizer
from utils.pdf_processor import PDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def example_1_simple_document():
    """Example 1: Process a simple document."""
    print("\n=== Example 1: Process a Simple Document ===\n")
    
    # Initialize OCR system with default configuration
    system = RenaissanceOCRSystem()
    
    # Define document path
    document_path = "data/raw/sample_document.jpg"
    
    # Process the document
    start_time = time.time()
    result = system.process_document(document_path)
    elapsed_time = time.time() - start_time
    
    # Print results
    print(f"Processing completed in {elapsed_time:.2f} seconds")
    print(f"Status: {result.get('status', 'unknown')}")
    
    if 'text' in result:
        print("\nExtracted Text:")
        print("--------------")
        print(result['text'][:500] + "..." if len(result['text']) > 500 else result['text'])
    
    # Save the result
    output_dir = "examples/output/example_1"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save text output
    with open(os.path.join(output_dir, "text.txt"), 'w', encoding='utf-8') as f:
        f.write(result.get('text', ''))
    
    # Visualize the result
    if os.path.exists(document_path):
        image = cv2.imread(document_path)
        if image is not None:
            # Generate visualization
            OCRVisualizer.generate_full_report(result, image, output_dir)
            print(f"\nVisualization saved to {output_dir}")

def example_2_batch_processing():
    """Example 2: Batch process multiple documents."""
    print("\n=== Example 2: Batch Process Multiple Documents ===\n")
    
    # Initialize OCR system
    system = RenaissanceOCRSystem()
    
    # Define document paths
    document_dir = "data/raw/batch"
    output_dir = "examples/output/example_2"
    
    # Find all image files in the directory
    import glob
    document_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
        document_paths.extend(glob.glob(os.path.join(document_dir, ext)))
    
    # Process the documents
    print(f"Processing {len(document_paths)} documents...")
    start_time = time.time()
    
    results = system.process_batch(document_paths, output_dir)
    
    elapsed_time = time.time() - start_time
    
    # Print results summary
    print(f"Batch processing completed in {elapsed_time:.2f} seconds")
    print(f"Processed {len(results)} documents")
    print(f"Results saved to {output_dir}")

def example_3_pdf_processing():
    """Example 3: Process a PDF document."""
    print("\n=== Example 3: Process a PDF Document ===\n")
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Define PDF path
    pdf_path = "data/raw/sample_document.pdf"
    output_dir = "examples/output/example_3"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images from PDF
    print("Extracting pages from PDF...")
    image_paths = pdf_processor.extract_images_from_pdf(pdf_path, output_dir)
    
    print(f"Extracted {len(image_paths)} pages")
    
    # Initialize OCR system
    system = RenaissanceOCRSystem()
    
    # Process each image
    print("Processing extracted pages...")
    all_text = ""
    
    for i, image_path in enumerate(image_paths):
        print(f"Processing page {i+1}/{len(image_paths)}")
        
        result = system.process_document(image_path)
        
        # Save page result
        page_dir = os.path.join(output_dir, f"page_{i+1:03d}")
        os.makedirs(page_dir, exist_ok=True)
        
        with open(os.path.join(page_dir, "text.txt"), 'w', encoding='utf-8') as f:
            f.write(result.get('text', ''))
        
        # Add to combined text
        all_text += result.get('text', '') + "\n\n"
    
    # Save combined text
    with open(os.path.join(output_dir, "combined_text.txt"), 'w', encoding='utf-8') as f:
        f.write(all_text)
    
    print(f"PDF processing completed. Results saved to {output_dir}")

def example_4_custom_configuration():
    """Example 4: Use custom configuration."""
    print("\n=== Example 4: Use Custom Configuration ===\n")
    
    # Create custom configuration
    config = Config()
    
    # Modify preprocessing parameters
    config.preprocessing.dpi = 400
    config.preprocessing.apply_clahe = True
    config.preprocessing.clahe_clip_limit = 3.0
    
    # Modify OCR agent configuration
    config.ocr_agent.confidence_threshold = 0.6
    
    # Initialize OCR system with custom configuration
    system = RenaissanceOCRSystem(config)
    
    # Define document path
    document_path = "data/raw/sample_document.jpg"
    
    # Process the document
    print("Processing document with custom configuration...")
    result = system.process_document(document_path)
    
    # Save the result
    output_dir = "examples/output/example_4"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "text.txt"), 'w', encoding='utf-8') as f:
        f.write(result.get('text', ''))
    
    print(f"Processing completed. Results saved to {output_dir}")

def example_5_fine_tuning():
    """Example 5: Fine-tune the model."""
    print("\n=== Example 5: Fine-tune the Model ===\n")
    
    # Initialize OCR system
    system = RenaissanceOCRSystem()
    
    # Define training data directory and output directory
    train_data_dir = "data/training"
    output_model_dir = "examples/output/example_5/fine_tuned_model"
    
    # Fine-tune the model
    print("Fine-tuning the model...")
    print("Note: This is a simplified demonstration. Full fine-tuning would require much more data.")
    
    system.fine_tune(
        train_data_dir,
        output_model_dir,
        num_epochs=1,  # Use a small number for demonstration
        batch_size=2,
        learning_rate=1e-5
    )
    
    print(f"Fine-tuning completed. Model saved to {output_model_dir}")
    
    # Test the fine-tuned model
    print("\nTesting fine-tuned model...")
    
    # Load the fine-tuned model
    fine_tuned_system = RenaissanceOCRSystem()
    fine_tuned_system.load_fine_tuned_model(output_model_dir)
    
    # Define test document
    test_document = "data/raw/test_document.jpg"
    
    # Process with fine-tuned model
    result = fine_tuned_system.process_document(test_document)
    
    # Save the result
    test_output_dir = os.path.join("examples/output/example_5/test_results")
    os.makedirs(test_output_dir, exist_ok=True)
    
    with open(os.path.join(test_output_dir, "text.txt"), 'w', encoding='utf-8') as f:
        f.write(result.get('text', ''))
    
    print(f"Testing completed. Results saved to {test_output_dir}")

def run_examples():
    """Run all examples."""
    # Create necessary directories
    os.makedirs("examples/output", exist_ok=True)
    
    # Run examples
    print("Running Renaissance OCR System Examples")
    print("======================================")
    
    try:
        example_1_simple_document()
    except Exception as e:
        print(f"Error in Example 1: {str(e)}")
    
    try:
        example_2_batch_processing()
    except Exception as e:
        print(f"Error in Example 2: {str(e)}")
    
    try:
        example_3_pdf_processing()
    except Exception as e:
        print(f"Error in Example 3: {str(e)}")
    
    try:
        example_4_custom_configuration()
    except Exception as e:
        print(f"Error in Example 4: {str(e)}")
    
    try:
        example_5_fine_tuning()
    except Exception as e:
        print(f"Error in Example 5: {str(e)}")
    
    print("\nAll examples completed!")

if __name__ == "__main__":
    run_examples()