# run_ocr_on_pdf.py

import os
import sys
from pathlib import Path

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Define input and output paths
input_pdf = "data/raw/printed/Buencio.pdf"
output_file = os.path.join(output_dir, "Buencio.txt")

# Check if input file exists
if not os.path.exists(input_pdf):
    print(f"Error: Input file not found: {input_pdf}")
    sys.exit(1)

print(f"Processing {input_pdf}...")

# Use the command-line interface with subprocess
import subprocess

try:
    # Run the OCR process
    result = subprocess.run(
        ["python", "-m", "renaissance_ocr_system", "process", "--input", input_pdf, "--output", output_dir],
        check=True,
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    # Check if output was generated
    if os.path.exists(os.path.join(output_dir, Path(input_pdf).stem, "text.txt")):
        # Copy the text file to the desired location
        with open(os.path.join(output_dir, Path(input_pdf).stem, "text.txt"), 'r', encoding='utf-8') as f:
            text = f.read()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully processed PDF. Text saved to {output_file}")
    else:
        print("Processing completed but no text output was found.")
        
except subprocess.CalledProcessError as e:
    print(f"Error running OCR process: {e}")
    print("Error output:")
    print(e.stderr)
    
    # Try alternative approach
    print("\nTrying alternative approach...")
    try:
        # Manual processing with minimal dependencies
        from pdf2image import convert_from_path
        import pytesseract
        from PIL import Image
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = convert_from_path(input_pdf, dpi=300)
        
        print(f"Extracted {len(images)} pages")
        
        # Process each image with pytesseract
        all_text = []
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}")
            text = pytesseract.image_to_string(image, lang='spa')  # Use Spanish language model
            all_text.append(text)
        
        # Combine text and save
        combined_text = "\n\n".join(all_text)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        print(f"Processed using alternative method. Text saved to {output_file}")
        
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        print("Please install the missing dependencies:")
        print("pip install pdf2image pytesseract")
        print("And make sure Tesseract OCR is installed on your system")