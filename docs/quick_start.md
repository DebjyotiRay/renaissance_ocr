# Renaissance OCR: Quick Start Guide

This guide will help you get started with the Renaissance OCR system for processing historical documents.

## Installation

### Prerequisites

- Python 3.8+ installed
- CUDA-capable GPU (recommended)
- PyTorch-compatible system

### Install via pip

```bash
# Install from PyPI
pip install renaissance-ocr

# Or install in development mode from the repository
git clone https://github.com/DebjyotiRay/renaissance-ocr.git
cd renaissance-ocr
pip install -e .
```

## Basic Usage

### From the Command Line

The simplest way to use Renaissance OCR is through the command-line interface:

```bash
# Process a single document
renaissance-ocr --input your_document.jpg --output results process

# Generate visualizations
renaissance-ocr --input your_document.jpg --output results process --visualize
```

### As a Python Library

```python
from renaissance_ocr_system import RenaissanceOCRSystem

# Initialize the system
ocr_system = RenaissanceOCRSystem()

# Process a document
result = ocr_system.process_document("your_document.jpg")

# Get the OCR text
print(result["text"])
```

## Working with Different Document Types

### JPG/PNG Images

```bash
renaissance-ocr --input document.jpg --output results process
```

### PDF Documents

```bash
renaissance-ocr --input document.pdf --output results process
```

### Batch Processing

```bash
# Process all documents in a directory
renaissance-ocr --input documents_folder/ --output results process
```

## Understanding Results

The system outputs multiple files in the specified output directory:

- `text.txt`: The extracted text
- `result.json`: Full results including region information and confidence scores
- `regions.jpg`: Visualization of detected text regions
- `confidence_heatmap.jpg`: Visualization of confidence scores
- `report.html`: HTML report with all visualizations and results

## Advanced Usage

### Custom Configuration

You can customize the system behavior by creating a configuration file:

```python
# config.py
from renaissance_ocr.configs.config import Config

# Create custom configuration
config = Config()

# Modify preprocessing parameters
config.preprocessing.dpi = 400
config.preprocessing.apply_clahe = True

# Modify OCR agent parameters
config.ocr_agent.confidence_threshold = 0.6

# Save configuration
import json
with open("custom_config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)
```

Then use the configuration:

```bash
renaissance-ocr --input document.jpg --output results --config custom_config.json process
```

### Evaluating OCR Quality

To evaluate OCR accuracy against ground truth:

```bash
renaissance-ocr --input document.jpg --output results --ground-truth ground_truth.txt evaluate
```

### Fine-tuning for Specific Document Types

For improved accuracy on specific types of Renaissance documents:

```bash
# Prepare training data with document images and transcriptions
renaissance-ocr --input training_docs/ --output model/ --train-data training_data/ finetune
```

After fine-tuning:

```python
# Use the fine-tuned model
ocr_system = RenaissanceOCRSystem()
ocr_system.load_fine_tuned_model("model/")
result = ocr_system.process_document("test_document.jpg")
```

## Troubleshooting

### Memory Issues

If you encounter GPU memory errors:

1. Reduce batch size in the configuration
2. Use more aggressive quantization settings
3. Process documents at a lower resolution

### Accuracy Issues

If OCR accuracy is poor:

1. Make sure the document quality is sufficient
2. Try adjusting preprocessing parameters
3. Consider fine-tuning the model on similar documents

### Runtime Errors

If you encounter runtime errors:

1. Check your PyTorch and CUDA setup
2. Ensure all dependencies are correctly installed
3. Check input document format and readability
4. Look at the logs in `renaissance_ocr.log`

## Next Steps

- See the full [documentation](README.md) for detailed information
- Explore [example scripts](examples/usage_examples.py) for advanced usage
- See [the model architecture](PROJECT_STRUCTURE.md) for technical details

## Getting Help

If you encounter issues or have questions:

- Check the documentation
- Open an issue on the GitHub repository
- Contact the project maintainers at [email]