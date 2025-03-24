"""
Visualization utilities for Renaissance OCR system.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OCRVisualizer:
    """
    Utility class for visualizing OCR results.
    """
    
    @staticmethod
    def visualize_regions(image: np.ndarray, regions: List[Dict[str, Any]], 
                         with_text: bool = True) -> np.ndarray:
        """
        Visualize detected regions on an image.
        
        Args:
            image: Input image
            regions: List of region dictionaries
            with_text: Whether to include OCR text
            
        Returns:
            Visualization image
        """
        # Create a copy of the image for visualization
        if len(image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw each region
        for i, region in enumerate(regions):
            # Get coordinates
            if 'bbox' not in region:
                continue
                
            x, y, w, h = region['bbox']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for valid regions
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            
            # Add region ID
            cv2.putText(vis_image, f"R{i}", (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Add text if available and requested
            if with_text and 'text' in region:
                text = region['text']
                # Truncate text if too long
                if len(text) > 30:
                    text = text[:27] + "..."
                    
                # Add text below the region
                cv2.putText(vis_image, text, (x, y+h+20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        return vis_image
    
    @staticmethod
    def create_side_by_side_comparison(original_image: np.ndarray, 
                                      processed_image: np.ndarray,
                                      title: str = "Original vs. Processed") -> np.ndarray:
        """
        Create a side-by-side comparison of original and processed images.
        
        Args:
            original_image: Original image
            processed_image: Processed image
            title: Comparison title
            
        Returns:
            Comparison image
        """
        # Ensure both images are in color
        if len(original_image.shape) == 2:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_rgb = original_image.copy()
            
        if len(processed_image.shape) == 2:
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        else:
            processed_rgb = processed_image.copy()
        
        # Resize images to the same height if needed
        h1, w1 = original_rgb.shape[:2]
        h2, w2 = processed_rgb.shape[:2]
        
        if h1 != h2:
            # Resize to match the smaller height
            target_height = min(h1, h2)
            scale1 = target_height / h1
            scale2 = target_height / h2
            
            original_rgb = cv2.resize(original_rgb, (int(w1 * scale1), target_height))
            processed_rgb = cv2.resize(processed_rgb, (int(w2 * scale2), target_height))
        
        # Create side-by-side image
        h, w1, _ = original_rgb.shape
        h, w2, _ = processed_rgb.shape
        comparison = np.zeros((h + 50, w1 + w2 + 10, 3), dtype=np.uint8) + 255
        
        # Add title
        cv2.putText(comparison, title, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Add images
        comparison[50:50+h, 0:w1] = original_rgb
        comparison[50:50+h, w1+10:w1+10+w2] = processed_rgb
        
        # Add labels
        cv2.putText(comparison, "Original", (10, h+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(comparison, "Processed", (w1+10, h+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        return comparison
    
    @staticmethod
    def create_correction_visualization(original_text: str, corrected_text: str) -> np.ndarray:
        """
        Create a visualization of text corrections.
        
        Args:
            original_text: Original OCR text
            corrected_text: Corrected text
            
        Returns:
            Visualization image
        """
        # Create a blank image
        width, height = 1200, 800
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add title
        cv2.putText(img, "Text Correction Visualization", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Split texts into lines for better display
        def split_into_lines(text, max_chars=60):
            words = text.split()
            lines = []
            current_line = ""
            
            for word in words:
                if len(current_line) + len(word) + 1 <= max_chars:
                    current_line += " " + word if current_line else word
                else:
                    lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
                
            return lines
        
        original_lines = split_into_lines(original_text)
        corrected_lines = split_into_lines(corrected_text)
        
        # Draw original text
        cv2.putText(img, "Original Text:", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        
        y_pos = 120
        for line in original_lines:
            cv2.putText(img, line, (40, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += 30
        
        # Draw corrected text
        cv2.putText(img, "Corrected Text:", (20, y_pos + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2, cv2.LINE_AA)
        
        y_pos += 60
        for line in corrected_lines:
            cv2.putText(img, line, (40, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
            y_pos += 30
        
        return img
    
    @staticmethod
    def plot_ocr_confidence_heatmap(image: np.ndarray, regions: List[Dict[str, Any]], 
                                  output_path: Optional[str] = None):
        """
        Create a confidence heatmap for OCR regions.
        
        Args:
            image: Input image
            regions: List of region dictionaries with confidence scores
            output_path: Optional path to save the heatmap
        """
        # Create a copy of the image
        if len(image.shape) == 2:  # Grayscale
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Create a transparent overlay for the heatmap
        overlay = vis_image.copy()
        
        # Draw confidence heatmap for each region
        for region in regions:
            if 'bbox' not in region or 'confidence' not in region:
                continue
                
            x, y, w, h = region['bbox']
            confidence = region.get('confidence', 0.0)
            
            # Color based on confidence
            if confidence > 0.8:
                color = (0, 255, 0)  # Green for high confidence
            elif confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)  # Red for low confidence
            
            # Fill the region with the confidence color
            alpha = 0.3  # Transparency factor
            cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
        
        # Blend the overlay with the original image
        result = cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0)
        
        # Add a legend
        cv2.putText(result, "Confidence:", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(result, (150, 10), (180, 30), (0, 255, 0), -1)
        cv2.putText(result, "High", (190, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(result, (250, 10), (280, 30), (0, 255, 255), -1)
        cv2.putText(result, "Medium", (290, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(result, (380, 10), (410, 30), (0, 0, 255), -1)
        cv2.putText(result, "Low", (420, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Save or display the result
        if output_path:
            cv2.imwrite(output_path, result)
            logger.info(f"Confidence heatmap saved to {output_path}")
        
        return result
    
    @staticmethod
    def generate_full_report(ocr_result: Dict[str, Any], 
                           original_image: np.ndarray,
                           output_dir: str):
        """
        Generate a full visual report for OCR results.
        
        Args:
            ocr_result: OCR result dictionary
            original_image: Original document image
            output_dir: Output directory for the report
        """
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Generating visual report in {output_dir}")
        
        # 1. Create region visualization
        if 'layout' in ocr_result and 'regions' in ocr_result['layout']:
            regions = ocr_result['layout']['regions']
            region_vis = OCRVisualizer.visualize_regions(original_image, regions)
            cv2.imwrite(os.path.join(output_dir, "regions.jpg"), region_vis)
        
        # 2. Create preprocessed image visualization
        if 'layout' in ocr_result and 'visualization' in ocr_result['layout']:
            preproc_vis = ocr_result['layout']['visualization']
            cv2.imwrite(os.path.join(output_dir, "preprocessed.jpg"), preproc_vis)
            
            # Create side-by-side comparison
            comparison = OCRVisualizer.create_side_by_side_comparison(
                original_image, preproc_vis, "Original vs. Preprocessed"
            )
            cv2.imwrite(os.path.join(output_dir, "comparison.jpg"), comparison)
        
        # 3. Create confidence heatmap
        if 'ocr' in ocr_result and 'text_regions' in ocr_result['ocr']:
            regions = ocr_result['ocr']['text_regions']
            heatmap = OCRVisualizer.plot_ocr_confidence_heatmap(
                original_image, regions, 
                os.path.join(output_dir, "confidence_heatmap.jpg")
            )
        
        # 4. Create text correction visualization
        if ('ocr' in ocr_result and 'raw_text' in ocr_result['ocr'] and
            'text' in ocr_result):
            original_text = ocr_result['ocr']['raw_text']
            corrected_text = ocr_result['text']
            
            correction_vis = OCRVisualizer.create_correction_visualization(
                original_text, corrected_text
            )
            cv2.imwrite(os.path.join(output_dir, "text_correction.jpg"), correction_vis)
        
        # 5. Create an HTML report
        html_content = OCRVisualizer._generate_html_report(ocr_result, output_dir)
        
        with open(os.path.join(output_dir, "report.html"), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Visual report generated in {output_dir}")
    
    @staticmethod
    def _generate_html_report(ocr_result: Dict[str, Any], output_dir: str) -> str:
        """
        Generate an HTML report for OCR results.
        
        Args:
            ocr_result: OCR result dictionary
            output_dir: Output directory for the report
            
        Returns:
            HTML content as a string
        """
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Renaissance OCR Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2 { color: #2c3e50; }
        .section { margin-bottom: 30px; }
        .image-container { margin: 20px 0; }
        .image-container img { max-width: 100%; border: 1px solid #ddd; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .metadata { display: flex; flex-wrap: wrap; }
        .metadata div { margin-right: 20px; margin-bottom: 10px; }
        .confidence-high { color: green; }
        .confidence-medium { color: orange; }
        .confidence-low { color: red; }
    </style>
</head>
<body>
    <h1>Renaissance OCR Report</h1>
"""
        
        # Add metadata
        if 'metadata' in ocr_result:
            metadata = ocr_result['metadata']
            html += """
    <div class="section">
        <h2>Document Metadata</h2>
        <div class="metadata">
"""
            for key, value in metadata.items():
                html += f'            <div><strong>{key}:</strong> {value}</div>\n'
            
            html += """
        </div>
    </div>
"""
        
        # Add images
        html += """
    <div class="section">
        <h2>Document Analysis</h2>
        <div class="image-container">
            <h3>Region Detection</h3>
            <img src="regions.jpg" alt="Detected Regions">
        </div>
        
        <div class="image-container">
            <h3>Comparison: Original vs. Preprocessed</h3>
            <img src="comparison.jpg" alt="Original vs. Preprocessed">
        </div>
        
        <div class="image-container">
            <h3>Confidence Heatmap</h3>
            <img src="confidence_heatmap.jpg" alt="Confidence Heatmap">
        </div>
    </div>
"""
        
        # Add OCR results
        if 'text' in ocr_result:
            html += """
    <div class="section">
        <h2>OCR Results</h2>
        <h3>Final Text</h3>
        <pre>"""
            html += ocr_result['text']
            html += """</pre>
    </div>
"""
        
        # Add correction details if available
        if 'spelling' in ocr_result and 'correction_summary' in ocr_result['spelling']:
            summary = ocr_result['spelling']['correction_summary']
            html += """
    <div class="section">
        <h2>Spelling Correction Summary</h2>
        <p><strong>Correction Rate:</strong> """
            html += f"{summary.get('correction_rate', 0) * 100:.1f}% ({summary.get('corrected_regions', 0)}/{summary.get('total_regions', 0)} regions)"
            html += """</p>
        
        <h3>Top Corrections</h3>
        <table>
            <tr>
                <th>Original</th>
                <th>Corrected</th>
                <th>Occurrences</th>
            </tr>
"""
            
            for correction in summary.get('top_corrections', []):
                html += f"""
            <tr>
                <td>{correction.get('original', '')}</td>
                <td>{correction.get('corrected', '')}</td>
                <td>{correction.get('count', 0)}</td>
            </tr>"""
                
            html += """
        </table>
    </div>
"""
        
        # Add footer
        html += """
    <div class="section">
        <h2>About</h2>
        <p>Generated by Renaissance OCR System</p>
    </div>
</body>
</html>
"""
        
        return html