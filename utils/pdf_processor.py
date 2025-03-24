"""
PDF processing utilities for Renaissance OCR system.
"""

import os
import logging
import tempfile
from typing import List, Optional, Dict, Any, Tuple, Union
import PyPDF2
from PIL import Image
import numpy as np
import cv2
import concurrent.futures
import re
import json

logger = logging.getLogger(__name__)

class PDFProcessor:
    """
    Utility class for processing PDF documents with advanced features for Renaissance document OCR.
    """
    
    def __init__(self, dpi: int = 600, use_ocr_detection: bool = True, 
                detect_layout: bool = True, preprocess_images: bool = True,
                thread_count: int = 4):
        """
        Initialize the PDF processor with Renaissance document-specific capabilities.
        
        Args:
            dpi: Resolution for PDF rendering
            use_ocr_detection: Whether to attempt to detect if a PDF already has OCR text
            detect_layout: Whether to use advanced layout detection during extraction
            preprocess_images: Whether to apply Renaissance-specific preprocessing to extracted images
            thread_count: Number of threads to use for parallel processing
        """
        self.dpi = dpi
        self.use_ocr_detection = use_ocr_detection
        self.detect_layout = detect_layout
        self.preprocess_images = preprocess_images
        self.thread_count = thread_count
        logger.info(f"PDF processor initialized with {dpi} DPI and Renaissance document optimizations")
    
    def extract_images_from_pdf(self, pdf_path: str, output_dir: Optional[str] = None,
                              page_range: Optional[Tuple[int, int]] = None) -> List[str]:
        """
        Extract pages from a PDF as images with Renaissance document optimizations.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Optional directory to save extracted images
            page_range: Optional tuple (start, end) for processing specific pages
            
        Returns:
            List of paths to extracted images
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting images from PDF with Renaissance optimizations: {pdf_path}")
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        else:
            # Create temporary directory
            output_dir = tempfile.mkdtemp()
        
        try:
            # Check if the PDF has OCR text already
            has_text = False
            if self.use_ocr_detection:
                has_text = self._check_for_existing_ocr(pdf_path)
                if has_text:
                    logger.info("PDF appears to already have OCR text")
            
            # Import pdf2image for high-quality rendering
            from pdf2image import convert_from_path
            
            # Get page count for proper reporting
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
            
            # Handle page range if specified
            if page_range:
                start_page, end_page = page_range
                start_page = max(0, start_page)
                end_page = min(total_pages - 1, end_page)
                page_numbers = list(range(start_page, end_page + 1))
                logger.info(f"Processing page range {start_page+1} to {end_page+1} of {total_pages}")
            else:
                page_numbers = None  # Process all pages
            
            # Convert PDF to images with optimal settings for Renaissance documents
            logger.info(f"Converting PDF to images with Renaissance-optimized settings (DPI={self.dpi})")
            
            # Key parameters optimized for historical documents
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi, 
                output_folder=None,  # We'll handle saving manually for better control
                fmt="jpg",
                thread_count=self.thread_count,
                use_pdftocairo=True,  # Better quality for historical documents
                first_page=None if page_numbers is None else page_numbers[0] + 1,
                last_page=None if page_numbers is None else page_numbers[-1] + 1,
                grayscale=False,  # Keep color for now, preprocessing will handle this
                transparent=False,
                use_cropbox=False,  # Use media box for Renaissance documents
                output_file=None,
                poppler_path=None,
                size=None,  # Native size with DPI control
                paths_only=False
            )
            
            # Process and save images
            image_paths = []
            metadata = {'pdf_info': {'has_ocr_text': has_text, 'total_pages': total_pages}}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.thread_count) as executor:
                # Create processing tasks for all images
                futures = []
                
                for i, image in enumerate(images):
                    page_num = i if page_numbers is None else page_numbers[i]
                    output_path = os.path.join(output_dir, f"page_{page_num+1:03d}.jpg")
                    
                    # Submit processing task for this page
                    futures.append(
                        executor.submit(
                            self._process_and_save_page, 
                            image, 
                            output_path, 
                            page_num,
                            pdf_reader.pages[page_num] if has_text else None
                        )
                    )
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        image_paths.append(result['path'])
                        # Add page metadata
                        if 'metadata' in result:
                            metadata.setdefault('pages', {})[result['page_num']] = result['metadata']
            
            # Sort the paths by page number for consistency
            image_paths.sort(key=lambda x: int(re.search(r'page_(\d+)', x).group(1)))
            
            # Save metadata
            metadata_path = os.path.join(output_dir, "pdf_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Extracted {len(image_paths)} pages to {output_dir}")
            return image_paths
            
        except ImportError:
            logger.error("pdf2image not installed. Required for Renaissance OCR. Please install with: pip install pdf2image")
            logger.error("You also need poppler-utils installed on your system")
            raise ImportError("pdf2image is required for Renaissance OCR. See log for installation instructions.")
    
    def _process_and_save_page(self, image: Image.Image, output_path: str, 
                              page_num: int, pdf_page: Optional[Any] = None) -> Dict[str, Any]:
        """
        Process a single PDF page with Renaissance document optimizations and save it.
        
        Args:
            image: PIL Image of the page
            output_path: Path to save the processed image
            page_num: Page number (0-based)
            pdf_page: PyPDF2 Page object if text extraction is needed
            
        Returns:
            Dictionary with path and metadata information
        """
        try:
            # Apply Renaissance-specific preprocessing if configured
            if self.preprocess_images:
                # Convert to numpy for OpenCV processing
                np_image = np.array(image)
                
                # Apply Renaissance-specific image optimizations
                enhanced_image = self._enhance_renaissance_page(np_image)
                
                # Convert back to PIL for saving
                image = Image.fromarray(enhanced_image)
            
            # Save the image with high quality
            image.save(output_path, "JPEG", quality=95, optimize=True)
            
            # Extract page metadata
            metadata = {}
            
            # Extract text content if available
            if pdf_page is not None:
                try:
                    metadata['text'] = pdf_page.extract_text()
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num+1}: {str(e)}")
            
            # Add image dimensions
            width, height = image.size
            metadata['dimensions'] = {'width': width, 'height': height}
            
            # Detect page layout features if enabled (simplified version)
            if self.detect_layout:
                layout_info = self._detect_renaissance_layout(np.array(image))
                if layout_info:
                    metadata['layout'] = layout_info
            
            return {
                'path': output_path, 
                'page_num': page_num,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing page {page_num+1}: {str(e)}")
            # Save the original image as fallback
            try:
                image.save(output_path, "JPEG")
                return {'path': output_path, 'page_num': page_num, 'error': str(e)}
            except:
                logger.error(f"Could not save even original image for page {page_num+1}")
                return None
    
    def _enhance_renaissance_page(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Renaissance document-specific enhancements to a page image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Enhanced image optimized for Renaissance OCR
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Particularly effective for Renaissance documents with uneven illumination
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Apply denoising (adjust parameters based on document type)
            denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, 
                                              searchWindowSize=21, templateWindowSize=7)
            
            # Apply unsharp masking to improve text clarity
            gaussian = cv2.GaussianBlur(denoised, (0, 0), 3)
            unsharp = cv2.addWeighted(denoised, 1.5, gaussian, -0.5, 0)
            
            # Convert back to RGB if the input was RGB
            if len(image.shape) == 3:
                unsharp = cv2.cvtColor(unsharp, cv2.COLOR_GRAY2RGB)
            
            return unsharp
            
        except Exception as e:
            logger.warning(f"Error during Renaissance image enhancement: {str(e)}")
            # Return original image if enhancement fails
            return image
    
    def _detect_renaissance_layout(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect layout features specific to Renaissance documents.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary of detected layout features
        """
        try:
            # Convert to grayscale if not already
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply thresholding
            _, binary = cv2.threshold(gray, 0, 255, 
                                   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for text block regions
            text_regions = []
            image_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                # Skip very small regions
                if w < 50 or h < 20 or area < 500:
                    continue
                
                # Calculate aspect ratio and density
                aspect_ratio = w / h if h > 0 else 0
                density = area / (w * h) if w * h > 0 else 0
                
                # Classify the region
                if 0.1 < aspect_ratio < 10 and density > 0.1:
                    if aspect_ratio > 5 or aspect_ratio < 0.2:
                        # Likely to be decorative or header/footer
                        image_regions.append({
                            'bbox': [x, y, w, h],
                            'type': 'decoration'
                        })
                    else:
                        # Likely to be text block
                        text_regions.append({
                            'bbox': [x, y, w, h],
                            'type': 'text'
                        })
            
            # Detect page orientation/skew
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            skew_angle = 0.0
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    # Filter horizontal and vertical lines
                    if 0.1 < theta < 1.5 or 1.7 < theta < 3.1:
                        angle = np.degrees(theta - np.pi/2)
                        angles.append(angle)
                
                if angles:
                    skew_angle = np.median(angles)
            
            # Combine all layout information
            layout_info = {
                'text_regions': text_regions,
                'image_regions': image_regions,
                'skew_angle': skew_angle,
                'columns': 1 if len(text_regions) <= 1 else self._estimate_column_count(text_regions),
                'has_decoration': len(image_regions) > 0
            }
            
            return layout_info
            
        except Exception as e:
            logger.warning(f"Error during Renaissance layout detection: {str(e)}")
            return {}
    
    def _estimate_column_count(self, text_regions: List[Dict[str, Any]]) -> int:
        """
        Estimate the number of columns in a Renaissance document based on text regions.
        
        Args:
            text_regions: List of detected text regions
            
        Returns:
            Estimated number of columns
        """
        if not text_regions or len(text_regions) < 2:
            return 1
        
        # Extract x-coordinates of region centers
        x_centers = []
        for region in text_regions:
            x, _, w, _ = region['bbox']
            x_centers.append(x + w/2)
        
        # Cluster x-coordinates to find columns
        from sklearn.cluster import KMeans
        try:
            x_array = np.array(x_centers).reshape(-1, 1)
            
            # Try different numbers of clusters
            best_score = float('-inf')
            best_k = 1
            
            for k in range(1, min(4, len(x_centers))):  # Test up to 3 columns
                kmeans = KMeans(n_clusters=k, random_state=0).fit(x_array)
                score = kmeans.score(x_array)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            
            return best_k
        except:
            # Fallback if clustering fails
            # Heuristic: if regions have distinct x positions, likely multiple columns
            x_coords = sorted(x_centers)
            diffs = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
            if diffs and max(diffs) > np.mean(diffs) * 2:
                return 2
            return 1
    
    def _check_for_existing_ocr(self, pdf_path: str) -> bool:
        """
        Check if a PDF already has OCR text layer.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF appears to have OCR text, False otherwise
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check a few pages for text
                pages_to_check = min(5, len(pdf_reader.pages))
                text_found = 0
                
                for i in range(pages_to_check):
                    text = pdf_reader.pages[i].extract_text()
                    
                    # If we find significant text, increment counter
                    if text and len(text.strip()) > 100:
                        text_found += 1
                
                # If majority of checked pages have text, assume OCR exists
                return text_found > (pages_to_check / 2)
                
        except Exception as e:
            logger.warning(f"Error checking for OCR text: {str(e)}")
            return False
    
    def merge_results_to_pdf(self, image_paths: List[str], 
                           ocr_results: List[Dict[str, Any]], 
                           output_path: str,
                           optimize_for_renaissance: bool = True):
        """
        Merge OCR results into a searchable PDF with Renaissance document optimizations.
        
        Args:
            image_paths: List of paths to page images
            ocr_results: List of OCR results for each page
            output_path: Output path for the searchable PDF
            optimize_for_renaissance: Whether to apply Renaissance-specific PDF optimizations
        """
        logger.info(f"Creating searchable PDF with Renaissance optimizations: {output_path}")
        
        try:
            # Try to use PyMuPDF for advanced PDF features
            import fitz  # PyMuPDF
            
            # Create a new PDF
            doc = fitz.open()
            
            # Process each page
            for i, (image_path, ocr_result) in enumerate(zip(image_paths, ocr_results)):
                logger.info(f"Adding page {i+1}/{len(image_paths)}")
                
                # Create a page from the image
                img = fitz.open(image_path)
                pdfbytes = img.convert_to_pdf()
                imgpdf = fitz.open("pdf", pdfbytes)
                
                # Create a new page and show the image
                page = doc.new_page(width=imgpdf[0].rect.width, 
                                    height=imgpdf[0].rect.height)
                page.show_pdf_page(page.rect, imgpdf, 0)
                
                # Extract text and region information
                text = ocr_result.get('text', '')
                regions = []
                
                # Get text regions if available
                if 'text_regions' in ocr_result:
                    regions = ocr_result['text_regions']
                elif 'ocr' in ocr_result and 'text_regions' in ocr_result['ocr']:
                    regions = ocr_result['ocr']['text_regions']
                elif 'spelling' in ocr_result and 'corrected_regions' in ocr_result['spelling']:
                    regions = ocr_result['spelling']['corrected_regions']
                
                # Add text layer with positioning based on regions
                if regions:
                    # Add text for each region
                    for region in regions:
                        if 'text' in region and 'bbox' in region:
                            region_text = region['text']
                            x, y, w, h = region['bbox']
                            
                            # Create a text block for this region
                            rect = fitz.Rect(x, y, x+w, y+h)
                            page.insert_textbox(rect, region_text, 
                                              fontname="Times-Roman",
                                              fontsize=11,
                                              align=fitz.TEXT_ALIGN_LEFT)
                else:
                    # No regions available, add text as one block
                    if text:
                        # Insert text at appropriate position
                        page.insert_text((50, 50), text, fontname="Times-Roman", fontsize=11)
                
                # Apply Renaissance-specific PDF optimizations if requested
                if optimize_for_renaissance:
                    # Add metadata if available
                    if 'metadata' in ocr_result:
                        metadata = ocr_result['metadata']
                        if 'filename' in metadata:
                            page.set_metadata({"title": metadata['filename']})
                    
                    # Add language information based on language settings
                    if 'spelling' in ocr_result and 'correction_summary' in ocr_result['spelling']:
                        lang = "es" # Default to Spanish for Renaissance
                        page.set_metadata({"language": lang})
            
            # Set document metadata
            doc.set_metadata({
                "title": os.path.basename(output_path).split('.')[0],
                "subject": "Renaissance OCR Document",
                "keywords": "Renaissance, OCR, historical document",
                "creator": "Renaissance OCR System",
                "producer": "Renaissance OCR with PyMuPDF"
            })
            
            # Set PDF optimization options for Renaissance documents
            doc.save(output_path, 
                   garbage=4,       # Maximum garbage collection
                   deflate=True,    # Compress streams
                   clean=True,      # Remove unnecessary elements
                   linear=True)     # Optimize for web viewing
            
            doc.close()
            
            logger.info(f"Created searchable PDF with Renaissance optimizations: {output_path}")
            
        except ImportError:
            logger.warning("PyMuPDF not available. Falling back to alternative method.")
            self._create_searchable_pdf_fallback(image_paths, ocr_results, output_path)
    
    def _create_searchable_pdf_fallback(self, image_paths: List[str],
                                     ocr_results: List[Dict[str, Any]],
                                     output_path: str):
        """
        Fallback method to create searchable PDF if PyMuPDF is not available.
        
        Args:
            image_paths: List of paths to page images
            ocr_results: List of OCR results for each page
            output_path: Output path for the searchable PDF
        """
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            import io
            
            # Try to register a font suitable for Renaissance documents
            try:
                # Try to use a suitable font for Renaissance text
                font_path = os.path.join(os.path.dirname(__file__), "..", "resources", "fonts", "GentiumPlus-Regular.ttf")
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont("Gentium", font_path))
                    font_name = "Gentium"
                else:
                    font_name = "Times-Roman"
            except:
                font_name = "Times-Roman"
            
            # Create a new PDF
            output_pdf = PyPDF2.PdfWriter()
            
            for i, (image_path, ocr_result) in enumerate(zip(image_paths, ocr_results)):
                logger.info(f"Adding page {i+1}/{len(image_paths)}")
                
                # Extract text from OCR result
                if 'text' in ocr_result:
                    text = ocr_result['text']
                elif 'ocr' in ocr_result and 'raw_text' in ocr_result['ocr']:
                    text = ocr_result['ocr']['raw_text']
                else:
                    text = ""
                
                # Create a PDF with the image
                img_buffer = io.BytesIO()
                
                # Get image dimensions
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                # Create PDF page with custom size
                img_canvas = canvas.Canvas(
                    img_buffer, 
                    pagesize=(img_width, img_height)
                )
                
                # Add the image
                img_canvas.drawImage(
                    image_path, 
                    0, 0,
                    width=img_width,
                    height=img_height
                )
                
                # Add text layer (not visible but searchable)
                if text:
                    img_canvas.setFont(font_name, 0.2)  # Tiny invisible text
                    img_canvas.setFillColorRGB(1, 1, 1)  # White/invisible
                    
                    # Break text into lines to avoid overflow
                    lines = text.split('\n')
                    y_position = img_height - 5
                    for line in lines:
                        if line.strip():
                            img_canvas.drawString(5, y_position, line)
                            y_position -= 1
                
                img_canvas.save()
                
                # Create a PDF page from the buffer
                img_buffer.seek(0)
                img_pdf = PyPDF2.PdfReader(img_buffer)
                
                # Add the page to the output PDF
                output_pdf.add_page(img_pdf.pages[0])
                
                # Add page metadata
                if hasattr(output_pdf, 'add_metadata_to_page'):
                    try:
                        metadata = {
                            "/Type": "/Page",
                            "/OCRText": text  # Custom tag for OCR text
                        }
                        output_pdf.add_metadata_to_page(i, metadata)
                    except:
                        pass
            
            # Add document metadata
            output_pdf.add_metadata({
                "/Title": os.path.basename(output_path).split('.')[0],
                "/Subject": "Renaissance OCR Document",
                "/Keywords": "Renaissance, OCR, historical document",
                "/Creator": "Renaissance OCR System",
                "/Producer": "Renaissance OCR PDF Generator"
            })
            
            # Save the PDF
            with open(output_path, 'wb') as f:
                output_pdf.write(f)
            
            logger.info(f"Created basic searchable PDF: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating fallback searchable PDF: {str(e)}")
            raise