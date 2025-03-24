"""
Evaluation metrics for Renaissance OCR system.
"""

import Levenshtein
import re
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from configs.config import EvaluationConfig

logger = logging.getLogger(__name__)

class OCREvaluator:
    """
    Evaluator for OCR results on Renaissance documents.
    """
    
    def __init__(self, config: EvaluationConfig):
        """
        Initialize the OCR evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        logger.info("OCR evaluator initialized")
    
    def evaluate(self, ocr_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Evaluate OCR results against ground truth.
        
        Args:
            ocr_text: OCR-generated text
            ground_truth: Ground truth text
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating OCR results")
        
        results = {}
        
        # Character Error Rate (CER)
        if self.config.calculate_cer:
            results["cer"] = self.calculate_cer(ocr_text, ground_truth)
            results["cer_percentage"] = results["cer"] * 100.0
            logger.info(f"CER: {results['cer_percentage']:.2f}%")
        
        # Word Error Rate (WER)
        if self.config.calculate_wer:
            results["wer"] = self.calculate_wer(ocr_text, ground_truth)
            results["wer_percentage"] = results["wer"] * 100.0
            logger.info(f"WER: {results['wer_percentage']:.2f}%")
        
        # Historical accuracy metrics
        if self.config.calculate_historical_accuracy:
            historical_metrics = self.calculate_historical_accuracy(ocr_text, ground_truth)
            results.update(historical_metrics)
            logger.info(f"Historical accuracy: {results.get('historical_accuracy_score', 0):.2f}")
        
        # Layout accuracy metrics
        if self.config.calculate_layout_f1 and "regions" in ocr_text:
            # This is a placeholder - proper implementation would need region information
            results["layout_f1"] = 0.0
            logger.info("Layout F1 calculation requires region information")
        
        # Add summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def calculate_cer(self, ocr_text: str, ground_truth: str) -> float:
        """
        Calculate Character Error Rate.
        
        Args:
            ocr_text: OCR-generated text
            ground_truth: Ground truth text
            
        Returns:
            Character Error Rate (0.0 to 1.0)
        """
        # Normalize texts: remove extra whitespace
        ocr_norm = self._normalize_text(ocr_text)
        gt_norm = self._normalize_text(ground_truth)
        
        # Calculate Levenshtein distance
        edit_distance = Levenshtein.distance(ocr_norm, gt_norm)
        
        # Character Error Rate = edit distance / length of ground truth
        cer = edit_distance / max(len(gt_norm), 1)
        
        return cer
    
    def calculate_wer(self, ocr_text: str, ground_truth: str) -> float:
        """
        Calculate Word Error Rate.
        
        Args:
            ocr_text: OCR-generated text
            ground_truth: Ground truth text
            
        Returns:
            Word Error Rate (0.0 to 1.0)
        """
        # Normalize and tokenize texts
        ocr_words = self._tokenize_text(ocr_text)
        gt_words = self._tokenize_text(ground_truth)
        
        # Calculate Levenshtein distance at word level
        edit_distance = Levenshtein.distance(ocr_words, gt_words)
        
        # Word Error Rate = edit distance / number of words in ground truth
        wer = edit_distance / max(len(gt_words), 1)
        
        return wer
    
    def calculate_historical_accuracy(self, ocr_text: str, ground_truth: str) -> Dict[str, Any]:
        """
        Calculate historical accuracy metrics.
        
        Args:
            ocr_text: OCR-generated text
            ground_truth: Ground truth text
            
        Returns:
            Dictionary of historical accuracy metrics
        """
        # Normalize texts
        ocr_norm = self._normalize_text(ocr_text)
        gt_norm = self._normalize_text(ground_truth)
        
        # Extract metrics specific to historical documents
        
        # 1. Archaic character recognition
        archaic_chars = ['ſ', 'æ', 'œ', 'ƒ', 'ß', 'þ', 'ð', 'ȝ', 'ꝛ', 'ꝑ', 'ꝓ', 'ꝗ']
        archaic_correct = 0
        archaic_total = 0
        
        for char in archaic_chars:
            gt_count = gt_norm.count(char)
            if gt_count > 0:
                ocr_count = ocr_norm.count(char)
                archaic_correct += min(ocr_count, gt_count)
                archaic_total += gt_count
        
        archaic_accuracy = archaic_correct / max(archaic_total, 1)
        
        # 2. Abbreviation accuracy
        abbrev_pattern = r'\b[A-Za-z]\.'
        gt_abbrevs = re.findall(abbrev_pattern, gt_norm)
        ocr_abbrevs = re.findall(abbrev_pattern, ocr_norm)
        
        abbrev_correct = sum(1 for abbr in gt_abbrevs if abbr in ocr_abbrevs)
        abbrev_accuracy = abbrev_correct / max(len(gt_abbrevs), 1)
        
        # 3. Period-specific spelling accuracy
        # This would require a dictionary of period-specific spellings
        # Simplified implementation for demonstration
        period_spelling_accuracy = max(0.0, 1.0 - self.calculate_cer(ocr_norm, gt_norm) * 1.5)
        
        # Combine into an overall historical accuracy score
        # Weighted average of the metrics
        weights = {
            'archaic': 0.3,
            'abbrev': 0.3,
            'spelling': 0.4
        }
        
        historical_accuracy = (
            weights['archaic'] * archaic_accuracy +
            weights['abbrev'] * abbrev_accuracy +
            weights['spelling'] * period_spelling_accuracy
        )
        
        return {
            'historical_accuracy_score': historical_accuracy,
            'archaic_character_accuracy': archaic_accuracy,
            'abbreviation_accuracy': abbrev_accuracy,
            'period_spelling_accuracy': period_spelling_accuracy
        }
    
    def calculate_layout_accuracy(self, ocr_regions: List[Dict[str, Any]], 
                                 gt_regions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate layout accuracy metrics.
        
        Args:
            ocr_regions: OCR-detected regions
            gt_regions: Ground truth regions
            
        Returns:
            Dictionary of layout accuracy metrics
        """
        # This is a simplified implementation
        # Real implementation would involve more sophisticated region matching
        
        # Calculate IoU for each OCR region with each GT region
        matches = []
        
        for ocr_region in ocr_regions:
            ocr_bbox = ocr_region.get('bbox', (0, 0, 0, 0))
            
            best_iou = 0.0
            best_match = None
            
            for gt_region in gt_regions:
                gt_bbox = gt_region.get('bbox', (0, 0, 0, 0))
                
                iou = self._calculate_iou(ocr_bbox, gt_bbox)
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = gt_region
            
            if best_iou > 0.5:  # Threshold for considering a match
                matches.append((ocr_region, best_match, best_iou))
        
        # Calculate precision, recall, F1
        precision = len(matches) / max(len(ocr_regions), 1)
        recall = len(matches) / max(len(gt_regions), 1)
        
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        return {
            'layout_precision': precision,
            'layout_recall': recall,
            'layout_f1': f1
        }
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for evaluation.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', text).strip()
        
        # Normalize line endings
        normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
        
        return normalized
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        if not text:
            return []
        
        # Normalize first
        normalized = self._normalize_text(text)
        
        # Split into words
        words = re.findall(r'\b\w+\b', normalized.lower())
        
        return words
    
    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], 
                      bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x, y, w, h)
            bbox2: Second bounding box (x, y, w, h)
            
        Returns:
            IoU value (0.0 to 1.0)
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate coordinates of the intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.
        
        Args:
            results: Evaluation results
            
        Returns:
            Summary string
        """
        summary = []
        
        summary.append("OCR Evaluation Summary")
        summary.append("=====================")
        
        if "cer" in results:
            summary.append(f"Character Error Rate (CER): {results['cer_percentage']:.2f}%")
        
        if "wer" in results:
            summary.append(f"Word Error Rate (WER): {results['wer_percentage']:.2f}%")
        
        if "historical_accuracy_score" in results:
            summary.append(f"Historical Accuracy: {results['historical_accuracy_score']:.2f}")
            summary.append(f"  - Archaic Character Accuracy: {results.get('archaic_character_accuracy', 0):.2f}")
            summary.append(f"  - Abbreviation Accuracy: {results.get('abbreviation_accuracy', 0):.2f}")
            summary.append(f"  - Period Spelling Accuracy: {results.get('period_spelling_accuracy', 0):.2f}")
        
        if "layout_f1" in results:
            summary.append(f"Layout F1 Score: {results['layout_f1']:.2f}")
            summary.append(f"  - Layout Precision: {results.get('layout_precision', 0):.2f}")
            summary.append(f"  - Layout Recall: {results.get('layout_recall', 0):.2f}")
        
        # Overall assessment
        if "cer" in results and "wer" in results:
            avg_error = (results["cer"] + results["wer"]) / 2
            
            if avg_error < 0.05:
                assessment = "Excellent"
            elif avg_error < 0.1:
                assessment = "Very Good"
            elif avg_error < 0.15:
                assessment = "Good"
            elif avg_error < 0.25:
                assessment = "Fair"
            else:
                assessment = "Poor"
            
            summary.append(f"Overall Quality: {assessment} ({(1-avg_error)*100:.1f}%)")
        
        return "\n".join(summary)