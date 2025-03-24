"""
Historical Spelling Agent that corrects OCR outputs based on historical linguistic rules.
"""

import re
import logging
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
import difflib
from collections import defaultdict

from agents.base_agent import BaseAgent
from configs.config import SpellingAgentConfig

logger = logging.getLogger(__name__)

class HistoricalSpellingAgent(BaseAgent):
    """
    Agent for correcting OCR text based on historical spelling patterns and linguistics.
    """
    
    def __init__(self, config: SpellingAgentConfig):
        """
        Initialize the Historical Spelling agent.
        
        Args:
            config: Spelling agent configuration
        """
        super().__init__("HistoricalSpelling", config.__dict__)
        
        # Initialize dictionaries and rules based on language and period
        self.dictionary = self._load_dictionary()
        self.rules = self._load_rules()
        
        self.log_info(f"Historical Spelling agent initialized for {config.language} ({config.historical_period})")
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Correct OCR text using historical linguistic rules.
        
        Args:
            inputs: Dictionary containing:
                - 'text_regions': List of region dictionaries with 'text' and 'bbox' keys
                - 'full_text': Combined text from all regions
                
        Returns:
            Dictionary with corrected text
        """
        # Validate inputs
        required_keys = ['text_regions']
        if not self.validate_inputs(inputs, required_keys):
            return {'error': 'Invalid inputs', 'corrected_regions': []}
        
        text_regions = inputs['text_regions']
        self.log_info(f"Correcting text from {len(text_regions)} regions")
        
        # Apply corrections to each region
        corrected_regions = []
        
        for region in text_regions:
            region_id = region.get('id', 0)
            original_text = region['text']
            
            self.log_debug(f"Processing region {region_id}")
            
            # Apply the correction
            corrected_text = self.correct_spelling(original_text)
            
            # Calculate confidence adjustment based on correction amount
            confidence_adjustment = self._calculate_confidence_adjustment(original_text, corrected_text)
            original_confidence = region.get('confidence', 0.8)
            
            # Create corrected region entry
            corrected_region = region.copy()
            corrected_region['original_text'] = original_text
            corrected_region['text'] = corrected_text
            corrected_region['confidence'] = max(0.0, min(1.0, original_confidence + confidence_adjustment))
            corrected_region['correction_applied'] = (original_text != corrected_text)
            
            corrected_regions.append(corrected_region)
        
        # Combine corrected regions into full text
        full_corrected_text = self._combine_region_texts(corrected_regions)
        
        self.log_info("Correction completed")
        
        result = {
            'corrected_regions': corrected_regions,
            'full_text': full_corrected_text,
            'correction_summary': self._generate_correction_summary(corrected_regions)
        }
        
        return result
    
    def correct_spelling(self, text: str) -> str:
        """
        Apply historical spelling corrections to the text.
        
        Args:
            text: Original OCR text
            
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        corrected_text = text
        
        # Apply word-level corrections
        words = re.findall(r'\b\w+\b', corrected_text)
        for word in words:
            # Skip very short words
            if len(word) <= 1:
                continue
                
            corrected_word = self._correct_word(word)
            if corrected_word != word:
                # Replace the word with proper word boundary preservation
                corrected_text = re.sub(r'\b' + re.escape(word) + r'\b', corrected_word, corrected_text)
        
        # Apply pattern-based corrections
        for pattern, replacement in self.rules.get('patterns', {}).items():
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        # Fix common OCR errors specific to the historical period
        for error, fix in self.rules.get('ocr_errors', {}).items():
            corrected_text = corrected_text.replace(error, fix)
            
        # Apply period-specific punctuation rules
        if self.rules.get('punctuation'):
            for punct_rule in self.rules.get('punctuation', []):
                pattern = punct_rule.get('pattern')
                replacement = punct_rule.get('replacement')
                if pattern and replacement:
                    corrected_text = re.sub(pattern, replacement, corrected_text)
        
        return corrected_text
    
    def _correct_word(self, word: str) -> str:
        """
        Correct a single word using dictionary lookup and rule-based methods.
        
        Args:
            word: Original word
            
        Returns:
            Corrected word
        """
        # Skip correction if already in dictionary
        if word.lower() in self.dictionary:
            return word
        
        # Try specific period-appropriate replacements
        for old, new in self.rules.get('word_replacements', {}).items():
            if word.lower() == old.lower():
                return new if word[0].isupper() else new.lower()
        
        # Skip if using context correction but word doesn't meet threshold
        if self.config["use_contextual_correction"]:
            # Find closest match in dictionary
            candidates = difflib.get_close_matches(
                word.lower(), 
                self.dictionary, 
                n=1, 
                cutoff=0.8
            )
            
            if candidates:
                corrected = candidates[0]
                # Preserve capitalization
                if word[0].isupper():
                    corrected = corrected.capitalize()
                return corrected
        
        # Apply spelling rules specific to the historical period
        for rule_pattern, rule_replacement in self.rules.get('spelling_rules', {}).items():
            if re.search(rule_pattern, word):
                corrected = re.sub(rule_pattern, rule_replacement, word)
                return corrected
        
        return word
        
    def _calculate_confidence_adjustment(self, original: str, corrected: str) -> float:
        """
        Calculate confidence adjustment based on amount of correction.
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Confidence adjustment factor
        """
        if original == corrected:
            return 0.0
        
        # Calculate the similarity ratio
        similarity = difflib.SequenceMatcher(None, original, corrected).ratio()
        
        # If very similar (few corrections), slightly increase confidence
        # If very different (many corrections), decrease confidence
        if similarity > 0.95:
            return 0.05  # Slight improvement
        elif similarity > 0.8:
            return 0.0   # Neutral
        else:
            return -0.1  # Significant changes, reduce confidence
    
    def _combine_region_texts(self, regions: List[Dict[str, Any]]) -> str:
        """
        Combine corrected texts from multiple regions.
        
        Args:
            regions: List of corrected region dictionaries
            
        Returns:
            Combined corrected text
        """
        # Sort regions by vertical position (top to bottom)
        sorted_regions = sorted(regions, key=lambda r: r['bbox'][1])
        
        # Combine texts
        full_text = "\n\n".join(region['text'] for region in sorted_regions)
        
        return full_text
    
    def _generate_correction_summary(self, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a summary of corrections made.
        
        Args:
            regions: List of corrected region dictionaries
            
        Returns:
            Correction summary
        """
        total_regions = len(regions)
        corrected_regions = sum(1 for r in regions if r.get('correction_applied', False))
        
        # Count specific types of corrections
        word_counts = defaultdict(int)
        
        for region in regions:
            if not region.get('correction_applied', False):
                continue
                
            original = region.get('original_text', '')
            corrected = region.get('text', '')
            
            # Compare words
            original_words = re.findall(r'\b\w+\b', original)
            corrected_words = re.findall(r'\b\w+\b', corrected)
            
            for orig_word, corr_word in zip(original_words, corrected_words):
                if orig_word != corr_word:
                    word_counts[(orig_word, corr_word)] += 1
        
        # Get top corrections
        top_corrections = sorted(
            [(orig, corr, count) for (orig, corr), count in word_counts.items()],
            key=lambda x: x[2],
            reverse=True
        )[:10]  # Top 10
        
        summary = {
            'total_regions': total_regions,
            'corrected_regions': corrected_regions,
            'correction_rate': corrected_regions / total_regions if total_regions > 0 else 0,
            'top_corrections': [
                {'original': orig, 'corrected': corr, 'count': count}
                for orig, corr, count in top_corrections
            ]
        }
        
        return summary
    
    def _load_dictionary(self) -> List[str]:
        """
        Load the appropriate historical dictionary based on language and period.
        
        Returns:
            List of valid words for the period
        """
        # Check for custom dictionary
        if self.config["custom_dictionary_path"] and os.path.exists(self.config["custom_dictionary_path"]):
            with open(self.config["custom_dictionary_path"], 'r', encoding='utf-8') as f:
                words = [line.strip().lower() for line in f if line.strip()]
            self.log_info(f"Loaded custom dictionary with {len(words)} words")
            return words
        
        # Fallback to built-in dictionaries
        language = self.config["language"]
        period = self.config["historical_period"]
        
        # This is a placeholder - in a real implementation, we'd have proper dictionaries
        # Here we're creating a minimal Spanish Renaissance dictionary for demonstration
        if language == "spanish" and period == "renaissance":
            # Some common Spanish Renaissance words
            words = [
                "rey", "reyna", "señor", "señora", "vuestra", "merced", "dios",
                "santa", "sancta", "christo", "jesu", "joseph", "maria", "virgen",
                "iglesia", "templo", "fe", "esperança", "caridad", "amor", "cielo",
                "tierra", "mar", "fuego", "agua", "ayre", "cavallero", "caballero",
                "hidalgo", "nobleza", "honra", "honor", "virtud", "valor", "espada",
                "escudo", "lança", "guerra", "paz", "hermano", "hermana", "padre",
                "madre", "hijo", "hija", "niño", "niña", "moço", "moça", "hombre",
                "muger", "dulce", "dulcissimo", "infinitamente", "amable", "doctor",
                "dignasteis", "llamaros", "tambien", "asistir", "doctores", "consagra",
                "humilde", "pequeña", "instruccion", "juventud", "aprendio", "precisa",
                "explicacion", "deben", "estudiar", "titulo", "confiáis", "educacion",
                "compañia", "exemplar", "todas", "virtudes", "abreviado", "seguro",
                "diseño", "edad", "religion", "devota", "assistecia", "piedad",
                "obediencia", "rendida", "modestia", "deseo", "saber", "mayores",
                "gustando", "preguntar", "definir", "resolver", "sabiduria", "soberana",
                "dignacion", "natural", "ignorancia", "indispensable", "necessidad"
            ]
            self.log_info(f"Loaded built-in Spanish Renaissance dictionary with {len(words)} words")
            return words
        
        # Default empty dictionary
        self.log_warning(f"No dictionary found for {language} ({period}), using empty dictionary")
        return []
    
    def _load_rules(self) -> Dict[str, Any]:
        """
        Load the appropriate historical spelling rules based on language and period.
        
        Returns:
            Dictionary of rules
        """
        language = self.config["language"]
        period = self.config["historical_period"]
        
        # This is a placeholder - in a real implementation, we'd have proper rule sets
        # Here we're creating Spanish Renaissance rules for demonstration
        if language == "spanish" and period == "renaissance":
            rules = {
                # Common spelling variations in Renaissance Spanish
                'word_replacements': {
                    "mui": "muy",
                    "reyno": "reino",
                    "assí": "así",
                    "dixo": "dijo",
                    "officio": "oficio",
                    "excellent": "excelente",
                    "sciencia": "ciencia",
                    "ayre": "aire",
                    "sant": "san",
                    "fecho": "hecho",
                    "dubda": "duda",
                    "cibdad": "ciudad",
                    "escripto": "escrito"
                },
                
                # Regex patterns for systematic spelling changes
                'patterns': {
                    r'ph': 'f',
                    r'th': 't',
                    r'ſ': 's',  # Long s
                    r'ct\b': 'to',
                    r'ç': 'z',
                    r'ss': 's',
                    r'ff': 'f',
                    r'tt': 't',
                    r'pp': 'p',
                    r'ſſ': 'ss'
                },
                
                # Rules for specific spelling adjustments
                'spelling_rules': {
                    r'nn': 'ñ',
                    r'ii([aeou])': r'j\1',
                    r'x([ei])': r'j\1',
                    r'ſ': 's',
                    r'u([aeiou])': r'v\1',
                    r'v([lrndgbcm])': r'u\1'
                },
                
                # Common OCR errors in Renaissance texts
                'ocr_errors': {
                    "cl": "d",
                    "rn": "m",
                    "li": "h",
                    "i1": "il",
                    "l1": "li",
                    "0": "o",
                    "1": "l",
                    "iii": "m",
                    "vv": "w",
                    "I": "J"  # In some contexts
                },
                
                # Punctuation rules
                'punctuation': [
                    {'pattern': r'(?<=[a-zA-Z]),(?=[a-zA-Z])', 'replacement': ', '},  # Add space after comma
                    {'pattern': r'(?<=[a-zA-Z])\.(?=[a-zA-Z])', 'replacement': '. '}  # Add space after period
                ]
            }
            
            self.log_info(f"Loaded built-in Spanish Renaissance spelling rules")
            return rules
        
        # Default empty rules
        self.log_warning(f"No rules found for {language} ({period}), using empty rules")
        return {}