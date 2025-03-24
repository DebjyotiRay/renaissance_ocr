"""
Agent Orchestrator for coordinating the OCR multi-agent system.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union, Tuple

from agents.base_agent import BaseAgent
from agents.ocr_agent import OCRAgent
from agents.layout_validator_agent import LayoutValidatorAgent
from agents.historical_spelling_agent import HistoricalSpellingAgent
from configs.config import AgentOrchestratorConfig

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    """
    Orchestrator that coordinates the multi-agent workflow for Renaissance document OCR.
    """
    
    def __init__(
        self,
        ocr_agent: OCRAgent,
        layout_validator: LayoutValidatorAgent,
        spelling_agent: HistoricalSpellingAgent,
        config: AgentOrchestratorConfig
    ):
        """
        Initialize the agent orchestrator.
        
        Args:
            ocr_agent: OCR agent for text extraction
            layout_validator: Layout validator agent
            spelling_agent: Historical spelling agent
            config: Orchestrator configuration
        """
        self.agents = {
            "ocr": ocr_agent,
            "layout": layout_validator,
            "spelling": spelling_agent
        }
        
        self.config = config
        self.max_turns = config.max_turns
        self.debug_mode = config.debug_mode
        
        logger.info("Agent orchestrator initialized")
    
    def process_document(self, image, preprocessing_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a document through the multi-agent workflow.
        
        Args:
            image: Input document image
            preprocessing_result: Result from the image preprocessor
            
        Returns:
            Final processing result
        """
        logger.info("Starting multi-agent document processing")
        
        # Initialize workflow state
        state = {
            "image": image,
            "regions": preprocessing_result.get("regions", []),
            "binary": preprocessing_result.get("binary", None),
            "current_turn": 0,
            "start_time": time.time(),
            "outputs": {},
            "status": "initialized"
        }
        
        # Process through the agent workflow
        for turn in range(self.max_turns):
            state["current_turn"] = turn
            
            # Update status
            logger.info(f"Turn {turn+1}/{self.max_turns}")
            state["status"] = f"processing_turn_{turn+1}"
            
            # Determine next agent
            next_agent = self._determine_next_agent(state)
            
            if next_agent is None:
                logger.info("Workflow completed successfully")
                state["status"] = "completed"
                break
            
            # Process with the selected agent
            logger.info(f"Processing with agent: {next_agent}")
            agent_input = self._prepare_agent_input(next_agent, state)
            
            agent = self.agents[next_agent]
            agent_output = agent.process(agent_input)
            
            # Store agent output
            state["outputs"][next_agent] = agent_output
            
            # Update state based on agent output
            self._update_state(next_agent, agent_output, state)
            
            # Check for completion
            if self._check_workflow_completion(state):
                logger.info("Workflow completion criteria met")
                state["status"] = "completed"
                break
        
        # Check if max turns reached without completion
        if state["status"] != "completed":
            logger.warning("Maximum turns reached without completion")
            state["status"] = "max_turns_reached"
        
        # Prepare final result
        result = self._prepare_final_result(state)
        
        # Log processing time
        processing_time = time.time() - state["start_time"]
        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        
        return result
    
    def _determine_next_agent(self, state: Dict[str, Any]) -> Optional[str]:
        """
        Determine the next agent to run based on the current state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Name of the next agent, or None if workflow is complete
        """
        current_turn = state["current_turn"]
        outputs = state["outputs"]
        
        # First turn: always start with layout validation
        if current_turn == 0:
            return "layout"
        
        # Second turn: OCR agent processes validated regions
        if current_turn == 1:
            return "ocr"
        
        # Third turn: Historical spelling corrects OCR output
        if current_turn == 2:
            return "spelling"
        
        # Additional turns can implement more complex decision logic
        # For example, we could re-run OCR agent on regions with low confidence
        if current_turn == 3 and "spelling" in outputs:
            corrected_regions = outputs["spelling"].get("corrected_regions", [])
            
            # Check if any regions have low confidence
            low_confidence_regions = [
                r for r in corrected_regions 
                if r.get("confidence", 1.0) < 0.7
            ]
            
            if low_confidence_regions:
                return "ocr"  # Re-run OCR on problematic regions
        
        # No more agents needed
        return None
    
    def _prepare_agent_input(self, agent_name: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare input for a specific agent.
        
        Args:
            agent_name: Name of the agent
            state: Current workflow state
            
        Returns:
            Input dictionary for the agent
        """
        outputs = state["outputs"]
        
        if agent_name == "layout":
            # Layout validator needs the image and initial regions
            return {
                "image": state["image"],
                "regions": state["regions"]
            }
        
        elif agent_name == "ocr":
            # OCR agent needs the image and validated regions
            if "layout" in outputs:
                return {
                    "image": state["image"],
                    "regions": outputs["layout"].get("validated_regions", state["regions"])
                }
            else:
                return {
                    "image": state["image"],
                    "regions": state["regions"]
                }
        
        elif agent_name == "spelling":
            # Spelling agent needs the text regions from OCR
            if "ocr" in outputs:
                return {
                    "text_regions": outputs["ocr"].get("text_regions", []),
                    "full_text": outputs["ocr"].get("full_text", "")
                }
            else:
                return {
                    "text_regions": [],
                    "full_text": ""
                }
        
        # Default empty input
        return {}
    
    def _update_state(self, agent_name: str, agent_output: Dict[str, Any], state: Dict[str, Any]):
        """
        Update the workflow state based on agent output.
        
        Args:
            agent_name: Name of the agent
            agent_output: Output from the agent
            state: Current workflow state to update
        """
        if agent_name == "layout":
            # Update regions with validated regions
            if "validated_regions" in agent_output:
                state["regions"] = agent_output["validated_regions"]
        
        elif agent_name == "ocr":
            # Store OCR results
            if "text_regions" in agent_output:
                state["text_regions"] = agent_output["text_regions"]
            if "full_text" in agent_output:
                state["full_text"] = agent_output["full_text"]
        
        elif agent_name == "spelling":
            # Store spelling correction results
            if "corrected_regions" in agent_output:
                state["corrected_regions"] = agent_output["corrected_regions"]
            if "full_text" in agent_output:
                state["corrected_text"] = agent_output["full_text"]
    
    def _check_workflow_completion(self, state: Dict[str, Any]) -> bool:
        """
        Check if the workflow has completed successfully.
        
        Args:
            state: Current workflow state
            
        Returns:
            True if workflow is complete, False otherwise
        """
        # Basic completion: we've processed through layout, OCR, and spelling
        outputs = state["outputs"]
        
        required_outputs = ["layout", "ocr", "spelling"]
        has_required = all(output in outputs for output in required_outputs)
        
        if has_required and "corrected_text" in state:
            return True
        
        return False
    
    def _prepare_final_result(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the final result from the workflow state.
        
        Args:
            state: Current workflow state
            
        Returns:
            Final processing result
        """
        outputs = state["outputs"]
        
        # Prepare final OCR result
        result = {
            "status": state["status"],
            "processing_time": time.time() - state["start_time"]
        }
        
        # Include layout information
        if "layout" in outputs:
            layout_output = outputs["layout"]
            result["layout"] = {
                "regions": layout_output.get("validated_regions", []),
                "visualization": layout_output.get("visualization", None)
            }
        
        # Include OCR information
        if "ocr" in outputs:
            ocr_output = outputs["ocr"]
            result["ocr"] = {
                "text_regions": ocr_output.get("text_regions", []),
                "raw_text": ocr_output.get("full_text", "")
            }
        
        # Include spelling correction information
        if "spelling" in outputs:
            spelling_output = outputs["spelling"]
            result["spelling"] = {
                "corrected_regions": spelling_output.get("corrected_regions", []),
                "correction_summary": spelling_output.get("correction_summary", {})
            }
        
        # Add final corrected text
        result["text"] = state.get("corrected_text", state.get("full_text", ""))
        
        # Include debug information if enabled
        if self.debug_mode:
            result["debug"] = {
                "state": {k: v for k, v in state.items() if k != "image"},
                "agent_outputs": outputs
            }
        
        return result