"""
Base agent class for the OCR multi-agent system.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the OCR system.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            config: Agent configuration
        """
        self.name = name
        self.config = config
        logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process inputs and produce outputs.
        
        Args:
            inputs: Input data for the agent
            
        Returns:
            Processed outputs
        """
        pass
    
    def log_info(self, message: str):
        """
        Log information with the agent's name.
        
        Args:
            message: Message to log
        """
        logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """
        Log error with the agent's name.
        
        Args:
            message: Error message to log
        """
        logger.error(f"[{self.name}] {message}")
    
    def log_debug(self, message: str):
        """
        Log debug information with the agent's name.
        
        Args:
            message: Debug message to log
        """
        logger.debug(f"[{self.name}] {message}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the agent.
        
        Returns:
            Status information
        """
        return {
            "name": self.name,
            "type": self.__class__.__name__,
            "ready": True
        }

    def validate_inputs(self, inputs: Dict[str, Any], required_keys: List[str]) -> bool:
        """
        Validate that required keys are present in the inputs.
        
        Args:
            inputs: Input data to validate
            required_keys: List of required keys
            
        Returns:
            True if all required keys are present, False otherwise
        """
        missing_keys = [key for key in required_keys if key not in inputs]
        
        if missing_keys:
            self.log_error(f"Missing required inputs: {missing_keys}")
            return False
        
        return True