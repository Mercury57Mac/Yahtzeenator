from abc import ABC, abstractmethod
from typing import List, Dict

class BaseAgent(ABC):
    """
    Abstract base class for a Yahtzee agent.
    """
    
    @abstractmethod
    def choose_action(self, state: Dict) -> tuple:
        """
        Decides on an action based on the current game state.
        
        Args:
            state: A dictionary containing:
                - 'dice': List[int] current face values
                - 'rolls_left': int (0-3)
                - 'scorecard': Dict[str, int] current scores (or None)
                
        Returns:
            (action_type, action_payload)
            - action_type: "roll" or "score"
            - action_payload: list of indices to hold OR category string
        """
        pass
