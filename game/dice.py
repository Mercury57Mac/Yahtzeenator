import random
from typing import List, Optional

class Dice:
    """
    Represents the 5 dice in a Yahtzee game.
    Handles rolling, holding, and state management.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the dice.
        Args:
            seed: Optional random seed for reproducibility.
        """
        self.num_dice = 5
        self.faces = [0] * self.num_dice  # 0 indicates not rolled yet
        self.held = [False] * self.num_dice
        self.rng = random.Random(seed)

    def roll(self) -> List[int]:
        """
        Rolls all dice that are not currently held.
        Returns:
            The current state of the dice faces.
        """
        for i in range(self.num_dice):
            if not self.held[i]:
                self.faces[i] = self.rng.randint(1, 6)

        return self.faces

    def hold(self, indices: List[int]) -> None:
        """
        Updates the held state of the dice.
        Args:
            indices: List of indices (0-4) of dice to hold.
            Any dice NOT in this list will be un-held.
        """
        self.held = [False] * self.num_dice
        for i in indices:
            self.held[i] = True
    
    def reset(self) -> None:
        """
        Resets the dice for a new turn (all unheld, faces 0).
        """
        self.faces = [0] * self.num_dice
        self.held = [False] * self.num_dice

    def get_state(self) -> List[int]:
        """
        Returns the current face values of the dice.
        """
        return self.faces
