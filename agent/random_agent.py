import random
from typing import Dict, List
from agent.base_agent import BaseAgent
from game.scorecard import Scorecard

class RandomAgent(BaseAgent):
    """
    A baseline agent that makes random valid moves.
    """
    def __init__(self):
        self.scorecard_helper = Scorecard()

    def choose_action(self, state: Dict) -> tuple:
        dice = state["dice"]
        rolls_left = state["rolls_left"]
        scorecard_state = state["scorecard"]
        
        # If we have rolls left, flip a coin to decide whether to roll again or score
        should_roll = False
        if rolls_left > 0:
            should_roll = random.random() < 0.8
            
        if should_roll:
            # Choose random dice to hold
            num_to_hold = random.randint(0, 5)
            indices = random.sample(range(5), num_to_hold)
            return "roll", indices
        else:
            # Choose a random OPEN category to score
            categories = [
                "ones", "twos", "threes", "fours", "fives", "sixes",
                "three_of_a_kind", "four_of_a_kind", "full_house",
                "small_straight", "large_straight", "yahtzee", "chance"
            ]
            # Filter for open categories
            open_cats = [c for c in categories if scorecard_state[c] is None]
            
            if not open_cats:
                # Should not happen if game logic is correct
                return "score", "chance" 
                
            chosen_cat = random.choice(open_cats)
            return "score", chosen_cat
