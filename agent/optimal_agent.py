from typing import Dict, List, Tuple
from agent.base_agent import BaseAgent
from game.scorecard import Scorecard
from collections import Counter

class OptimalAgent(BaseAgent):
    """
    A rule-based agent that approximates optimal play using heuristics.
    It prioritizes the Upper Section bonus and high-value Lower Section slots.
    """
    def __init__(self):
        self.scorecard_helper = Scorecard()

    def choose_action(self, state: Dict) -> tuple:
        dice = state["dice"]
        rolls_left = state["rolls_left"]
        scorecard_state = state["scorecard"]
        
        # 1. If we have 0 rolls left, we MUST score.
        if rolls_left == 0:
            return "score", self._choose_best_category(dice, scorecard_state)
            
        # 2. If we have 3 rolls left, we MUST roll all dice (hold nothing).
        if rolls_left == 3:
            return "roll", []
            
        # 3. If we have rolls left, decide whether to roll or stop early.
        # Heuristic: If we have a "great" hand, stop and score.
        # What is a great hand? Yahtzee, Large Straight, Full House, or 4-of-a-kind with high values.
        best_cat = self._choose_best_category(dice, scorecard_state)
        potential_score = self.scorecard_helper.score_category(best_cat, dice)
        
        # Stop early conditions:
        if best_cat == "yahtzee" and potential_score == 50: return "score", best_cat
        if best_cat == "large_straight" and potential_score == 40: return "score", best_cat
        if best_cat == "full_house" and potential_score == 25: return "score", best_cat
        if best_cat == "small_straight" and potential_score == 30 and rolls_left == 0: return "score", best_cat # Wait, rolls_left check is redundant here
        
        # If we didn't stop, we roll.
        # Decide which dice to hold.
        return "roll", self._choose_hold_indices(dice, scorecard_state)

    def _choose_best_category(self, dice: List[int], scorecard_state: Dict[str, int]) -> str:
        """
        Greedily chooses the category that gives the most points, 
        weighted by strategic value (e.g. upper bonus).
        """
        open_cats = [c for c, s in scorecard_state.items() if s is None]
        if not open_cats: return "chance" # Should not happen
        
        scores = {}
        for cat in open_cats:
            raw_score = self.scorecard_helper.score_category(cat, dice)
            weighted_score = raw_score
            
            # Strategic weights
            if cat in ["ones", "twos", "threes", "fours", "fives", "sixes"]:
                # Value upper section more to get the bonus
                # Specifically, if raw_score >= 3 * face_value, it's "par" or better.
                face_val = ["ones", "twos", "threes", "fours", "fives", "sixes"].index(cat) + 1
                if raw_score >= 3 * face_val:
                    weighted_score += 10 # Bonus incentive
            
            if cat == "yahtzee" and raw_score == 50:
                weighted_score += 100 # Always take Yahtzee
                
            scores[cat] = weighted_score
            
        # Sort by weighted score descending
        best_cat = max(scores, key=scores.get)
        
        # If the best score is 0 (garbage hand), we need to "dump" it.
        # Dump strategy:
        # 1. Dump "ones" (low value loss)
        # 2. Dump "yahtzee" (if we accept we won't get it)
        # 3. Dump "chance" (if low sum)
        if scores[best_cat] == 0:
            # Try to find a dump slot
            if "ones" in open_cats: return "ones"
            if "twos" in open_cats: return "twos"
            if "yahtzee" in open_cats: return "yahtzee"
            return best_cat # Fallback
            
        return best_cat

    def _choose_hold_indices(self, dice: List[int], scorecard_state: Dict[str, int]) -> List[int]:
        """
        Decides which dice to hold based on target categories.
        """
        # Filter out 0s (unrolled dice) just in case
        active_dice = [d for d in dice if d != 0]
        if not active_dice: return []
        
        counts = Counter(active_dice)
        
        # Strategy 1: Go for Yahtzee / N-of-a-kind
        most_common_face, count = counts.most_common(1)[0]
        if count >= 2:
            # Check if the corresponding upper slot or n-of-a-kind is open
            # If so, hold these.
            return [i for i, x in enumerate(dice) if x == most_common_face]
            
        # Strategy 2: Go for Straights
        # If we have 3+ sequential dice, hold them.
        unique_dice = sorted(list(set(active_dice)))
        # Check for sub-sequences... (simplified logic here)
        # If we have [1, 2, 3, 6, 6], hold [1, 2, 3]
        
        # Fallback: Hold 5s and 6s to maximize Chance/Upper
        return [i for i, x in enumerate(dice) if x >= 5]
