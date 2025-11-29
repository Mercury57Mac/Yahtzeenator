from typing import List, Dict, Optional

class Scorecard:
    """
    Manages the scorecard for a Yahtzee game.
    Tracks filled categories, calculates scores, and validates moves.
    """
    def __init__(self):
        # Define categories explicitly to make iteration safer
        self.upper_categories = ["ones", "twos", "threes", "fours", "fives", "sixes"]
        self.lower_categories = ["three_of_a_kind", "four_of_a_kind", "full_house",
                                 "small_straight", "large_straight", "yahtzee", "chance"]
        
        # Merge them for the main storage
        self.scores: Dict[str, Optional[int]] = {
            cat: None for cat in self.upper_categories + self.lower_categories
        }
        
        self.upper_bonus_threshold = 63
        self.upper_bonus_score = 35
        self.yahtzee_bonuses = 0 # Just track the count

    def score_category(self, category: str, dice: List[int]) -> int:
        """
        Calculates the score for a given category with the provided dice.
        Does NOT update the scorecard state.
        """
        counts = {x: dice.count(x) for x in range(1, 7)}
        total_sum = sum(dice)
        
        # Upper Section
        if category == "ones": return counts.get(1, 0) * 1
        if category == "twos": return counts.get(2, 0) * 2
        if category == "threes": return counts.get(3, 0) * 3
        if category == "fours": return counts.get(4, 0) * 4
        if category == "fives": return counts.get(5, 0) * 5
        if category == "sixes": return counts.get(6, 0) * 6
        
        # Lower Section
        if category == "three_of_a_kind":
            return total_sum if any(c >= 3 for c in counts.values()) else 0
            
        if category == "four_of_a_kind":
            return total_sum if any(c >= 4 for c in counts.values()) else 0
            
        if category == "full_house":
            has_three = any(c == 3 for c in counts.values())
            has_two = any(c == 2 for c in counts.values())
            has_five = any(c == 5 for c in counts.values()) # Joker rule/5-of-a-kind
            return 25 if (has_three and has_two) or has_five else 0
            
        if category == "small_straight":
            unique_dice = sorted(list(set(dice)))
            if self._is_subsequence([1, 2, 3, 4], unique_dice) or \
               self._is_subsequence([2, 3, 4, 5], unique_dice) or \
               self._is_subsequence([3, 4, 5, 6], unique_dice):
                return 30
            return 0
            
        if category == "large_straight":
            unique_dice = sorted(list(set(dice)))
            return 40 if unique_dice == [1, 2, 3, 4, 5] or unique_dice == [2, 3, 4, 5, 6] else 0
            
        if category == "yahtzee":
            return 50 if any(c == 5 for c in counts.values()) else 0
            
        if category == "chance":
            return total_sum
            
        return 0

    def _is_subsequence(self, sub: List[int], main: List[int]) -> bool:
        """Helper to check if sub is a subsequence of main."""
        it = iter(main)
        return all(x in it for x in sub)

    def validate_move(self, category: str, dice: List[int]) -> bool:
        if category not in self.scores:
            return False
        return self.scores[category] is None

    # --- THE CRITICAL HELPER METHOD ---
    def get_upper_section_score(self) -> int:
        """
        Returns the sum of the upper section only.
        Used to calculate the +35 bonus trigger.
        """
        total = 0
        for cat in self.upper_categories:
            val = self.scores[cat]
            if val is not None:
                total += val
        return total

    def register_score(self, category: str, dice: List[int]) -> int:
        if not self.validate_move(category, dice):
            raise ValueError(f"Category {category} is already filled or invalid.")
            
        counts = {x: dice.count(x) for x in range(1, 7)}
        is_yahtzee = any(c == 5 for c in counts.values())
        
        score = self.score_category(category, dice)
        
        # Joker Logic
        if is_yahtzee and self.scores["yahtzee"] == 50:
            self.yahtzee_bonuses += 1
            
            # Joker Rule: If Lower Section, force full points
            if category == "full_house": score = 25
            if category == "small_straight": score = 30
            if category == "large_straight": score = 40
        
        self.scores[category] = score
        return score

    def get_total_score(self) -> int:
        """
        Calculates the total score including bonuses.
        """
        # Sum all filled slots
        current_score = sum(s for s in self.scores.values() if s is not None)
        
        # Add Upper Bonus
        if self.get_upper_section_score() >= self.upper_bonus_threshold:
            current_score += self.upper_bonus_score
            
        # Add Yahtzee bonuses (100 per extra Yahtzee)
        current_score += (self.yahtzee_bonuses * 100)
        
        return current_score

    def is_complete(self) -> bool:
        return all(s is not None for s in self.scores.values())

    def calculate_max_potential_score(self, dice: List[int]) -> int:
        """
        Calculates the maximum possible score for the current dice 
        across all OPEN categories. Used for reward shaping.
        """
        max_score = 0
        for cat in self.scores:
            if self.scores[cat] is None:
                score = self.score_category(cat, dice)
                if score > max_score:
                    max_score = score
        return max_score