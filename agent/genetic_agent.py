import numpy as np
import tensorflow as tf
import random
from typing import Dict, List, Tuple
from agent.base_agent import BaseAgent
from game.scorecard import Scorecard

class GeneticAgent(BaseAgent):
    """
    A Genetic Algorithm agent for Yahtzee.
    Uses a neural network with fixed topology, where weights are evolved.
    """
    def __init__(self, state_size: int = 73, action_size: int = 45):
        self.state_size = state_size
        self.action_size = action_size
        
        # Helper for scoring logic
        self.scorecard_helper = Scorecard()
        
        self.categories = [
            "ones", "twos", "threes", "fours", "fives", "sixes",
            "three_of_a_kind", "four_of_a_kind", "full_house",
            "small_straight", "large_straight", "yahtzee", "chance"
        ]

        # Build model
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Builds a dense neural network model using Keras.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(256, activation='relu')) 
        model.add(tf.keras.layers.Dense(256, activation='relu')) 
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        return model

    def choose_action(self, state_dict: Dict) -> tuple:
        """
        Selects an action based on the highest Q-value from the network.
        """
        # 1. Preprocess state
        state_vec = self._preprocess_state(state_dict)
        state_vec = np.reshape(state_vec, [1, self.state_size])
        
        # 2. Determine valid actions
        valid_actions = self._get_valid_actions(state_dict)
        
        # 3. Forward pass
        q_values = self.model(state_vec, training=False).numpy()[0]
        
        # Mask invalid actions
        mask = np.full(self.action_size, -np.inf)
        mask[valid_actions] = 0
        masked_q_values = q_values + mask
        
        action_idx = np.argmax(masked_q_values)
            
        return self._decode_action(action_idx)

    def get_genome(self) -> np.ndarray:
        """
        Returns the flattened weights of the model (the genome).
        """
        weights = self.model.get_weights()
        flat_weights = np.concatenate([w.flatten() for w in weights])
        return flat_weights

    def set_genome(self, genome: np.ndarray):
        """
        Sets the model weights from a flattened genome.
        """
        weights = self.model.get_weights()
        new_weights = []
        start = 0
        for w in weights:
            shape = w.shape
            size = np.prod(shape)
            chunk = genome[start:start+size]
            new_weights.append(chunk.reshape(shape))
            start += size
        self.model.set_weights(new_weights)

    # --- Helper Methods (Copied/Adapted from DQNAgent) ---

    def _preprocess_state(self, state: Dict) -> np.ndarray:
        # 1. Dice: One-hot encode 5 dice (values 1-6) -> 30 features
        dice = state["dice"]
        dice_vec = []
        for d in dice:
            d_onehot = [0] * 6
            if d > 0: d_onehot[d-1] = 1
            dice_vec.extend(d_onehot)
            
        # 2. Scorecard: 13 features (0 if open, 1 if filled)
        scorecard = state["scorecard"]
        score_vec = []
        for cat in self.categories:
            if scorecard[cat] is None:
                score_vec.append(0)
            else:
                score_vec.append(1) 
                
        # 3. Rolls left: One-hot encode (0, 1, 2, 3) -> 4 features
        rolls = state["rolls_left"]
        rolls_vec = [0] * 4
        rolls_vec[rolls] = 1
        
        # 4. Upper Section Progress (Normalized by 63)
        upper_prog = state.get("upper_section_progress", 0)
        upper_vec = [upper_prog / 63.0]

        # 5. Potential Scores (Normalized by 50)
        potential_scores = []
        for cat in self.categories:
            if scorecard[cat] is not None:
                potential_scores.append(0)
            else:
                score = self.scorecard_helper.score_category(cat, dice)
                potential_scores.append(score / 50.0)
        
        # 6. Dice Counts (Normalized by 5)
        counts = [0] * 7 
        for d in dice:
            if d > 0: counts[d] += 1
        counts_vec = [c / 5.0 for c in counts[1:]] 

        # 7. Sum of Dice (Normalized by 30)
        dice_sum = sum(dice)
        sum_vec = [dice_sum / 30.0]

        # 8. Straight Indicators
        unique_dice = sorted(list(set([d for d in dice if d > 0])))
        consecutive_count = 0
        max_consecutive = 0
        if len(unique_dice) > 0:
            consecutive_count = 1
            max_consecutive = 1
            for i in range(len(unique_dice) - 1):
                if unique_dice[i+1] == unique_dice[i] + 1:
                    consecutive_count += 1
                    max_consecutive = max(max_consecutive, consecutive_count)
                else:
                    consecutive_count = 1
        
        has_small_straight = 1.0 if max_consecutive >= 4 else 0.0
        has_large_straight = 1.0 if max_consecutive >= 5 else 0.0
        straights_vec = [has_small_straight, has_large_straight]

        # 9. Max Frequency (Normalized by 5)
        max_freq = max(counts[1:]) if any(counts[1:]) else 0
        freq_vec = [max_freq / 5.0]

        # 10. Turn Number (Normalized by 13)
        current_turn = state.get("current_turn", 1)
        turn_vec = [current_turn / 13.0]

        # 11. Upper Bonus Difference (Normalized by 63)
        dist_to_bonus = (63 - upper_prog) / 63.0
        bonus_diff_vec = [dist_to_bonus]

        return np.array(dice_vec + score_vec + rolls_vec + upper_vec + potential_scores + 
                        counts_vec + sum_vec + straights_vec + freq_vec + turn_vec + bonus_diff_vec)

    def _get_valid_actions(self, state: Dict) -> List[int]:
        valid_actions = []
        rolls_left = state["rolls_left"]
        scorecard = state["scorecard"]
        
        if rolls_left > 0:
            valid_actions.extend(range(32))
            
        if rolls_left < 3:
            for i, cat in enumerate(self.categories):
                if scorecard[cat] is None:
                    valid_actions.append(32 + i)
                    
        return valid_actions

    def _decode_action(self, action_idx: int) -> Tuple[str, any]:
        if action_idx < 32:
            binary = format(action_idx, '05b')
            indices = [i for i, bit in enumerate(binary) if bit == '1']
            return "roll", indices
        else:
            cat_idx = action_idx - 32
            return "score", self.categories[cat_idx]
