from typing import List, Dict, Optional
from game.dice import Dice
from game.scorecard import Scorecard
import numpy as np

class GameEngine:
    """
    Manages the flow of a Yahtzee game.
    Handles turns, rolls, and agent interactions.
    """
    def __init__(self, seed: Optional[int] = None):
        self.dice = Dice(seed)
        self.scorecard = Scorecard()
        self.current_turn = 1
        self.max_turns = 13
        self.rolls_left = 3
        self.game_over = False

    def start_game(self):
        """
        Initializes the game state.
        """
        self.dice.reset()
        self.scorecard = Scorecard()
        self.current_turn = 1
        self.game_over = False
        self.start_turn()

    def reset(self):
        """
        Resets the game and returns the initial state.
        Compatible with standard RL environments.
        """
        self.start_game()
        return self.get_state()

    def start_turn(self):
        """
        Sets up a new turn.
        """
        self.dice.reset()
        self.dice.roll()
        self.rolls_left = 2
        # Auto-roll the first roll of the turn

    def step(self, action_type: str, action_payload: any):
        """
        Executes a step in the game based on the agent's action.
        
        Args:
            action_type: "roll" or "score"
            action_payload: 
                - If "roll": list of indices to hold (e.g., [0, 1, 4])
                - If "score": category string (e.g., "full_house")
        
        Returns:
            observation: The new state of the game.
            reward: The reward for this step (usually 0 unless scoring).
            done: Boolean indicating if the game is over.
            info: Extra debug info.
        """
        reward = 0
        done = False
        info = {}

        if self.game_over:
            return self.get_state(), 0, True, {"error": "Game is already over."}

        if action_type == "roll":
            if self.rolls_left > 0:
                # 1. Hold dice
                indices_to_hold = action_payload if action_payload is not None else []
                self.dice.hold(indices_to_hold)
                
                # Calculate potential BEFORE roll
                potential_before = self.scorecard.calculate_max_potential_score(self.dice.get_state())

                # 2. Roll unheld dice
                self.dice.roll()
                self.rolls_left -= 1
                info["roll_result"] = self.dice.get_state()

                # Calculate potential AFTER roll
                potential_after = self.scorecard.calculate_max_potential_score(self.dice.get_state())   

                # Reward is the difference between potential scores
                delta = potential_after - potential_before 
                if delta > 0:
                    reward = delta * 0.1

            else:
                # Invalid roll action (no rolls left)
                reward = -10 # Penalty for invalid actio
                print("ILLEGAL MOVE")
                info["error"] = "No rolls left for this turn. Must score."
        elif action_type == "score":
            category = action_payload
            try:
                # 1. Track Upper Score BEFORE the move
                upper_before = self.scorecard.get_upper_section_score() # You need this method
                
                # 2. Register the score
                points_scored = self.scorecard.register_score(category, self.dice.get_state())
                
                # 3. Track Upper Score AFTER the move
                upper_after = self.scorecard.get_upper_section_score()
                
                # 4. Base Reward
                reward = points_scored
                
                # Check if we just crossed the 63 threshold this specific turn
                if upper_before < 63 and upper_after >= 63:
                    reward += 35  # TEACH IT TO CHASE THE BONUS!
                    info["bonus_triggered"] = True
                
                info["category"] = category
                info["score"] = points_scored
                
                # End of turn logic
                if self.scorecard.is_complete() or self.current_turn >= self.max_turns:
                    self.game_over = True
                    done = True
                else:
                    self.current_turn += 1
                    self.start_turn()
                    done = False
                    
            except ValueError as e:
                # ... (Your error handling is good) ...
                print("ILLEGAL MOVE")
                reward = -100
                done = True
                self.game_over = True
            
        else:
            raise ValueError(f"Unknown action type: {action_type}")

        return self.get_state(), reward, done, info

    def get_state(self):
        upper_score = self.scorecard.get_upper_section_score()
        return {
            "dice": self.dice.get_state(),
            "rolls_left": self.rolls_left,
            "scorecard": self.scorecard.scores.copy(),
            # Only show 35 if the UPPER section specifically is high enough
            "upper_bonus": 35 if upper_score >= 63 else 0, 
            "upper_section_progress": upper_score, # Optional: Help the bot see how close it is
            "current_turn": self.current_turn
        }
