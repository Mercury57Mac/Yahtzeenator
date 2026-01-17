import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from agent.dqn_agent import DQNAgent

def verify_features():
    print("Initializing DQNAgent...")
    # Initialize with 0 to force auto-calculation of state_size
    agent = DQNAgent(state_size=0)
    
    print(f"Agent state_size: {agent.state_size}")
    expected_size = 73
    if agent.state_size != expected_size:
        print(f"ERROR: Expected state_size {expected_size}, got {agent.state_size}")
        return False
    else:
        print("SUCCESS: state_size is correct.")

    # Create a dummy state
    dummy_state = {
        "dice": [1, 2, 3, 4, 5], # Large straight
        "rolls_left": 2,
        "scorecard": {
            "ones": None, "twos": None, "threes": None, "fours": None, "fives": None, "sixes": None,
            "three_of_a_kind": None, "four_of_a_kind": None, "full_house": None,
            "small_straight": None, "large_straight": None, "yahtzee": None, "chance": None
        },
        "upper_section_progress": 30,
        "current_turn": 5
    }

    print("\nTesting _preprocess_state...")
    state_vec = agent._preprocess_state(dummy_state)
    print(f"State vector shape: {state_vec.shape}")
    
    if state_vec.shape[0] != expected_size:
        print(f"ERROR: Preprocessed state vector has wrong shape. Expected {expected_size}, got {state_vec.shape[0]}")
        return False
    
    # Check specific feature values if possible
    # Indices:
    # 0-29: Dice (one-hot)
    # 30-42: Scorecard (binary)
    # 43-46: Rolls left (one-hot)
    # 47: Upper progress
    # 48-60: Potential scores
    # 61-66: Dice counts (6)
    # 67: Sum (1)
    # 68-69: Straights (2)
    # 70: Max freq (1)
    # 71: Turn (1)
    # 72: Upper diff (1)

    # Check Straights (indices 68, 69)
    # Dice [1, 2, 3, 4, 5] is a large straight (and small straight)
    has_small = state_vec[68]
    has_large = state_vec[69]
    print(f"Small Straight Feature: {has_small} (Expected 1.0)")
    print(f"Large Straight Feature: {has_large} (Expected 1.0)")
    
    if has_small != 1.0 or has_large != 1.0:
        print("ERROR: Straight detection failed.")
        return False

    print("\nTesting choose_action...")
    try:
        action = agent.choose_action(dummy_state)
        print(f"Action chosen: {action}")
    except Exception as e:
        print(f"ERROR: choose_action failed with exception: {e}")
        return False

    print("\nALL CHECKS PASSED.")
    return True

if __name__ == "__main__":
    verify_features()
