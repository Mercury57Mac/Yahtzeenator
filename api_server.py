import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import os
import sys

# Add project root to path so we can import agent and game modules
sys.path.append(os.getcwd())

from agent.dqn_agent import DQNAgent
from game.scorecard import Scorecard

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class ScorecardModel(BaseModel):
    ones: Optional[int] = None
    twos: Optional[int] = None
    threes: Optional[int] = None
    fours: Optional[int] = None
    fives: Optional[int] = None
    sixes: Optional[int] = None
    threeOfAKind: Optional[int] = None
    fourOfAKind: Optional[int] = None
    fullHouse: Optional[int] = None
    smallStraight: Optional[int] = None
    largeStraight: Optional[int] = None
    yahtzee: Optional[int] = None
    chance: Optional[int] = None

class GameStateRequest(BaseModel):
    dice: List[int]
    rolls_remaining: int
    scorecard: ScorecardModel
    difficulty: str

class AIResponse(BaseModel):
    score_category: str
    dice_to_keep: List[int]

# --- Global Agent ---
agent = None

def get_agent():
    global agent
    if agent is None:
        print("Loading DQN Agent...")
        # Initialize agent
        # Note: state_size will be auto-calculated (should be 73)
        agent = DQNAgent(state_size=0)
        
        # Load weights
        # User requested dqn_model_120, assuming 1200 as 120 doesn't exist
        model_path = "data/models/dqn_model_1200.weights.h5"
        
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            try:
                agent.load(model_path)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Agent will use random weights!")
        else:
            print(f"Warning: Model file {model_path} not found. Agent will use random weights.")
            
    return agent

# --- Helper Functions ---

def map_frontend_scorecard_to_backend(fe_scorecard: ScorecardModel) -> Dict:
    """
    Maps frontend camelCase keys to backend snake_case keys.
    """
    mapping = {
        "ones": "ones",
        "twos": "twos",
        "threes": "threes",
        "fours": "fours",
        "fives": "fives",
        "sixes": "sixes",
        "threeOfAKind": "three_of_a_kind",
        "fourOfAKind": "four_of_a_kind",
        "fullHouse": "full_house",
        "smallStraight": "small_straight",
        "largeStraight": "large_straight",
        "yahtzee": "yahtzee",
        "chance": "chance"
    }
    
    be_scorecard = {}
    fe_dict = fe_scorecard.dict()
    
    for fe_key, be_key in mapping.items():
        be_scorecard[be_key] = fe_dict.get(fe_key)
        
    return be_scorecard

def map_backend_category_to_frontend(be_category: str) -> str:
    """
    Maps backend snake_case category to frontend camelCase.
    """
    mapping = {
        "ones": "ones",
        "twos": "twos",
        "threes": "threes",
        "fours": "fours",
        "fives": "fives",
        "sixes": "sixes",
        "three_of_a_kind": "threeOfAKind",
        "four_of_a_kind": "fourOfAKind",
        "full_house": "fullHouse",
        "small_straight": "smallStraight",
        "large_straight": "largeStraight",
        "yahtzee": "yahtzee",
        "chance": "chance"
    }
    return mapping.get(be_category, "chance")

def calculate_upper_score(scorecard: Dict) -> int:
    upper_cats = ["ones", "twos", "threes", "fours", "fives", "sixes"]
    score = 0
    for cat in upper_cats:
        val = scorecard.get(cat)
        if val is not None:
            score += val
    return score

def estimate_turn_number(scorecard: Dict) -> int:
    """
    Estimates the current turn number (1-13) based on how many categories are filled.
    """
    filled_count = 0
    for val in scorecard.values():
        if val is not None:
            filled_count += 1
    
    return min(filled_count + 1, 13)

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    get_agent()

import time

@app.post("/ai-move", response_model=AIResponse)
async def get_ai_move(request: GameStateRequest):
    agent = get_agent()
    
    be_scorecard = map_frontend_scorecard_to_backend(request.scorecard)
    
    upper_score = calculate_upper_score(be_scorecard)
    current_turn = estimate_turn_number(be_scorecard)
    
    state = {
        "dice": request.dice,
        "rolls_left": request.rolls_remaining,
        "scorecard": be_scorecard,
        "upper_bonus": 35 if upper_score >= 63 else 0,
        "upper_section_progress": upper_score,
        "current_turn": current_turn
    }
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    try:
        action_tuple = agent.choose_action(state)
    finally:
        agent.epsilon = original_epsilon
        
    action_type, action_payload = action_tuple
    
    response = AIResponse(score_category="chance", dice_to_keep=[])
    
    if action_type == "roll":
        response.dice_to_keep = list(action_payload)
        pass
        
    elif action_type == "score":
        fe_category = map_backend_category_to_frontend(action_payload)
        response.score_category = fe_category
        response.dice_to_keep = [] 
        
    
        
        if request.rolls_remaining > 0:
            response.dice_to_keep = [0, 1, 2, 3, 4]
        
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
