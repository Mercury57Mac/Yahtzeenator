import numpy as np
import os
import json
from agent.dqn_agent import DQNAgent
from game.engine import GameEngine

# Training Parameters
EPISODES = 5000
BATCH_SIZE = 256
TARGET_UPDATE_FREQ = 10
SAVE_FREQ = 100

def train():
    # Initialize environment and agent
    engine = GameEngine()
    # agent = DQNAgent() // Blank Agent 
    agent = DQNAgent()
    
    # Create directories for saving
    os.makedirs("data/models", exist_ok=True)
    
    print(f"Starting training for {EPISODES} episodes...")
    # Tracking
    scores = []
    avg_turns_history = []
    
    # Detailed tracking for visualization
    history = {
        "episodes": [],
        "scores": [],
        "avg_turns": [],
        "epsilon": [],
        "actions": {
            "roll": [], # Count of roll actions per episode
            "score": [], # Count of score actions per episode
            "holds": [] # List of hold counts (0-5) for every roll action across all episodes
        }
    }

    for e in range(EPISODES):
        state = engine.reset()
        total_reward = 0
        done = False
        turns_survived = 0
        step_count = 0 # Initialize step_count for each episode
        
        # Episode stats
        ep_rolls = 0
        ep_scores = 0
        game_score = 0
        
        while not done:
            step_count += 1
            turns_survived += 1
            
            # 1. Choose action
            action_tuple = agent.choose_action(state)
            action_type, action_payload = action_tuple
            
            # Track action stats
            if action_type == "roll":
                ep_rolls += 1
                # Payload is tuple of dice indices to hold. Length is num held.
                num_held = len(action_payload)
                history["actions"]["holds"].append(num_held)
            elif action_type == "score":
                ep_scores += 1
            
            # 2. Take action
            next_state_dict, reward, done, _ = engine.step(action_type, action_payload)
            
            if action_type == "score":
                game_score += reward
            
            # 3. Store experience
            # We need to encode the action for storage
            action_idx = agent._encode_action(action_tuple)
            
            # Preprocess states for storage
            state_vec = agent._preprocess_state(state)
            next_state_vec = agent._preprocess_state(next_state_dict)
            
            # Calculate valid action mask for next state
            next_valid_actions = agent._get_valid_actions(next_state_dict)
            next_mask = np.zeros(agent.action_size, dtype=bool)
            next_mask[next_valid_actions] = True
            
            agent.remember(state_vec, action_idx, reward, next_state_vec, done, next_mask)
            
            state = next_state_dict
            total_reward += reward
            
            # 4. Train
            if step_count % 4 == 0:
                agent.replay(BATCH_SIZE)
                
        # End of episode
        agent.decay_epsilon()
        
        # Update Target Network every 10 episodes
        if (e + 1) % 10 == 0:
            agent.update_target_model()
        
        scores.append(game_score) # Track GAME SCORE, not total reward
        avg_turns_history.append(turns_survived)
        avg_turns = np.mean(avg_turns_history[-100:])
        avg_score = np.mean(scores[-100:])
        
        # Update history
        history["episodes"].append(e)
        history["scores"].append(game_score) # Save GAME SCORE
        history["avg_turns"].append(avg_turns)
        history["epsilon"].append(agent.epsilon)
        history["actions"]["roll"].append(ep_rolls)
        history["actions"]["score"].append(ep_scores)
        
        print(f"Episode {e+1}/{EPISODES} | Game Score: {game_score} | Reward: {total_reward:.1f} | Turns: {turns_survived} | Avg Score: {avg_score:.2f} | Epsilon: {agent.epsilon:.2f}")
        
        if (e + 1) % SAVE_FREQ == 0:
            agent.save(f"data/models/dqn_model_{e+1}.weights.h5")
            # Save history
            with open("data/training_history.json", "w") as f:
                json.dump(history, f)
            
    # Save final model
    agent.save("data/models/dqn_final.weights.h5")
    print("Training complete!")

if __name__ == "__main__":
    train()
