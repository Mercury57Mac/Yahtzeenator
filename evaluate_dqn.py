import numpy as np
import os
import matplotlib.pyplot as plt
from agent.dqn_agent import DQNAgent
from game.engine import GameEngine

def evaluate(num_episodes=100, model_path="data/models/dqn_final.weights.h5"):
    print(f"Evaluating DQN agent for {num_episodes} episodes...")
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    # Initialize environment and agent
    engine = GameEngine()
    agent = DQNAgent()
    
    # Load the trained weights
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Set epsilon to 0 for pure exploitation (no random moves)
    agent.epsilon = 0.0
    
    scores = []
    
    for e in range(num_episodes):
        state = engine.reset()
        total_reward = 0
        game_score = 0
        done = False
        
        while not done:
            # Choose action (greedy)
            action_tuple = agent.choose_action(state)
            action_type, action_payload = action_tuple
            
            # Execute action
            next_state_dict, reward, done, _ = engine.step(action_type, action_payload)
            
            if action_type == "score":
                game_score += reward
            
            state = next_state_dict
            total_reward += reward
            
        scores.append(game_score)
        if (e + 1) % 10 == 0:
            print(f"Episode {e+1}/{num_episodes} | Game Score: {game_score}")

    # Statistics
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_dev = np.std(scores)
    
    print("\n--- Evaluation Results ---")
    print(f"Episodes: {num_episodes}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Max Score: {max_score}")
    print(f"Min Score: {min_score}")
    print(f"Std Dev: {std_dev:.2f}")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(avg_score, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_score:.2f}')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title(f'DQN Evaluation Score Distribution ({num_episodes} Games)')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    os.makedirs("data/plots", exist_ok=True)
    plot_path = "data/plots/evaluation_distribution.png"
    plt.savefig(plot_path)
    print(f"Score distribution plot saved to {plot_path}")

if __name__ == "__main__":
    evaluate()
