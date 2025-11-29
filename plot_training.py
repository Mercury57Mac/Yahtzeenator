import json
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(history_file="data/training_history.json"):
    if not os.path.exists(history_file):
        print(f"History file {history_file} not found.")
        return

    with open(history_file, "r") as f:
        history = json.load(f)

    # Create plots directory
    os.makedirs("data/plots", exist_ok=True)

    # 1. Scores and Epsilon
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score', color=color)
    ax1.plot(history['episodes'], history['scores'], color=color, alpha=0.3, label='Score')
    
    # Moving average of scores
    window = 100
    if len(history['scores']) >= window:
        avg_scores = np.convolve(history['scores'], np.ones(window)/window, mode='valid')
        ax1.plot(history['episodes'][window-1:], avg_scores, color='darkblue', linewidth=2, label=f'Avg Score ({window})')

    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Epsilon', color=color)  # we already handled the x-label with ax1
    ax2.plot(history['episodes'], history['epsilon'], color=color, linestyle='--', label='Epsilon')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Training Progress: Scores and Epsilon')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("data/plots/scores_epsilon.png")
    plt.close()

    # 2. Hold Counts Distribution
    holds = history['actions']['holds']
    if holds:
        plt.figure(figsize=(10, 6))
        plt.hist(holds, bins=range(7), align='left', rwidth=0.8, color='purple', edgecolor='black')
        plt.xlabel('Number of Dice Held')
        plt.ylabel('Frequency')
        plt.title('Distribution of Dice Held Counts')
        plt.xticks(range(6))
        plt.grid(axis='y', alpha=0.5)
        plt.savefig("data/plots/hold_distribution.png")
        plt.close()

    # 3. Action Types (Roll vs Score) per Episode
    rolls_per_ep = history['actions']['roll']
    if rolls_per_ep:
        plt.figure(figsize=(12, 6))
        plt.plot(history['episodes'], rolls_per_ep, color='green', alpha=0.5, label='Rolls per Game')
        
        # Moving average
        if len(rolls_per_ep) >= window:
            avg_rolls = np.convolve(rolls_per_ep, np.ones(window)/window, mode='valid')
            plt.plot(history['episodes'][window-1:], avg_rolls, color='darkgreen', linewidth=2, label=f'Avg Rolls ({window})')
            
        plt.xlabel('Episode')
        plt.ylabel('Rolls per Game')
        plt.title('Rolls per Game (Strategy Aggressiveness)')
        plt.legend()
        plt.savefig("data/plots/rolls_per_game.png")
        plt.close()

    print("Plots saved to data/plots/")

if __name__ == "__main__":
    plot_training_history()
