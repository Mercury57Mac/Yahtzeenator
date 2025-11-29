# Yahtzeenator 🎲🤖

**Yahtzeenator** is a Deep Reinforcement Learning project designed to master the game of Yahtzee. Built from scratch using Python and TensorFlow, this project explores how an AI agent can learn complex probability management and long-term strategic planning without prior knowledge of the game.

## 🚀 Project Overview

The insperation for this project came from playing Yahtzee at a office game night, I thought to myself there has to be an optimal way to play this game, and so I set out to create a program that could learn to play Yahtzee optimally. So far the bot has been learned to play the game, but it is still in development to gain human level performance. 
That being said... I do beleive it could do well in a elementary school tournament. 

### Key Features
*   **Custom Game Engine:** A fully simulated Yahtzee environment built in Python.
*   **Deep Q-Network (DQN):** The core agent utilizes a neural network to approximate Q-values for state-action pairs.
*   **Double DQN (DDQN):** Implemented to mitigate the overestimation bias common in standard DQN.
*   **Valid Action Masking:** The network's output is masked to ensure the agent only considers legal moves, significantly speeding up convergence.
*   **Reward Shaping:** Custom reward functions designed to encourage strategic play, such as prioritizing the Upper Section bonus (+35 points).
*   **Training Stability:** Utilizes **Huber Loss** and optimized hyperparameters to ensure stable learning curves.

## 📊 Current Performance

The agent is currently in active training and development.
*   **Baseline (Random Agent):** ~60 points
*   **Current DQN Agent:** ~120 points (Average) d
*   **Optimistic Target (Human Expert):** >250 points

The agent has successfully learned basic strategies (e.g., prioritizing Yahtzees and large straights) and is now refining its decision-making for more subtle trade-offs.

## 🛠️ Tech Stack
*   **Language:** Python 3.12
*   **ML Framework:** TensorFlow / Keras
*   **Data Processing:** NumPy
*   **Visualization:** Matplotlib

## 🔮 Roadmap & Future Experiments

This project is a playground for advanced AI concepts. Future iterations will explore:

*   **🧬 Genetic Algorithms:** Evolving a population of neural networks to find optimal weights without gradient descent.

## 💻 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mercury57Mac/Yahtzeenator.git
    cd Yahtzeenator
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the agent:**
    ```bash
    python train_dqn.py
    ```

## 👨‍💻 Author

**Macallum Harvey**
*   [GitHub](https://github.com/Mercury57Mac)
