import numpy as np
import tensorflow as tf
import random
from collections import deque
from typing import Dict, List, Tuple
from agent.base_agent import BaseAgent
from game.scorecard import Scorecard

class DQNAgent(BaseAgent):
    """
    A Deep Q-Network (DQN) agent for Yahtzee.
    Uses a neural network to approximate the Q-value function Q(s, a).
    """
    def __init__(self, state_size: int = 0, action_size: int = 45, load_model: str = None, 
                 learning_rate: float = 0.00008, gamma: float = 0.80, epsilon_decay: float = 0.999):
        """
        Initialize the DQN Agent.
        
        Args:
            state_size: Dimension of the state vector (calculated dynamically if 0).
            action_size: Total number of possible actions (32 hold combos + 13 scoring cats).
            load_model: Path to a saved model to load.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.mem_size = 250000
        self.mem_cntr = 0
        
        # Memory allocation moved to after state_size determination
        self.gamma = gamma   # Discount factor (future rewards importance)
        self.epsilon = 1.0   # Exploration rate (start high)
        self.epsilon_min = 0.01 # Minimum exploration rate
        self.epsilon_decay = epsilon_decay # Decay rate per episode
        self.learning_rate = learning_rate
        self.model_path = "data/models/dqn_model.keras"
        
        # Helper for scoring logic
        self.scorecard_helper = Scorecard()
        
        # Define the mapping of actions
        # Actions 0-31: Hold combinations (binary representation)
        # Actions 32-44: Score categories
        self.categories = [
            "ones", "twos", "threes", "fours", "fives", "sixes",
            "three_of_a_kind", "four_of_a_kind", "full_house",
            "small_straight", "large_straight", "yahtzee", "chance"
        ]

        # Build the main network and the target network
        # We need to determine state size first if not provided
        # State: 5 dice (one-hot? or normalized) + 13 scorecard slots (normalized) + rolls_left (one-hot)
        # Dice: 5 * 6 = 30 inputs (one-hot)
        # Scorecard: 13 inputs (0 if open, 1 if filled? Or normalized score?)
        # Rolls left: 3 inputs (one-hot)
        # Upper progress: 1 input (normalized)
        # Total approx: 30 + 13 + 4 + 1 = 48
        # State: 5 dice (one-hot) + 13 scorecard slots (binary) + rolls_left (one-hot) + upper_progress (norm) + 13 potential scores (norm)
        #        + 6 dice counts + 1 sum + 2 straights + 1 max freq + 1 turn + 1 upper diff
        # 30 + 13 + 4 + 1 + 13 + 6 + 1 + 2 + 1 + 1 + 1 = 73
        if self.state_size == 0:
            self.state_size = 73 

        # Pre-allocate memory for faster access (Now that state_size is known)
        self.state_memory = np.zeros((self.mem_size, self.state_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.state_size), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        # Add mask memory for next state valid actions
        self.mask_memory = np.zeros((self.mem_size, self.action_size), dtype=bool)

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        if load_model:
            self.load(load_model)

    def _build_model(self):
        """
        Builds a dense neural network model using Keras.
        Input: State vector
        Output: Q-values for all 45 actions
        """
        with tf.device('/GPU:0'):
            model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(self.state_size,)))
        model.add(tf.keras.layers.Dense(256, activation='relu')) # Hidden layer 1
        model.add(tf.keras.layers.Dense(256, activation='relu')) # Hidden layer 2
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear')) # Output layer
        
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """
        Updates the target network weights to match the main network.
        This stabilizes training.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done, next_mask):
        """
        Stores a transition tuple in the replay buffer.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mask_memory[index] = next_mask
        self.mem_cntr += 1

    def choose_action(self, state_dict: Dict) -> tuple:
        """
        Selects an action using an epsilon-greedy policy.
        """
        # 1. Preprocess state
        state_vec = self._preprocess_state(state_dict)
        state_vec = np.reshape(state_vec, [1, self.state_size])
        
        # 2. Determine valid actions
        # We must mask invalid actions (e.g., scoring in a filled slot, or rolling when 0 rolls left)
        valid_actions = self._get_valid_actions(state_dict)
        
        # 3. Epsilon-greedy logic
        if np.random.rand() <= self.epsilon:
            # Explore: Choose random valid action
            action_idx = random.choice(valid_actions)
        else:
            # Exploit: Choose best valid action from model prediction
            # Optimization: Use model(..., training=False) instead of predict() for speed
            q_values = self.model(state_vec, training=False).numpy()[0]
            
            # Mask invalid actions by setting their Q-values to negative infinity
            mask = np.full(self.action_size, -np.inf)
            mask[valid_actions] = 0
            masked_q_values = q_values + mask
            
            action_idx = np.argmax(masked_q_values)
            
            if np.random.rand() < 0.01: # Only print 1% of the time to avoid spam
                print(f"DEBUG: Max Q-Value: {np.max(q_values):.2f}")
            
        return self._decode_action(action_idx)

    def replay(self, batch_size: int):
        if self.mem_cntr < batch_size:
            return
            
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        dones = self.terminal_memory[batch]
        masks = self.mask_memory[batch]
        
        # Convert to tensors
        states_t = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states_t = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards_t = tf.convert_to_tensor(rewards, dtype=tf.float32)
        actions_t = tf.convert_to_tensor(actions, dtype=tf.int32)
        dones_t = tf.convert_to_tensor(dones, dtype=bool)
        masks_t = tf.convert_to_tensor(masks, dtype=bool)
        
        self._train_step(states_t, next_states_t, actions_t, rewards_t, dones_t, masks_t)

    @tf.function
    def _train_step(self, states, next_states, actions, rewards, dones, masks):
        # 1. Double DQN Target Calculation
        next_q_main = self.model(next_states, training=False)
        
        # Apply mask to next_q_main to ensure we don't pick invalid actions
        # Create a large negative tensor where mask is False
        inf_mask = tf.where(masks, tf.zeros_like(next_q_main), tf.fill(tf.shape(next_q_main), -1e9))
        masked_next_q_main = next_q_main + inf_mask
        
        best_actions = tf.argmax(masked_next_q_main, axis=1, output_type=tf.int32)
        
        next_q_target = self.target_model(next_states, training=False)
        
        # Gather Q_target(s', a_best)
        batch_indices = tf.range(tf.shape(next_states)[0], dtype=tf.int32)
        gather_indices = tf.stack([batch_indices, best_actions], axis=1)
        future_val = tf.gather_nd(next_q_target, gather_indices)
        
        # Bellman Update
        updated_q_values = rewards + self.gamma * future_val * (1.0 - tf.cast(dones, tf.float32))
        
        # 2. Gradient Descent
        with tf.GradientTape() as tape:
            q_values = self.model(states, training=True)
            
            # Gather Q(s, a) for the taken actions
            action_indices = tf.stack([batch_indices, actions], axis=1)
            pred_action_values = tf.gather_nd(q_values, action_indices)
            
            # Compute Huber Loss
            loss = tf.keras.losses.Huber()(updated_q_values, pred_action_values)
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        
    def decay_epsilon(self):
        """
        Decays the exploration rate. Should be called at the end of each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name: str):
        self.model.load_weights(name)

    def save(self, name: str):
        self.model.save_weights(name)

    # --- Helper Methods ---

    def _preprocess_state(self, state: Dict) -> np.ndarray:
        """
        Converts the game state dictionary into a flat neural network input vector.
        """
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
                score_vec.append(1) # Filled
                
        # 3. Rolls left: One-hot encode (0, 1, 2, 3) -> 4 features
        rolls = state["rolls_left"]
        rolls_vec = [0] * 4
        rolls_vec[rolls] = 1
        
        # 4. Upper Section Progress (Normalized by 63)
        upper_prog = state.get("upper_section_progress", 0)
        upper_vec = [upper_prog / 63.0]

        # 5. Potential Scores (Normalized by 50)
        # This gives the agent "glasses" to see what each category is worth right now
        potential_scores = []
        current_dice = state["dice"]
        
        # We need a temporary scorecard helper to calculate scores without modifying state
        # But wait, self.scorecard_helper is already initialized in __init__
        # We can use it.
        for cat in self.categories:
            # If category is already filled, potential is 0 (or we could show what we got, but 0 implies "no gain possible")
            if scorecard[cat] is not None:
                potential_scores.append(0)
            else:
                score = self.scorecard_helper.score_category(cat, current_dice)
                potential_scores.append(score / 50.0) # Normalize (max score is 50 for Yahtzee)
        
        # 6. Dice Counts (Normalized by 5)
        # Count how many of each face value we have
        counts = [0] * 7 # indices 0-6, ignore 0
        for d in dice:
            if d > 0: counts[d] += 1
        counts_vec = [c / 5.0 for c in counts[1:]] # 6 features

        # 7. Sum of Dice (Normalized by 30)
        dice_sum = sum(dice)
        sum_vec = [dice_sum / 30.0]

        # 8. Straight Indicators
        # Small Straight (4 consecutive)
        # Large Straight (5 consecutive)
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
        # Useful for 3/4 of a kind, Yahtzee, Full House
        max_freq = max(counts[1:]) if any(counts[1:]) else 0
        freq_vec = [max_freq / 5.0]

        # 10. Turn Number (Normalized by 13)
        current_turn = state.get("current_turn", 1)
        turn_vec = [current_turn / 13.0]

        # 11. Upper Bonus Difference (Normalized by 63)
        # How far are we from the bonus?
        # If we are at 40, diff is 23. If at 70, diff is -7 (capped at 0 maybe? or let it be negative?)
        # Let's use signed distance normalized. 
        # If we have 0, distance is 63. If we have 63, distance is 0.
        # Range could be approx -30 to 63.
        dist_to_bonus = (63 - upper_prog) / 63.0
        bonus_diff_vec = [dist_to_bonus]

        return np.array(dice_vec + score_vec + rolls_vec + upper_vec + potential_scores + 
                        counts_vec + sum_vec + straights_vec + freq_vec + turn_vec + bonus_diff_vec)

    def _get_valid_actions(self, state: Dict) -> List[int]:
        """
        Returns a list of valid action indices.
        """
        valid_actions = []
        rolls_left = state["rolls_left"]
        scorecard = state["scorecard"]
        
        # Rolling actions (0-31)
        # Valid only if rolls_left > 0
        if rolls_left > 0:
            # All 32 hold combinations are valid
            valid_actions.extend(range(32))
            
        # Scoring actions (32-44)
        # Valid only if the category is open
        # AND usually we force scoring if rolls_left == 0 (but in our engine, you can score anytime)
        # BUT if rolls_left == 3 (start of turn), you usually can't score 0 points immediately? 
        # Actually, standard rules say you roll first.
        # Our engine allows scoring anytime, but let's restrict scoring to when dice are rolled (rolls < 3)
        # UNLESS we want to allow "scratching" a turn? No, you must roll at least once.
        if rolls_left < 3:
            for i, cat in enumerate(self.categories):
                if scorecard[cat] is None:
                    valid_actions.append(32 + i)
                    
        return valid_actions

    def _decode_action(self, action_idx: int) -> Tuple[str, any]:
        """
        Converts action index (0-44) back to (action_type, payload).
        """
        if action_idx < 32:
            # Hold action
            # Convert index to binary string, e.g. 5 -> "00101" -> hold indices 2 and 4
            # We need 5 bits
            binary = format(action_idx, '05b')
            indices = [i for i, bit in enumerate(binary) if bit == '1']
            return "roll", indices
        else:
            # Score action
            cat_idx = action_idx - 32
            return "score", self.categories[cat_idx]

    def _encode_action(self, action_tuple: Tuple[str, any]) -> int:
        """
        Converts (action_type, payload) back to action index.
        Used for replay buffer.
        """
        action_type, payload = action_tuple
        if action_type == "roll":
            # Payload is list of indices
            # Convert to binary integer
            val = 0
            for idx in payload:
                val |= (1 << (4 - idx)) # Be careful with endianness matching decode!
                # In decode: enumerate(binary) means index 0 is MSB?
                # format(5, '05b') -> '00101'. i=0 is '0', i=2 is '1' (value 4?), i=4 is '1' (value 1?)
                # Let's match strictly.
                # If indices=[2, 4], binary should have 1s at pos 2 and 4.
                # '00101'.
            
            # Re-implement decode logic to be sure:
            # binary = format(action_idx, '05b')
            # indices = [i for i, bit in enumerate(binary) if bit == '1']
            # So if action_idx=5 ('00101'), indices are [2, 4].
            
            # Encode:
            # We want to set bit i for each i in indices.
            # But '05b' string is MSB first.
            # So bit 0 corresponds to 2^4 (16), bit 4 corresponds to 2^0 (1).
            # Wait, enumerate gives index from left.
            # '00101': i=0(val 16) is 0. i=2(val 4) is 1. i=4(val 1) is 1.
            # So indices [2, 4] -> val 4 + 1 = 5.
            # Correct logic: val += 2^(4-i)
            val = 0
            for idx in payload:
                val += (1 << (4 - idx))
            return val
            
        else:
            # Score action
            cat = payload
            return 32 + self.categories.index(cat)
