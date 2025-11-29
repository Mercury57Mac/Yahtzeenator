import numpy as np
import os
import json
import optuna
from agent.dqn_agent import DQNAgent
from game.engine import GameEngine

# Optimization Settings
N_TRIALS = 20
EPISODES_PER_TRIAL = 3000
SAVE_DIR = "data/models"

os.makedirs(SAVE_DIR, exist_ok=True)

def objective(trial):
    # 1. Suggest Hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.80, 0.999)
    epsilon_decay = trial.suggest_float("epsilon_decay", 0.990, 0.9999)
    # batch_size = trial.suggest_categorical("batch_size", [128, 256, 512]) # Agent currently hardcodes this in replay(), would need refactor to pass it. 
    # For now let's stick to 512 or pass it if we update agent. 
    # The user didn't ask for batch size specifically, but let's stick to the agent's default or hardcode 512 in the replay call.
    # Actually, DQNAgent.replay takes batch_size as arg.
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    print(f"\n🧪 Trial {trial.number}: LR={learning_rate:.2e}, Gamma={gamma:.4f}, Decay={epsilon_decay:.5f}, Batch={batch_size}")

    # 2. Initialize
    engine = GameEngine()
    # We need to make sure we pass these params. 
    # Note: We updated DQNAgent to accept them.
    agent = DQNAgent(
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_decay=epsilon_decay
    )
    
    scores = []
    
    # 3. Training Loop
    for e in range(EPISODES_PER_TRIAL):
        state = engine.reset()
        done = False
        game_score = 0
        step_count = 0
        
        while not done:
            step_count += 1
            action_tuple = agent.choose_action(state)
            action_type, action_payload = action_tuple
            
            next_state_dict, reward, done, _ = engine.step(action_type, action_payload)
            
            if action_type == "score":
                game_score += reward
            
            # Store & Train
            action_idx = agent._encode_action(action_tuple)
            state_vec = agent._preprocess_state(state)
            next_state_vec = agent._preprocess_state(next_state_dict)
            
            agent.remember(state_vec, action_idx, reward, next_state_vec, done)
            state = next_state_dict
            
            if step_count % 4 == 0:
                agent.replay(batch_size)
        
        agent.decay_epsilon()
        if (e + 1) % 10 == 0:
            agent.update_target_model()
            
        scores.append(game_score)
        
        # Pruning (Optional but good for speed)
        # Report intermediate objective value
        if (e + 1) % 100 == 0:
            avg_score = np.mean(scores[-100:])
            trial.report(avg_score, step=e)
            
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                print(f"✂️ Pruning trial {trial.number} at episode {e+1} (Avg: {avg_score:.2f})")
                raise optuna.exceptions.TrialPruned()
                
            print(f"   Ep {e+1}: Avg {avg_score:.2f} | Epsilon {agent.epsilon:.2f}")

    # 4. Final Result
    final_avg_score = np.mean(scores[-100:])
    print(f"🏁 Trial {trial.number} Finished. Final Avg: {final_avg_score:.2f}")
    
    # Save if it's the best so far (Optuna tracks this, but we want the file)
    # We can check against a global best or just save every trial with its ID
    # best_value = trial.study.best_value if trial.study.best_value else -float('inf')
    # if final_avg_score > best_value:
    #     agent.save(f"{SAVE_DIR}/dqn_optuna_best.weights.h5")
    
    # Let's save every trial model just in case, or maybe just the best at the end of the script.
    # Saving inside objective is tricky because we don't know if it's strictly the best yet (study isn't updated till return).
    # But we can save a "candidate" file.
    agent.save(f"{SAVE_DIR}/dqn_trial_{trial.number}.weights.h5")
    
    return final_avg_score

def run_optimization():
    print("🚀 Starting Bayesian Optimization with Optuna")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value: {study.best_value:.2f}")
    print(f"Best Params: {study.best_params}")
    print("="*50)
    
    # Save best params to file
    with open("data/optuna_best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print("To use these params, update your DQNAgent or train_dqn.py!")
    print(f"The best model weights are in data/models/dqn_trial_{study.best_trial.number}.weights.h5")

if __name__ == "__main__":
    run_optimization()
