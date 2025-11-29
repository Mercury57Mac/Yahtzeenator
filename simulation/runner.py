from game.engine import GameEngine
from agent.base_agent import BaseAgent
from agent.random_agent import RandomAgent
import time

class SimulationRunner:
    """
    Runs simulations of the Yahtzee game with a given agent.
    """
    def __init__(self, agent: BaseAgent):
        self.agent = agent
        self.engine = GameEngine()

    def run_single_game(self, verbose: bool = False):
        """
        Runs a single game (or turn, in our simplified mode).
        """
        self.engine.start_game()
        done = False
        total_reward = 0
        
        if verbose:
            print("Starting new game...")
        
        while not done:
            state = self.engine.get_state()
            if verbose:
                print(f"Turn {state['current_turn']} | Rolls Left: {state['rolls_left']}")
                print(f"Dice: {state['dice']}")
                print(f"Scorecard: {state['scorecard']}")
                
            action_type, action_payload = self.agent.choose_action(state)
            
            if verbose:
                print(f"Agent Action: {action_type} -> {action_payload}")
                
            state, reward, done, info = self.engine.step(action_type, action_payload)
            total_reward += reward
            
            if verbose and info.get("error"):
                print(f"ERROR: {info['error']}")
            
            if done and verbose:
                print("-" * 20)
                print(f"Game Over!")
                print(f"Final Score: {self.engine.scorecard.get_total_score()}")
                print(f"Scorecard: {self.engine.scorecard.scores}")
                print("-" * 20)
                
        return self.engine.scorecard.get_total_score()

    def run_batch(self, num_games: int, save_logs: bool = True):
        """
        Runs a batch of games and returns statistics.
        """
        scores = []
        game_logs = []
        start_time = time.time()
        
        for i in range(num_games):
            # Run game (we need to modify run_single_game to return log data if we want detailed logs)
            # For now, let's just log the final score and scorecard.
            self.engine.start_game()
            done = False
            game_log = {"game_id": i, "turns": []}
            
            while not done:
                state = self.engine.get_state()
                action_type, action_payload = self.agent.choose_action(state)
                next_state, reward, done, info = self.engine.step(action_type, action_payload)
                
                turn_data = {
                    "turn": state["current_turn"],
                    "rolls_left": state["rolls_left"],
                    "dice_before": state["dice"],
                    "action_type": action_type,
                    "action_payload": action_payload,
                    "reward": reward,
                    "scorecard_snapshot": state["scorecard"].copy()
                }
                game_log["turns"].append(turn_data)
                
            score = self.engine.scorecard.get_total_score()
            scores.append(score)
            game_log["final_score"] = score
            game_log["final_scorecard"] = self.engine.scorecard.scores
            game_logs.append(game_log)
            
        duration = time.time() - start_time
        avg_score = sum(scores) / len(scores)
        print(f"Ran {num_games} games in {duration:.2f}s")
        print(f"Average Score: {avg_score:.2f}")
        print(f"Max Score: {max(scores)}")
        print(f"Min Score: {min(scores)}")
        
        if save_logs:
            import json
            import os
            from datetime import datetime
            
            log_dir = "data/logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{log_dir}/simulation_{timestamp}.json"
            
            with open(filename, "w") as f:
                json.dump({"summary": {"games": num_games, "avg_score": avg_score}, "logs": game_logs}, f, indent=2)
            print(f"Logs saved to {filename}")
        
        return scores

if __name__ == "__main__":
    # Simple manual test
    agent = RandomAgent()
    runner = SimulationRunner(agent)
    runner.run_single_game(verbose=True)
