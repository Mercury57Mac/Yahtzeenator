from simulation.runner import SimulationRunner
from agent.optimal_agent import OptimalAgent

if __name__ == "__main__":
    agent = OptimalAgent()
    runner = SimulationRunner(agent)
    print("Running 1,000 games with Optimal Agent...")
    runner.run_batch(1000, save_logs=True)
