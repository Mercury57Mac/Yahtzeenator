from simulation.runner import SimulationRunner
from agent.random_agent import RandomAgent

if __name__ == "__main__":
    agent = RandomAgent()
    runner = SimulationRunner(agent)
    print("Running 10,000 games...")
    runner.run_batch(10000)
