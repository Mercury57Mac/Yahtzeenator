import numpy as np
import random
import copy
import os
import json
from agent.genetic_agent import GeneticAgent
from game.engine import GameEngine
from tqdm import tqdm

# GA Parameters
POPULATION_SIZE = 50
GENERATIONS = 100
GAMES_PER_EVAL = 5
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.1
TOURNAMENT_SIZE = 3
ELITISM_COUNT = 2

def evaluate(agent, engine, genome, games=GAMES_PER_EVAL):
    """
    Evaluates a genome by playing a few games.
    """
    agent.set_genome(genome)
    total_score = 0
    for _ in range(games):
        state = engine.reset()
        done = False
        game_score = 0
        while not done:
            action_tuple = agent.choose_action(state)
            action_type, action_payload = action_tuple
            next_state, reward, done, _ = engine.step(action_type, action_payload)
            if action_type == "score":
                game_score += reward
            state = next_state
        total_score += game_score
    return total_score / games

def tournament_selection(population, fitnesses, k=TOURNAMENT_SIZE):
    """
    Selects the best individual from k random individuals.
    """
    selected_indices = np.random.choice(len(population), k, replace=False)
    best_idx = selected_indices[0]
    best_fit = fitnesses[best_idx]
    
    for idx in selected_indices[1:]:
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
            
    return population[best_idx]

def sbx_crossover(parent1, parent2, eta=15):
    """
    Simulated Binary Crossover (SBX).
    """
    # Create children arrays
    child1 = np.copy(parent1)
    child2 = np.copy(parent2)
    
    # Random number for each gene
    rand = np.random.random(parent1.shape)
    
    # Calculate beta
    beta = np.empty(parent1.shape)
    
    # For rand <= 0.5
    mask1 = rand <= 0.5
    beta[mask1] = (2.0 * rand[mask1]) ** (1.0 / (eta + 1.0))
    
    # For rand > 0.5
    mask2 = rand > 0.5
    beta[mask2] = (1.0 / (2.0 * (1.0 - rand[mask2]))) ** (1.0 / (eta + 1.0))
    
    # Crossover
    # We usually apply crossover with some probability (e.g. 1.0 here as we always cross in the loop)
    # But SBX is often applied variable-wise.
    # The formula:
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
    
    return child1, child2

def gaussian_mutation(genome, rate=MUTATION_RATE, strength=MUTATION_STRENGTH):
    """
    Gaussian Mutation: Add Gaussian noise to genes with probability 'rate'.
    """
    mask = np.random.rand(len(genome)) < rate
    noise = np.random.normal(0, strength, size=len(genome))
    genome[mask] += noise[mask]
    return genome

def run_evolution():
    # Setup
    engine = GameEngine()
    agent = GeneticAgent() # Shared agent instance
    
    # Determine genome size
    dummy_genome = agent.get_genome()
    genome_size = len(dummy_genome)
    print(f"Genome Size: {genome_size}")
    
    # Initialize Population
    # We start with the agent's initial random weights, and perturb them to create population
    base_genome = agent.get_genome()
    population = []
    for _ in range(POPULATION_SIZE):
        # Create random variations of the initial weights
        new_genome = base_genome + np.random.normal(0, 0.5, size=genome_size)
        population.append(new_genome)
        
    history = {
        "generation": [],
        "best_score": [],
        "avg_score": []
    }
    
    os.makedirs("data/ga_models", exist_ok=True)
    
    for gen in range(GENERATIONS):
        print(f"Generation {gen+1}/{GENERATIONS}")
        
        # 1. Evaluate
        fitnesses = []
        for genome in tqdm(population, desc="Evaluating"):
            fit = evaluate(agent, engine, genome)
            fitnesses.append(fit)
            
        # Stats
        best_score = max(fitnesses)
        avg_score = np.mean(fitnesses)
        best_idx = np.argmax(fitnesses)
        
        history["generation"].append(gen)
        history["best_score"].append(best_score)
        history["avg_score"].append(avg_score)
        
        print(f"Best: {best_score:.2f} | Avg: {avg_score:.2f}")
        
        # Save best model of this generation
        agent.set_genome(population[best_idx])
        agent.model.save_weights(f"data/ga_models/gen_{gen+1}_best.weights.h5")
        
        # 2. Selection & Reproduction
        new_population = []
        
        # Elitism
        sorted_indices = np.argsort(fitnesses)[::-1]
        for i in range(ELITISM_COUNT):
            new_population.append(copy.deepcopy(population[sorted_indices[i]]))
            
        # Generate rest of population
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            
            c1, c2 = sbx_crossover(p1, p2)
            
            c1 = gaussian_mutation(c1)
            c2 = gaussian_mutation(c2)
            
            new_population.append(c1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(c2)
                
        population = new_population
        
        # Save history
        with open("data/ga_history.json", "w") as f:
            json.dump(history, f)

if __name__ == "__main__":
    run_evolution()
