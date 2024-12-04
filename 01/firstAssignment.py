# Simple Genetic Algorithm with NumPy Random and Seed for Determinism

import numpy as np
import copy
import matplotlib.pyplot as plt

# Parameters (default settings)
POPULATION_SIZE = 100          # Adjust as needed
CHROMOSOME_LENGTH = 20         # Length of the bitstring
MAX_GENERATIONS = 100          # Maximum number of generations
MUTATION_PROBABILITY = 0.01    # Experiment with different values
CROSSOVER_PROBABILITY = 0.7    # Experiment with different values
SEED = 50                      # Set a seed for reproducibility

# Set the seed for NumPy's random number generator
np.random.seed(SEED)

def initialize_population():
    # Initialize the population with random individuals using NumPy
    population = []
    for _ in range(POPULATION_SIZE):
        # Create a random individual
        individual = np.random.randint(2, size=CHROMOSOME_LENGTH).tolist()
        population.append(individual)
    return population

def fitness(individual):
    # Define the fitness function
    # Uncomment the fitness function you want to use

    # For OneMAX problem (maximize number of ones)
    #fitness_value = sum(individual)

    # For alternating 1s and 0s starting with 0 (e.g., 010101...)
    # pattern = [i % 2 for i in range(CHROMOSOME_LENGTH)]
    # fitness_value = sum(1 for i in range(CHROMOSOME_LENGTH) if individual[i] == pattern[i])

    # For alternating 1s and 0s starting with 1 (e.g., 101010...)
    pattern = [(i + 1) % 2 for i in range(CHROMOSOME_LENGTH)]
    fitness_value = sum(1 for i in range(CHROMOSOME_LENGTH) if individual[i] == pattern[i])

    return fitness_value

def selection(population, fitnesses, N=POPULATION_SIZE):
    # Implement roulette wheel selection using NumPy's random choice with probabilities
    total_fitness = sum(fitnesses)
    selection_probabilities = [f / total_fitness for f in fitnesses]
    selected_indices = np.random.choice(len(population), size=N, p=selection_probabilities)
    selected = [copy.deepcopy(population[i]) for i in selected_indices]
    return selected

def crossover(parent1, parent2):
    # Implement single-point crossover using NumPy
    if np.random.rand() < CROSSOVER_PROBABILITY:
        crossover_point = np.random.randint(1, CHROMOSOME_LENGTH)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        # No crossover; children are copies of parents
        child1, child2 = parent1[:], parent2[:]
    return child1, child2

def mutate(individual):
    # Implement mutation operator using NumPy
    for i in range(CHROMOSOME_LENGTH):
        if np.random.rand() < MUTATION_PROBABILITY:
            # Flip the bit
            individual[i] = 1 - individual[i]
    return individual

def genetic_algorithm():
    population = initialize_population()
    best_fitnesses = []
    best_individual = None
    best_fitness = 0

    for generation in range(MAX_GENERATIONS):
        # Evaluate fitnesses
        fitnesses = [fitness(individual) for individual in population]
        # Record best fitness
        current_best_fitness = max(fitnesses)
        current_best_individual = population[fitnesses.index(current_best_fitness)]

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = current_best_individual

        best_fitnesses.append(best_fitness)
        # Print progress (optional)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # Check for convergence (optional)
        if best_fitness == CHROMOSOME_LENGTH:
            print(f"Optimal solution found at generation {generation}")
            break

        # Selection
        selected = selection(population, fitnesses)

        # Create next generation
        next_generation = []
        for i in range(0, POPULATION_SIZE, 2):
            parent1 = selected[i]
            parent2 = selected[(i + 1) % POPULATION_SIZE]  # Ensure circular indexing
            # Crossover
            child1, child2 = crossover(parent1, parent2)
            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])

        # Update population
        population = next_generation[:POPULATION_SIZE]  # Ensure population size

    # Print the best individual after the evolution
    print("\nBest Individual after Evolution:")
    print(f"Best Fitness: {best_fitness}")
    print(f"Best Individual: {best_individual}")

    return best_fitnesses, best_individual

if __name__ == "__main__":
    # Run the algorithm with the first setting
    print("Running Genetic Algorithm with Setting 1:")
    MUTATION_PROBABILITY = 0.01
    CROSSOVER_PROBABILITY = 0.7
    best_fitnesses1, best_individual1 = genetic_algorithm()

    # Reset the seed for reproducibility
    np.random.seed(SEED)

    # Run the algorithm with the second setting
    print("\nRunning Genetic Algorithm with Setting 2:")
    MUTATION_PROBABILITY = 0.05  # Changed mutation probability
    CROSSOVER_PROBABILITY = 0.9  # Changed crossover probability
    best_fitnesses2, best_individual2 = genetic_algorithm()

    # Plotting the convergence of both settings
    generations1 = list(range(len(best_fitnesses1)))
    generations2 = list(range(len(best_fitnesses2)))

    plt.plot(generations1, best_fitnesses1, label='Setting 1')
    plt.plot(generations2, best_fitnesses2, label='Setting 2')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Convergence Comparison of Genetic Algorithm')
    plt.legend()
    plt.savefig('convergence_plot.png')  # You can change the filename and format as needed
    plt.show()