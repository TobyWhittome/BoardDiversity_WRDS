import numpy as np
import pandas as pd
import MCDA as mcda
# Import your MCDA module here, make sure mcda.main(df, weights) is correctly defined.

# Load and clean your data
df = pd.read_excel('dataset/transformed_dataset.xlsx').dropna()

""" def generate_initial_population(pop_size, n_weights, increment):
population = []
for _ in range(pop_size):
    weights = np.random.choice(np.arange(increment, 1.0, increment), size=n_weights)
    weights /= weights.sum()
    population.append(weights)
return np.array(population) """

def generate_initial_population(pop_size, n_weights, increment, starting_weights=None):
    """Generate an initial population of random weights, including a given starting point."""
    population = []
    if starting_weights is not None:
        # Ensure the starting weights are in the correct format and normalized
        starting_weights = np.array(starting_weights)
        starting_weights /= starting_weights.sum()
        # Add the starting weights to the population
        population.append(starting_weights)
        pop_size -= 1  # Reduce the remaining population size to generate
    
    for _ in range(pop_size):
        weights = np.random.choice(np.arange(increment, 1.0, increment), size=n_weights)
        weights /= weights.sum()
        population.append(weights)
    
    return np.array(population)


def calculate_fitness(population):
    """Calculate fitness for each individual in the population."""
    fitness_scores = []
    for weights in population:
        fitness = mcda.main(df, weights)  # Assuming this function returns a correlation value.
        fitness_scores.append(fitness)
    return np.array(fitness_scores)

def select_parents(population, fitness, num_parents):
    """Select the best individuals in the current generation as parents for producing the offspring of the next generation."""
    parents = np.empty((num_parents, population.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0][0]
        parents[parent_num, :] = population[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999  # Ensure this parent is not selected again.
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1) % parents.shape[0]
        # The new offspring will have half of its genes from each parent
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutate(offspring_crossover, num_mutations=1, increment=0.001):
    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            random_value = np.random.choice(np.arange(increment, 1.0, increment))
            offspring_crossover[idx, gene_idx] += random_value
            gene_idx += mutations_counter
    # Ensure weights sum to 1
    offspring_crossover /= offspring_crossover.sum(axis=1)[:, np.newaxis]
    return offspring_crossover

""" def genetic_algorithm(generations=10, pop_size=50, n_weights=9, num_parents_mating=20, increment=0.01):
    population = generate_initial_population(pop_size, n_weights, increment)
    print("Initial population generated.")
    
    for generation in range(generations):
        print(f"Generation {generation}")
        fitness = calculate_fitness(population)
        parents = select_parents(population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, offspring_size=(pop_size - parents.shape[0], n_weights))
        offspring_mutation = mutate(offspring_crossover, increment=increment)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
    
    # Last generation
    fitness_last_gen = calculate_fitness(population)
    best_match_idx = np.where(fitness_last_gen == np.max(fitness_last_gen))[0][0]
    
    print("Best solution:", population[best_match_idx, :])
    print("Best solution fitness:", fitness_last_gen[best_match_idx]) """


def genetic_algorithm(generations=50, pop_size=50, n_weights=9, num_parents_mating=20, increment=0.001, starting_weights=None):
    # Generate initial population with starting weights
    population = generate_initial_population(pop_size, n_weights, increment, starting_weights=starting_weights)
    print("Initial population generated, including starting weights.")
    
    for generation in range(generations):
        print(f"Generation {generation}")
        fitness = calculate_fitness(population)
        print(fitness)
        parents = select_parents(population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, offspring_size=(pop_size - parents.shape[0], n_weights))
        offspring_mutation = mutate(offspring_crossover, increment=increment)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation
    
    # Evaluate the last generation
    fitness_last_gen = calculate_fitness(population)
    best_match_idx = np.where(fitness_last_gen == np.max(fitness_last_gen))[0][0]
    
    print("Best solution:", population[best_match_idx, :])
    print("Best solution fitness:", fitness_last_gen[best_match_idx])


# Run the GA
starting_weights = [0.01623804, 0.06813909, 0.05713667, 0.01904556, 0.00377353, 0.02620886,
 0.00809593, 0.02909047, 0.77227185]
genetic_algorithm(starting_weights=starting_weights)



#Outputs:
#Best solution [7.40128150e-04 4.36675609e-02 3.56649252e-02 2.00759761e-02, 1.12879591e-04 1.58934464e-02 5.14730934e-03 2.24856145e-02, 8.56212160e-01]
#Best solution fitness: 0.13772301489205954


