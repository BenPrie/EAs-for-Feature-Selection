# Imports as always...
import random
import numpy as np

from fitness import fitness_function


# EA parameters...
# These are literally the first values we've tried. They seem to work fine.
# Hence, there is plenty of room for experimentation!

# Population size (i.e. the number of candidate solutions in each generation).
pop_size = 50
# Length of each candidate solution  (i.e. the number of features.)
candidate_length = 30
# Limit on the number of generations to prevent excessive computation.
gen_limit = 100
# Size of the mating pool (must be even and smaller than pop_size).
pool_size = 20
# Size of the tournament for tournament selection (must be smaller than pool_size).
tournament_size = 5
# Crossover rate.
crossover_rate = 0.9
# Mutation rate.
mutation_rate = 0.2
# Threshold for improvement (used to decide when to terminate early).
improve_threshold = 0.001


# Tournament parent selection. 
# We will not use replacement, which introduces the possibility for asexual reproduction.
# The returned ordering of parents will be random (according to the random package)
def tournament_selection(fitness):
    parents = []

    for i in range(pool_size):
        # Select from competitors from the population.
        competitors = random.sample(list(range(pop_size)), tournament_size)

        # Find the winning competitor (i.e. the competitor with highest fitness).
        winner = competitors[0]
        for competitor in competitors:
            if fitness[competitor] > fitness[winner]:
                winner = competitor

        # Select the winner to be a parent.
        parents.append(winner)

    return parents


# One-point crossover for permutation representations.
# We will use notation: a and b for parents, and x and y for offspring.
def cut_and_crossfill(a, b):
    # Choose a random crossover point.
    # This cannot be the index o the last element, lest there be no second half!
    crossover_point = random.randint(0, candidate_length - 2)

    # Split the parents at the crossover point.
    a_first, a_second = a[:crossover_point], a[crossover_point:]
    b_first, b_second = b[:crossover_point], b[crossover_point:]

    # Construct the offspring from the parent segments.
    x = a_first + b_second
    y = b_first + a_second

    return x, y


# Mutate a permutation (by swapping two genes at random).
def swap_mutation(individual):
    mutant = individual.copy()

    # Choose two genes at random.
    gene_a = random.randint(0, candidate_length - 1)
    gene_b = random.randint(0, candidate_length - 1)

    # Do not allow the genes to be the same -- no swapping with itself!
    while gene_a == gene_b:
        gene_b = random.randint(0, candidate_length - 1)

    # Swap the genes.
    mutant[gene_a], mutant[gene_b] = mutant[gene_b], mutant[gene_a]

    return mutant


# mu + lambda survivor selection. 
def mu_plus_lambda(pop, fitness, offspring_pop, offspring_fitness):
    # Target population size.
    mu = pop_size

    # Join the population with the offspring (likewise with fitness).
    super_pop = pop + offspring_pop
    super_fitness = fitness + offspring_fitness

    # Create a list of candidate-fitness pairs.
    pairs = [(super_pop[i], super_fitness[i]) for i in range(len(super_pop))]

    # Sort the pairs by fitness (descending).
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Select the best (i.e. first) mu pairs.
    new_pop = list(map(lambda x: x[0], sorted_pairs))[:mu]
    new_fitness = list(map(lambda x: x[1], sorted_pairs))[:mu]

    return new_pop, new_fitness


# The main function for excecuting the algorithm.
def find_optimal_features():
    # Count the generations.
    gen = 0

    # Initialise the population.
    pop = [[random.randint(0, 1) for x in range(candidate_length)] for y in range(pop_size)]

    # Compute the fitness of the initial population.
    fitness = [fitness_function(individual) for individual in pop]

    # Begin evolution...

    two_means_ago = 0
    one_mean_ago = 0
    current_mean = 0
    while gen < gen_limit:
        # Select the parents.
        parents = tournament_selection(fitness)

        # Produce offspring from the parents.
        offspring_pop = []
        offspring_fitness = []
        parents_counter = 0
        while len(offspring_pop) < pool_size:
            # Recombination...
            if random.random() < crossover_rate:
                x, y = cut_and_crossfill(pop[parents[parents_counter]], pop[parents[parents_counter + 1]])
            else:
                x, y = pop[parents[parents_counter]].copy(), pop[parents[parents_counter + 1]].copy()

            # Mutation...
            if random.random() < mutation_rate:
                x = swap_mutation(x)
            if random.random() < mutation_rate:
                y = swap_mutation(y)

            # Add the offspring to the population and evaluate their fitness. 
            offspring_pop.append(x)
            offspring_fitness.append(fitness_function(x))
            offspring_pop.append(y)
            offspring_fitness.append(fitness_function(y))

            # Update the parent counter.
            parents_counter += 2

        # Select the survivors (which candidate solutions continue to the next generation?).
        pop, fitness = mu_plus_lambda(pop, fitness, offspring_pop, offspring_fitness)

        # Update the generation counter.
        gen += 1

        # Early termination if the generations are changing much anymore.
        if gen > 2:
            two_means_ago = one_mean_ago
            one_mean_ago = current_mean
            current_mean = np.mean(fitness)

            if abs(two_means_ago - one_mean_ago) < improve_threshold and abs(one_mean_ago - current_mean) < improve_threshold:
                break

    # Return the individual with the highest fitness.
    pairs = [(pop[i], fitness[i]) for i in range(len(pop))]
    sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return sorted_pairs[0][0]