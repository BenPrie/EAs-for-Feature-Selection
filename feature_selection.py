# Imports as always...
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class GeneSift:

    def __init__(self, pop_size, candidate_length, gen_limit, pool_size, tournament_size, crossover_rate, mutation_rate, improve_threshold):
        # Population size (i.e. the number of candidate solutions in each generation).
        self.pop_size = pop_size
        # Length of each candidate solution  (i.e. the number of features.)
        self.candidate_length = candidate_length
        # Limit on the number of generations to prevent excessive computation.
        self.gen_limit = gen_limit
        # Size of the mating pool (must be even and smaller than pop_size).
        self.pool_size = pool_size
        # Size of the tournament for tournament selection (must be smaller than pool_size).
        self.tournament_size = tournament_size
        # Crossover rate.
        self.crossover_rate = crossover_rate
        # Mutation rate.
        self.mutation_rate = mutation_rate
        # Threshold for improvement (used to decide when to terminate early).
        self.improve_threshold = improve_threshold

    # Establish the data we are selecting the features of.
    def establish_data(self, X, y):
        self.X = X
        self.y = y


    # Relatively cheap fitness function (accuracy of model).
    def fitness_function(self, individual):
        # Cast the candidate solution to a boolean array.
        selected_features = [bool(x) for x in individual]

        # Special case where no features are selected.
        if not any(selected_features):
            return 0

        X = self.X[self.X.columns[selected_features]]

        # Split the data into training and testing sets (arbitrarily an 80-20% split).
        X_train, X_test, y_train, y_test = train_test_split(X, self.y , test_size=0.2, random_state=0)

        # Normlaise the data for numerical stability.
        # We normalise after splitting to prevent data leakage.
        ss_train = StandardScaler()
        X_train = ss_train.fit_transform(X_train)

        ss_test = StandardScaler()
        X_test = ss_test.fit_transform(X_test)

        # Define and train a logistic regression model.
        model = LogisticRegression()

        model.fit(X_train, y_train)

        # Determine the accuracy of the model (and hence the fitness of the candidate solution).
        y_pred = model.predict(X_test)

        # TODO: Might we want to scale this with respect to the number of selected features (to favour smaller selections)?
        return accuracy_score(y_true=y_test, y_pred=y_pred)


    # Tournament parent selection. 
    # We will not use replacement, which introduces the possibility for asexual reproduction.
    # The returned ordering of parents will be random (according to the random package)
    def tournament_selection(self, fitness):
        parents = []

        for i in range(self.pool_size):
            # Select from competitors from the population.
            competitors = random.sample(list(range(self.pop_size)), self.tournament_size)

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
    def cut_and_crossfill(self, a, b):
        # Choose a random crossover point.
        # This cannot be the index o the last element, lest there be no second half!
        crossover_point = random.randint(0, self.candidate_length - 2)

        # Split the parents at the crossover point.
        a_first, a_second = a[:crossover_point], a[crossover_point:]
        b_first, b_second = b[:crossover_point], b[crossover_point:]

        # Construct the offspring from the parent segments.
        x = a_first + b_second
        y = b_first + a_second

        return x, y


    # Mutate a permutation (by flipping one gene at random).
    def flip_mutation(self, individual):
        mutant = individual.copy()

        # Choose a random gene.
        gene = random.randint(0, self.candidate_length - 1)

        # Flip it.
        if mutant[gene] == 0:
            mutant[gene] == 1
        else:
            mutant[gene] == 0

        return mutant


    # mu + lambda survivor selection. 
    def mu_plus_lambda(self, pop, fitness, offspring_pop, offspring_fitness):
        # Target population size.
        mu = self.pop_size

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
    def find_optimal_features(self):
        # Count the generations.
        gen = 0

        # Initialise the population.
        pop = [[random.randint(0, 1) for x in range(self.candidate_length)] for y in range(self.pop_size)]

        # Compute the fitness of the initial population.
        fitness = [self.fitness_function(individual) for individual in pop]

        # Begin evolution...

        two_means_ago = 0
        one_mean_ago = 0
        current_mean = 0
        while gen < self.gen_limit:
            # Select the parents.
            parents = self.tournament_selection(fitness)

            # Produce offspring from the parents.
            offspring_pop = []
            offspring_fitness = []
            parents_counter = 0
            while len(offspring_pop) < self.pool_size:
                # Recombination...
                if random.random() < self.crossover_rate:
                    x, y = self.cut_and_crossfill(pop[parents[parents_counter]], pop[parents[parents_counter + 1]])
                else:
                    x, y = pop[parents[parents_counter]].copy(), pop[parents[parents_counter + 1]].copy()

                # Mutation...
                if random.random() < self.mutation_rate:
                    x = self.flip_mutation(x)
                if random.random() < self.mutation_rate:
                    y = self.flip_mutation(y)

                # Add the offspring to the population and evaluate their fitness. 
                offspring_pop.append(x)
                offspring_fitness.append(self.fitness_function(x))
                offspring_pop.append(y)
                offspring_fitness.append(self.fitness_function(y))

                # Update the parent counter.
                parents_counter += 2

            # Select the survivors (which candidate solutions continue to the next generation?).
            pop, fitness = self.mu_plus_lambda(pop, fitness, offspring_pop, offspring_fitness)

            # Update the generation counter.
            gen += 1

            # Early termination if the generations are changing much anymore.
            if gen > 2:
                two_means_ago = one_mean_ago
                one_mean_ago = current_mean
                current_mean = np.mean(fitness)

                if abs(two_means_ago - one_mean_ago) < self.improve_threshold and abs(one_mean_ago - current_mean) < self.improve_threshold:
                    break

        # Return the individual with the highest fitness.
        pairs = [(pop[i], fitness[i]) for i in range(len(pop))]
        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        return sorted_pairs[0][0]