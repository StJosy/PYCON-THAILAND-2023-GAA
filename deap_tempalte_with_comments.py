import random
from deap import base, creator, tools, algorithms
import numpy

#The fitness function is a crucial part of genetic algorithms. 
# It measures and quantifies the phenotype's efficiency. 
# Typically, it takes an individual and returns its fitness in a tuple. 
# Although it can require significant computational resources, it can be parallelized.
def evaluate(individual):
    # Fitness calculation logic
    return fitness,

# Create a new fitness class named 'FitnessMax'. This class inherits from 'base.Fitness'.
# The 'weights' tuple with a single element 1.0 indicates that this is a maximization problem.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# Define a new class 'Individual' with list as its base class and a fitness attribute of type 'FitnessMax'.
creator.create("Individual", list, fitness=creator.FitnessMax)

# Create a toolbox for storing various functions and operators.
toolbox = base.Toolbox()

# Register 'attribute' as an alias for generating random integers between 0 and 10.
toolbox.register("attribute", random.randint, 0, 10)  # Example for an attribute

# Register 'individual' to create an individual instance by repeating 'attribute' function n=10 times.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=10)

# Register 'population' to create a list of individuals.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register 'mate' as an alias for the two-point crossover function.
toolbox.register("mate", tools.cxTwoPoint)

# Register 'mutate' as an alias for the flip-bit mutation function with a 5% chance per bit.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)

# Register 'select' as an alias for tournament selection with a tournament size of 3.
toolbox.register("select", tools.selTournament, tournsize=3)

# Register 'evaluate' as an alias for your custom evaluation function.
toolbox.register("evaluate", evaluate)

def main():
    # Initialize a population of 50 individuals.
    pop = toolbox.population(n=50)

    # Create a Hall of Fame to store the best individual.
    hof = tools.HallOfFame(1)

    # Set up a statistics object to track the performance of the population.
    # It captures the fitness values of individuals.
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    # Register statistical measures to be tracked: average, standard deviation, minimum, and maximum of the fitness values.
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    # Execute the genetic algorithm. 'eaSimple' is a basic evolutionary algorithm.
    # - pop: The population to evolve.
    # - toolbox: The toolbox with evolutionary operators.
    # - cxpb: Crossover probability.
    # - mutpb: Mutation probability.
    # - ngen: Number of generations to run the algorithm.
    # - stats: The statistics object to collect data.
    # - halloffame: The Hall of Fame object to store the best individual.
    # - verbose: If True, print log messages.
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    results = main()
    best_individual = results[2][0]
    print("Best Individual:", best_individual)
    print("Best Fitness:", best_individual.fitness.values)

