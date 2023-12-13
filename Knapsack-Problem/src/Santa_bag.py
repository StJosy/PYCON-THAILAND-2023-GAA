"""The Holly Bureau strikes again! This time, they've set a new limitation on the magical capacity of Santa's gift bag. 
In a test of holiday ingenuity, Santa must now select the most diverse and optimal assortment of presents from a given list, 
ensuring that each gift brings maximum joy while adhering to the new weight constraints. 
It's not just about fitting everything into the bag; it's about choosing the right combination of toys, games, 
and treats that will make every child's Christmas wish come true. 
Santa, with his list and a twinkle in his eye, must now become a master of merry mathematics to solve this festive conundrum."""

import json
from deap import base, creator, tools, algorithms
import random
import numpy

# Parsing the JSON input to get the items ( Name, weight, value)
santa_bag_items_json = """
[
    ["Toy Car", 500, 500],
    ["Doll" ,200, 600],
    ["Book",150, 160],
    ["Puzzle",60, 350],
    ["Chocolate" ,40, 333],
    ["Board Game",30, 192],
    ["Mints",5, 25],
    ["Toy Sword",10, 38],
    ["Tissues",15, 80],
    ["Phone",500, 200],
    ["Baseball Cap",100,70],
    ["Toy Gub",140,200]
]
"""
items = json.loads(santa_bag_items_json)


# Define the maximum weight the knapsack can hold
max_weight = 1400 #maximum weight in kg

# Genetic Algorithm constants
POPULATION_SIZE = 50
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1

# set the random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Create Fitness and Individual Classes
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(items))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation function
def knapsack_value(individual):
    value = 0
    weight = 0
    for item in range(len(individual)):
        if individual[item] == 1:
            value += items[item][1]
            weight += items[item][2]
    if weight > max_weight:
        return 0, 10000
    return value, weight

toolbox.register("evaluate", knapsack_value)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Genetic Algorithm
population = toolbox.population(n=POPULATION_SIZE)
hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("min", numpy.min)
stats.register("max", numpy.max)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=True)

# Best solution
best_individual = tools.selBest(population, k=1)[0]
best_items = [items[i][0] for i in range(len(best_individual)) if best_individual[i] == 1]
best_value, best_weight = knapsack_value(best_individual)

print("Best Items:", best_items)
print("Total Value:", best_value)
print("Total Weight:", best_weight)
