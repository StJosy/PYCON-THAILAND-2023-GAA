"""This year, Santa has a new challenge. The Holly Bureau has set a limit on how heavy Santa's bag can be. 
Santa needs to choose the best gifts to put in his bag, but he can't make it too heavy. 
He has many different kinds of gifts, like toys, books, and games. 
Each gift has a different size and brings a different amount of happiness to the children. 
Santa's job is to pick the best combination of gifts that will make the most children happy, but he must also make sure his bag is not too heavy to carry. It's like a puzzle where Santa must balance the weight of the bag and the joy each gift brings. He needs to think carefully to make the best choice for everyone's Christmas morning.
"""

import json
from deap import base, creator, tools, algorithms
import random
import numpy
import matplotlib.pyplot as plt

# Parsing the JSON input to get the items
santa_bag_items_json = """
[
    ["Laptop", 500, 2200],
    ["Tablet" ,200, 1100],
    ["Headphones",150, 160],
    ["Coffee Mug",60, 350],
    ["Notepad" ,40, 333],
    ["Water Bottle",30, 192],
    ["Mints",5, 25],
    ["Socks",10, 38],
    ["Tissues",15, 80],
    ["Phone",500, 200],
    ["Baseball Cap",100,70],
    ["Muppet",140,200]
]
"""
items = json.loads(santa_bag_items_json)


# Define the maximum weight the santa travel bag (knapsack) can hold
max_weight = 2500

# Genetic Algorithm constants
POPULATION_SIZE = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 1

# set the random seed
RANDOM_SEED = 42
#random.seed(RANDOM_SEED)

# Evaluation function
def knapsack_value(individual):
    value = 0
    weight = 0
    
    #it'S calclate the current weight
    for item in range(len(individual)):
        if individual[item] == 1:
            value += items[item][1]
            weight += items[item][2]
    #in case of vodiate the constraionts return extramay value.         
    if weight > max_weight:
        return 0, 10000
    return value, weight




def visualize_knapsack_pie(items, best_individual, focus='value'):
    # Filter the selected items
    selected_items = [(item[0], item[1], item[2]) for item, included in zip(items, best_individual) if included == 1]
    if not selected_items:
        print("No items selected in the best solution.")
        return

    # Depending on the focus, extract either values or weights
    if focus == 'value':
        labels, sizes = zip(*[(item[0], item[1]) for item in selected_items])
        title = 'Value Distribution in Knapsack'
    elif focus == 'weight':
        labels, sizes = zip(*[(item[0], item[2]) for item in selected_items])
        title = 'Weight Distribution in Knapsack'
    else:
        raise ValueError("Focus must be either 'value' or 'weight'.")

    # Plotting the pie chart
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=140)

    # Adding a legend
    plt.legend(wedges, labels, title="Items", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")

    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.title(title)

    plt.show()


def main():    
    # Create Fitness and Individual Classes
    
    #Each element in the weights tuple corresponds to an objective in a multi-objective optimization problem. 
    creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
    
    #An invidual descritpion what equvalent of a chromosome. Each element in list is a gene.
    creator.create("Individual", list, fitness=creator.Fitness)

    #A toolbox for evolution that contains the evolutionary operators.
    toolbox = base.Toolbox()
    
    #this registered function (toolbox.attr_bool) is used to randomly generate these binary values. 
    #Each call to toolbox.attr_bool() will return either 0 or 1, randomly.
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, len(items))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("evaluate", knapsack_value)
    
    #cxTwoPoint selects two points on the parent chromosomes and exchanges the genes between these points. 
    # It's a standard crossover method suitable for a wide range of genetic algorithms
    toolbox.register("mate", tools.cxTwoPoint)
    
    #This mutation operator flips the value of the genes in an individual. 
    # For a binary representation (like in the Knapsack Problem, where each gene is either 0 or 1), 
    # it changes 1 to 0 and vice versa.
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    
    #A group of individuals is randomly selected from the population. The size of this group is determined by tournsize.
    #The winner of this tournament is typically the individual with the highest fitness (for maximization problems) or lowest fitness (for minimization problems), 
    #depending on how fitness is defined in your problem.
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
    visualize_knapsack_pie(items,best_individual)


    
if __name__ == "__main__":
    main()
    

