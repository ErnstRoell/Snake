import Snake as sn  # Import Game
import NeuralNetwork as NNN  # Import Neural Network
from Population import Population
import Bot
from Trainer import Trainer
import configuration as cf
from pprint import pprint

"""
This script provides a toy example that is being trained.

"""

#  Start by changing configurations ( to make them easier and training time shorter).
cf.Graphics = False
cf.maxIt = 1000
cf.popSize = 50
cf.epochs = 20
cf.bestPerformers = 15
cf.luckyfew = 5


# Set path to save the trained NN.
filename = 'C:/Users/gebruiker/documents/programming/python/NN/ExampleTrainedPopulation.json'

# Generate initial population
pop = Population()
# Generates a new population.
# It is also possible to load a saved one for further training.
pop.generate_population()  
T = Trainer(pop)
T.train_population()
pop.save_population(filename)

