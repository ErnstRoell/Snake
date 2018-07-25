import Snake as sn  # Import Game
import NeuralNetwork as NNN  # Import Neural Network
from Population import Population
import Bot
from Trainer import Trainer
import configuration as cf
from pprint import pprint

# Set path to save the trained NN.
filename = 'C:/Users/gebruiker/documents/programming/python/NN/FullTrainedPopulation.json'

# Generate initial population
pop = Population()
# Generates a new population.
# It is also possible to load a saved one for further training.
pop.generate_population()  
T = Trainer(pop)
T.train_population()
pop.save_population(filename)

