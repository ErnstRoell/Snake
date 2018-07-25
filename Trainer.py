import random
import numpy as np
import Snake as sn  # Import Game
import NeuralNetwork as NNN  # Import Neural Network
from Population import Population
import Bot
import configuration as cf
from pprint import pprint


class Trainer():

    def __init__(self, pop):
        self.population = pop
        self.nexPopulation = []

    def train_population(self):
        for _ in range(cf.epochs):
            self.train_generation()
        
        self.compute_performance()
        print('Top 10')
        pprint(self.population.pop[0:10])

    def train_generation(self):
        self.compute_performance()
        self.select(cf.bestPerformers, cf.luckyFew)
        self.nextGeneration()  # Check this one!!!
        self.mutateGen()

    def compute_performance(self):
        S = sn.snake()
        bot = Bot.NNBot()
        for individual in self.population.pop:
            bot.load_brain(individual)
            S.startGame(bot)
            individual['score'] = bot.score[0]
            bot.score = []
    
        self.population.sort_population()

    def select(self, bestPerformers, luckyFew):
        self.nextPopulation = self.population.pop[0:bestPerformers]

        keyLuckyFew = random.sample(range(bestPerformers+1, 
                                          len(self.population.pop)),
                                    luckyFew)
        for ii in keyLuckyFew:
            self.nextPopulation.append(self.population.pop[ii])

    def nextGeneration(self):
        couples = np.random.choice(self.nextPopulation, (cf.popSize, 2))
        nextGen = []
        for ii in range(cf.popSize):
            nextGen.append(self.Crossover(couples[ii][0], couples[ii][1]))
        
        self.population.pop = nextGen

    def mutateGen(self):
        M = []
        for individual in self.population.pop:
            # random boolean mask for which values will be changed
            maskW1 = np.random.binomial(1, cf.change, individual['W1'].shape).astype(np.bool)
            maskW2 = np.random.binomial(1, cf.change, individual['W2'].shape).astype(np.bool)
            # random matrix the same shape of your data
            r1 = np.random.normal(0, cf.rate, individual['W1'].shape)
            r2 = np.random.normal(0, cf.rate, individual['W2'].shape)
            # use your mask to replace values in your input array
            individual['W1'][maskW1] = individual['W1'][maskW1] + r1[maskW1]
            individual['W2'][maskW2] = individual['W2'][maskW2] + r2[maskW2]
            M.append(individual)
        self.population.pop = M

    def maximum(seq):
        maxi = 0.0
        for x in seq:
            if float(x.score[0][0]) >= maxi:
                maxi = float(x.score[0][0])
        return maxi

    def Crossover(self, NN1, NN2):
        cutMax = cf.cutMax  # Hardcode
        NN = {}
        MM = {}

        NN["W1"] = np.ravel(NN1['W1'])
        NN["W2"] = np.ravel(NN1['W2'])
        MM["W1"] = np.ravel(NN2['W1'])
        MM["W2"] = np.ravel(NN2['W2'])

        idx = np.random.choice(range(1, len(NN["W1"])-1), cutMax, replace=False)
        idx = np.append(idx, [[0], [len(NN["W1"])]])
        idx = np.sort(idx)
        W1 = np.array([])
        W2 = np.array([])

        if cutMax >= 1:
            for ii in range((len(idx)-1)):
                if bool(np.random.binomial(1, 0.5)):
                    W1 = np.append(W1, NN["W1"][idx[ii]:idx[ii+1]])
                else:
                    W1 = np.append(W1, MM["W1"][idx[ii]:idx[ii+1]])

            for ii in range(len(idx)-1):
                if bool(np.random.binomial(1, 0.5)):
                    W2 = np.append(W2, NN["W2"][idx[ii]:idx[ii+1]])
                else:
                    W2 = np.append(W2, MM["W2"][idx[ii]:idx[ii+1]])

            W1 = W1.reshape(NN1['W1'].shape)
            W2 = W2.reshape(NN1['W2'].shape)
        else:
            if NN1.score > NN2.score:
                W1 = NN1['W1']
                W2 = NN1['W2']
            else:           
                W1 = NN2['W1']
                W2 = NN2['W2']

        return {'W1': W1,
                'W2': W2,
                'score': NN1['score'],
                'id': NN1['id']}

    def mean(seq):
        i = 0
        total = 0.0
        for x in seq:
            total += float(x.score[0][0])
            i += 1
        if i == 0:
            raise ValueError("cannot take mean of zero-length sequence")
        return total / i




"""
# First run comparison AutoBot
AU = AB.AutoBot()
AU.score = []
S = sn.snake()
S.startGame(AU)
S.quitGame()
print(AU.score)


#  Run main Loop
pop = generatePopulation(popSize)
sortedPopulation = computePerformance(pop)
with open('log.txt', 'w') as f:
    f = f.write("Hello\n")


for G in range(generations+1):
    nextPop = select(sortedPopulation, bestSample, luckyFew)
    nextGen = nextGeneration(nextPop, popSize)
    nextGen = mutateGen(nextGen)
    sortedPopulation = computePerformance(nextGen)
    MAX = np.round(maximum(sortedPopulation), 2)
    AVG = np.round(mean(sortedPopulation), 2)
    #print("Gen: {}, Avg: {}, Max: {}, Pop: {}".format(G,
    #                                                  AVG,
    #                                                  MAX,
    #                                                  len(sortedPopulation)))
    if G % 100 == 0:
        with open('log.txt', 'a') as f:
            f.write("Gen: {}, Avg: {}, Max: {}\n".format(G, AVG, MAX))
"""

if __name__ == "__main__":
    pop = Population()
    pop.generate_population()
    T = Trainer(pop)
    T.train_population()
