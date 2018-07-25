import numpy as np
import configuration as cf
from json_tricks import dump, load


class Population():

    def __init__(self):
        self.ids = range(cf.popSize)
        self.pop = []

    def generate_individual(self,  Id):
        W1 = np.random.normal(0, 1, (cf.inputLayerSize, cf.hiddenLayerSize))
        W2 = np.random.normal(0, 1, (cf.hiddenLayerSize,  cf.outputLayerSize))
        return {'W1': W1,
                'W2': W2,
                'id': Id,
                'score': 0
                }

    def generate_population(self):
        for Id in self.ids:
            self.pop.append(self.generate_individual(Id))

    def sort_population(self):
        self.pop.sort(key=lambda individual: individual['score'], reverse=True)

    def save_population(self, filename):
        self.sort_population()
        with open(filename, 'w') as f:
            dump(self.pop, f, indent=2)

    def load_population(self, filename):
        with open(filename, 'r') as f:
            self.pop = load(f, preserve_order=False)


if __name__ == "__main__":
    # Test wether the sorting goes well / correct.
    filename = 'C:/Users/gebruiker/documents/programming/python/NN/NN.json'
    pop = Population()
    pop.generate_population()
    pop.pop[3]['score']=5.22276373
    pop.save_population(filename)
    pop.load_population(filename)
