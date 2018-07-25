import NeuralNetwork as NNN
import numpy as np
import configuration as cf


class AutoBot():
    def __init__(self):
        self.id = 0
        self.score = []
        
    def move(self, lead_x, lead_y, randAppleX, randAppleY):
        if (lead_x-randAppleX) < 0:
            lead_x_change = 10
            lead_y_change = 0
        if (lead_x-randAppleX) > 0:
            lead_x_change = -10
            lead_y_change = 0
        if (lead_y-randAppleY) < 0:
            lead_x_change = 0
            lead_y_change = 10
        if (lead_y-randAppleY) > 0:
            lead_x_change = 0
            lead_y_change = -10
        return {'lead_x_change': lead_x_change,
                'lead_y_change': lead_y_change}


class NNBot():
    def __init__(self,  W1=[],  W2=[]):
        self.NN = NNN.Neural_Network()
        self.NN.W1 = W1
        self.NN.W2 = W2
        self.score = []
        self.id = id

    def load_brain(self, brain):
        self.NN.W1 = brain['W1']
        self.NN.W2 = brain['W2']
        self.id = brain['id']

    def save_brain(self):
        pass

    def move(self,  lead_x,  lead_y,  randAppleX,  randAppleY):
        # Normalise input to [0, 10]
        scaled = 100 * np.array([lead_x,  lead_y,
                                 randAppleX,  randAppleY]) / cf.display_width
        lead = self.NN.forward(scaled)
        if abs(lead[0]) >= abs(lead[1]):
            lead_x_change = np.sign(lead[0])*cf.block_size
            lead_y_change = 0
        else:
            lead_x_change = 0
            lead_y_change = np.sign(lead[1])*cf.block_size
        return {'lead_x_change': lead_x_change,
                'lead_y_change': lead_y_change}


# For refererence and checking
class AutoNN():
    def __init__(self):
        self.NN = NNN.Neural_Network()
        self.W1 = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
        self.W2 = np.array([[1, 0], [0, 1]])
        self.NN.W1 = self.W1
        self.NN.W2 = self.W2
        self.id = 0
        self.score = []

    def move(self,  lead_x,  lead_y,  randAppleX,  randAppleY):
        # Normalise input to [0, 10]
        scaled = 100 * np.array([lead_x,  lead_y,
                                 randAppleX,  randAppleY]) / cf.display_width
        lead = self.NN.forward(scaled)
        if abs(lead[0]) >= abs(lead[1]):
            lead_x_change = np.sign(lead[0])*cf.block_size
            lead_y_change = 0
        else: 
            lead_x_change = 0
            lead_y_change = np.sign(lead[1])*cf.block_size
        return {'lead_x_change': lead_x_change,
                'lead_y_change': lead_y_change}


#def move(self, lead_x, lead_y, randAppleX, randAppleY):
#        return self.NN.move(self, lead_x, lead_y, randAppleX, randAppleY)
