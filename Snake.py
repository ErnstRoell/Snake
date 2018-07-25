import random
import pygame
import numpy as np
import configuration as cf
import Bot
import pprint as pp
from Population import Population


class snake:
    def __init__(self, Graphics=False):
        self.rounds = cf.rounds
        # Make game display if nessecary
        if cf.Graphics:
            self.clock = pygame.time.Clock()
            pygame.init()
            self.gameDisplay = pygame.display.set_mode((cf.display_width,
                                                        cf.display_height))
            pygame.display.set_caption('Slither')

    def drawSnake(self, snakeList, block_size):
        for XnY in snakeList:
            pygame.draw.rect(self.gameDisplay,
                             cf.green,
                             [XnY[0],
                              XnY[1],
                              cf.block_size,
                              cf.block_size])
    
    def startGame(self, Bot):
        score = []
        # Initialise robot and neural network
        gameExit = False
        gameOver = False
        killed = False
        maxIt = False

        # Start position + snakeStart
        lead_x = cf.display_width/2
        lead_y = cf.display_height/2
        snakeList = []
        snakeLength = 1

        # Spawn Apple
        apple_x = round(random.randrange(0, cf.display_width
                                         - cf.apple_size)/10)*10
        apple_y = round(random.randrange(0, cf.display_height
                                         - cf.apple_size)/10)*10

        # Game ticker to stop after some time
        ticker = 1
        # NOTE: HUGE ERROR IN DETERMINING SCORE!!!!
        # Game Loop
        while not gameExit:
            if gameOver:
                if cf.Graphics and not gameExit:
                    self.gameDisplay.fill(cf.white)
                    pygame.display.update()

                self.rounds -= 1
                if self.rounds == 0:
                    self.rounds = cf.rounds
                    gameExit = True
                    gameOver = False
                    if killed:
                        score.append(snakeLength-1)
                        killed = False
                    else:
                        ticker = 1
                        maxIt = False
                        score.append(
                            snakeLength-1
                            + np.exp(-0.01*np.sqrt((lead_x-apple_x)**2
                                                   + (lead_y-apple_y)**2)))
                    Bot.score.append(score)
                    score = []
                else:
                    # Reset Game!!!!
                    gameOver = False
                    if killed:
                        score.append(snakeLength-1)
                    if maxIt:
                        score.append(
                            snakeLength - 1
                            + np.exp(-0.01*np.sqrt((lead_x-apple_x)**2
                                                   + (lead_y-apple_y)**2)))
                    # Initialise robot and neural network
                    gameExit = False
                    gameOver = False
                    killed = False
                    maxIt = False

                    # Start position + snakeStart
                    lead_x = cf.display_width/2
                    lead_y = cf.display_height/2
                    snakeList = []
                    snakeLength = 1

                    # Spawn Apple
                    apple_x = round(random.randrange(0, cf.display_width
                                                     - cf.apple_size)/10)*10
                    apple_y = round(random.randrange(0, cf.display_height
                                                     - cf.apple_size)/10)*10

                    # Game ticker to stop after some time
                    ticker = 1

            ticker += 1
            if ticker == cf.maxIt:
                gameOver = True
                maxIt = True

            change = Bot.move(lead_x, lead_y, apple_x, apple_y)

            lead_x += change['lead_x_change']
            lead_y += change['lead_y_change']

            snakeHead = []
            snakeHead.append(lead_x)
            snakeHead.append(lead_y)
            snakeList.append(snakeHead)

            if lead_x > cf.display_width:
                # lead_x=0
                gameOver = True
                killed = True

            elif lead_x < 0:
                # lead_x = cf.display_width
                gameOver = True
                killed = True

            if lead_y > cf.display_height:
                # lead_y=0
                gameOver = True
                self.killed = True
            elif lead_y < 0:
                # lead_y = cf.display_height
                gameOver = True
                self.killed = True

            if len(snakeList) > snakeLength:
                del snakeList[0]

            if cf.Graphics:
                self.gameDisplay.fill(cf.white)
                pygame.draw.rect(self.gameDisplay,
                                 cf.red,
                                 [apple_x,
                                  apple_y,
                                  cf.apple_size,
                                  cf.apple_size])
                self.drawSnake(snakeList, cf.block_size)
                self.clock.tick(cf.speed)
                pygame.display.update()

            if (apple_x - cf.apple_size < lead_x < apple_x+cf.apple_size and
                    apple_y - cf.apple_size < lead_y < apple_y+cf.apple_size):
                # History
                # print("nom nom nom")
                snakeLength += 1
                apple_x = round(
                    random.randrange(0,
                                     cf.display_width-cf.apple_size)
                    / cf.apple_size)*cf.apple_size
                apple_y = round(
                    random.randrange(0,
                                     cf.display_height-cf.apple_size)
                    / cf.apple_size)*cf.apple_size

    def quitGame(self):
        pygame.quit()


def run_game(population):
    S = snake()
    bot = Bot.NNBot()
    for individual in population.pop:
        bot.load_brain(individual)
        S.startGame(bot)
        individual['score'] = bot.score[0]
        bot.score = []
        S.quitGame()
    population.sort_population()
    pp.pprint(population.pop)


###############
# TESTS
###############

if __name__ == "__main__":
    brain = {'W1': np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]),
             'W2': np.array([[1, 0], [0, 1]]),
             'id': 0,
             'score': 0
             }

    filename = 'C:/Users/gebruiker/documents/programming/python/NN/NN.json'
    pop = Population()
    pop.load_population(filename)
    run_game(pop)
