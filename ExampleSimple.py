import numpy as np
import Snake as sn
import Bot
from Population import Population
import configuration as cf

# We override some of the default settings for demomstration purposes.
cf.Graphics = True  # Shows display.
cf.maxIt = 100  # Makes the simulation a bit longer.
cf.speed = 20  # Makes speed slower.


"""
This example is a minimum working example showing all the components to run the game with a neural network,
defined in "Brain", which is a dictionary.
"""

brain = {'W1': np.array([[-1, 0], [0, -1], [1, 0], [0, 1]]),
         'W2': np.array([[1, 0], [0, 1]]),
         'id': 0,
         'score': 0
         }

pop = Population()
pop.pop = [brain]
sn.run_game(pop)

"""
This example shows how to run the game with the automatic Neural network robot. This is the neural network
as the above, just loaded directly in the bot, instead of first putting it in a population first.
"""

bot = Bot.AutoBot()
S = sn.snake()
S.startGame(bot)
print(bot.score)


"""
This example shows how to run the game with the fully automatic robot. This does not use any neural network
and is used to debug the game mechanics.
"""

bot = Bot.AutoBot()
S = sn.snake()
S.startGame(bot)
print(bot.score)


