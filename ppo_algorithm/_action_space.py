from enum import Enum

class ActionSpace(Enum):
    """An action space type of an enviroment."""

    DISCRETE = 0        #Discrete action space.
    CONTINUOUS = 1      #Continuous action space.