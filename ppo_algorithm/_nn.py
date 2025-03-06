from torch.nn import Module, Sequential, Linear, ReLU
from torch.optim import Adam

class ActorCritic(Module):
    """A fully connected neural network of Actor and Critic."""

    def __init__(self, obs_size, action_size, fc_dim, fc_num, actor_fc_num, critic_fc_num):
        """Create new fully connected neural network of Actor and Critic.

        Parameters:
        --------------------
        obs_size: int
            observation size

        action_size: int
            action size

        fc_dim: int
            number of nodes for each hidden layer 

        fc_num: int
            number of hidden layers for the backbone

        actor_fc_num: int
            number of Actor's hidden layers

        critic_fc_num: int
            number of Critic's hidden layers
        """
        
        super().__init__()

        #Check the parameters.
        if obs_size <= 0:
            raise ValueError("obs_size must be an integer positive.")
        
        if action_size <= 0:
            raise ValueError("action_size must be an integer positive.")

        if fc_dim <= 0:
            raise ValueError("fc_dim must be an integer positive.")
        
        if fc_num <= 0:
            raise ValueError("fc_num must be an integer positive.")
        
        if actor_fc_num <= 0:
            raise ValueError("actor_fc_num must be an integer positive.")

        if critic_fc_num <= 0:
            raise ValueError("critic_fc_num must be an integer positive.")

        #Hidden layers of the backbone.
        self._backbone = Sequential()

        for i in range(fc_num):
            if i == 0:
                #First hidden layer.
                self._backbone.append(Linear(obs_size, fc_dim))
            else:
                #i-th hidden layer.
                self._backbone.append(Linear(fc_dim, fc_dim))
            self._backbone.append(ReLU())

        #Actor's hidden layers.
        self._actor = Sequential()

        for i in range(actor_fc_num):
            if i == actor_fc_num - 1:
                #Output layer.
                self._actor.append(Linear(fc_dim, action_size))
            else:
                #i-th hidden layer.
                self._actor.append(Linear(fc_dim, fc_dim))
                self._actor.append(ReLU())

        #Critic's hidden layers.
        self._critic = Sequential()

        for i in range(critic_fc_num):
            if i == critic_fc_num - 1:
                #Output layer.
                self._critic.append(Linear(fc_dim, 1))
            else:
                #i-th hidden layer.
                self._critic.append(Linear(fc_dim, fc_dim))
                self._critic.append(ReLU())

    def forward(self, x):
        """Compute x.
        
        Parameter
        --------------------
        x: torch.Tensor
            a tensor
            
        Return
        --------------------
        act_value: torch.Tensor
            action values
            
        value_state: torch.Tensor
            value-state value"""
        
        x = self._backbone(x)
        return self._actor(x), self._critic(x)