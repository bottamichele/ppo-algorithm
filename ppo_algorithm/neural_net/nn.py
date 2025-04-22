import torch as tc

from torch.nn import Module, Sequential, Linear, ReLU, Parameter
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal

from ._base_nnet import BaseActorCritic

__all__ = ["NNActorCriticDiscrete", "NNActorCriticContinuous"]

# ========================================
# ============== BASE CLASS ==============
# ========================================

class _BaseNNActorCritic(BaseActorCritic):
    """Base class of a fully connected neural network of Actor and Critic."""

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
            number of hidden layers for shared layers

        actor_fc_num: int
            number of Actor's hidden layers

        critic_fc_num: int
            number of Critic's hidden layers
        """
        
        Module.__init__(self)

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

    def value(self, obs):
        return self._critic(self._backbone(obs))

# ========================================
# =========== DISCRETE VERSION ===========
# ========================================

class NNActorCriticDiscrete(_BaseNNActorCritic):
    """A fully connected neural network of Actor and Critic for enviroments with discrete actions."""

    def action_and_value(self, obs, action=None):
        #Compute action probs and values.
        h = self._backbone(obs)
        action_probs = softmax(self._actor(h), dim=-1)
        value = self._critic(h)

        #An action is chosen if not specified.
        action_dist = Categorical(probs=action_probs)
        if action is None:
            action = action_dist.sample()

        return action, value, action_dist.log_prob(action), action_dist.entropy()
    
# ========================================
# ========== CONTINUOUS VERSION ==========
# ========================================

class NNActorCriticContinuous(_BaseNNActorCritic):
    """A fully connected neural network of Actor and Critic for enviroments with continuous actions."""

    def __init__(self, obs_size, action_size, fc_dim, fc_num, actor_fc_num, critic_fc_num):
        super().__init__(obs_size, action_size, fc_dim, fc_num, actor_fc_num, critic_fc_num)
        self._action_logstd = Parameter(tc.zeros(size=(1, action_size)), requires_grad=True)

    def action_and_value(self, obs, action=None):
        #Compute action probs and values.
        h = self._backbone(obs)
        action_mean = self._actor(h)
        value = self._critic(h)

        #An action is chosen if not specified.
        action_logstd = self._action_logstd.expand_as(action_mean)
        action_dist = Normal(loc=action_mean, scale=action_logstd.exp())
        if action is None:
            action = action_dist.sample()

        return action, value, action_dist.log_prob(action).sum(1), action_dist.entropy().sum(1)