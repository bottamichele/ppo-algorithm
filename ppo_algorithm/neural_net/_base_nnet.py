from abc import ABC, abstractmethod

from torch.nn import Module

class BaseActorCritic(Module, ABC):
    """Base class that can be implements Actor and Critic."""

    @abstractmethod
    def value(self, obs):
        """Compute value-state of a batch of observations.
        
        Parameter
        --------------------
        obs: torch.Tensor
            a batch of observations
            
        Return
        --------------------
        value: torch.Tensor
            value.state value of obs"""
        
    pass

    @abstractmethod
    def action_and_value(self, obs, action=None):
        """Compute actions and value-states.
        
        Parameters
        --------------------
        obs: torch.Tensor
            a batch of observations

        action: torch.Tensor, optional
            a batch of actions
            
        Returns
        --------------------
        action: torch.Tensor
            batch of actions
            
        value: torch.Tensor
            batch of value-states
            
        log_prob: torch.Tensor
            batch of log probabilities
            
        entropy: torch.Tensor
            batch of entropies"""
        
    pass