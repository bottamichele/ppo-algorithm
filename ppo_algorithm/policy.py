import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax
from torch.distributions import Categorical, MultivariateNormal

def policy_da(model, obs):
    """Choose a action from obs for a policy with disrete action space.
    
    Parameters
    --------------------
    model: PPO-like
        a neural network which gives action values and value-state as output

    obs: torch.Tensor
        an observation

    Return
    --------------------
    action: int
        action chosen by policy
    """

    if not isinstance(model, Module):
        """model must be a torch.nn.Module's subclass."""

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)

    with tc.no_grad():
        out, _ = model(obs)
        act_probs = softmax(out, dim=-1)
        act_dist = Categorical(probs=act_probs)

    return act_dist.sample().item()

def policy_ca(model, obs):
    """Choose a action from obs for a policy with continuous action space.
    
    Parameters
    --------------------
    model: PPO-like
        a neural network which gives action values and value-state as output

    obs: torch.Tensor
        an observation

    Return
    --------------------
    action: torch.Tensor
        action chosen by policy
    """

    if not isinstance(model, Module):
        """model must be a torch.nn.Module's subclass."""

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)

    with tc.no_grad():
        out, _ = model(obs)
        
        action_size = out.shape[1]
        act_dist = MultivariateNormal(loc=out, 
                                      covariance_matrix=tc.diag(tc.full(action_size, 
                                                                        0.5,
                                                                        device=obs.device)))

    return act_dist.sample().view(size=-1)