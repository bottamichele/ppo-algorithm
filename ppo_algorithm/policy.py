import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax
from torch.distributions import Categorical, Normal, MultivariateNormal

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

    obs = obs.unsqueeze(0)

    with tc.no_grad():
        out, _ = model(obs)
        
        action_size = out.shape[1]
        if action_size == 1:
            act_dist = Normal(loc=out.reshape(-1), scale=0.5)
        else:
            act_dist = MultivariateNormal(loc=out, 
                                          covariance_matrix=tc.diag(tc.full(size=(action_size,), 
                                                                            fill_value=0.5,
                                                                            device=obs.device)))

    return act_dist.sample().view(-1)