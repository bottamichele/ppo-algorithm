import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax, mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

# ========================================
# =============== POLICIES ===============
# ========================================

def train_policy_da(model, obs):
    """Choose a action from obs for a policy with disrete action space.
    
    Parameters
    --------------------
    model: PPO-like
        a neural network which gives action values and value-state as output

    obs: torch.Tensor
        an observation

    Returns
    --------------------
    action: torch.Tensor
        action chosen by policy

    value: torch.Tensor
        value-state of obs

    distribution: torch.distributions.distribution.Distribution
        distribution of action
    """

    if not isinstance(model, Module):
        """model must be a torch.nn.Module's subclass."""

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)

    out, value = model(obs)
    act_probs = softmax(out, dim=-1)
    act_dist = Categorical(probs=act_probs)

    return act_dist.sample(), value, act_dist

def train_policy_ca(model, obs):
    """Choose a action from obs for a policy with continuous action space.
    
    Parameters
    --------------------
    model: PPO-like
        a neural network which gives action values and value-state as output

    obs: torch.Tensor
        an observation

    Returns
    --------------------
    action: torch.Tensor
        action chosen by policy

    value: torch.Tensor
        value-state of obs

    distribution: torch.distributions.distribution.Distribution
        distribution of action"""

    if not isinstance(model, Module):
        """model must be a torch.nn.Module's subclass."""

    if obs.ndim == 1:
        obs = obs.unsqueeze(0)

    out, value = model(obs)
        
    action_size = out.shape[1]
    if action_size == 1:
        act_dist = Normal(loc=out.reshape(-1), scale=0.5)
    else:
        act_dist = MultivariateNormal(loc=out, 
                                      covariance_matrix=tc.diag(tc.full(action_size, 
                                                                        0.5,
                                                                        device=obs.device)))

    return act_dist.sample(), value, act_dist


# ========================================
# ============== TRAIN STEP ==============
# ========================================

def ppo_train_step(model, policy_fn, buffer, optimizer, norm_adv=True, n_epochs=6, batch_size=64, max_grad_norm=0.5, clip_range=0.2, kl_target=None, value_coeff=0.5, entr_coeff=0.0):
    """Do one train step of PPO.
    
    Parameters
    --------------------
    model: PPO-like
        a neural network which gives action values and value-state as output

    policy_fn: callable
        a policy which gives actions, value-states and distrutions

    buffer: ppo_algorithm.Buffer
        a buffer which store all step infos before to train model. It will be reseted at the end of train step
    
    optimizer: torch.optim.Optimizer
        an optimizer

    norm_adv: bool, optional
        whether or not to normalize advantage

    n_epochs: int, optional
        number of epochs that model will be trained

    batch_size: int, optional
        minibatch size

    max_grad_norm: float, optional
        maximum gradient norm

    clip_range: float, optional
        clip range for surrogate loss

    kl_target: float, optional
        KL target threshold

    value_coeff: float, optional
        coefficient for value loss
        
    entr_coeff: float, optional
        entropy coefficient
    """

    #Build a dataset which contains observations, actions and so on.
    dataset = TensorDataset(buffer.observations.reshape((-1,) + tuple(buffer.observations.shape[2:])), 
                            buffer.actions.reshape((-1,) + tuple(buffer.actions.shape[2:])), 
                            buffer.log_probs.reshape(-1),
                            buffer.advantages.reshape(-1),
                            buffer.returns.reshape(-1))
    
    for _ in range(n_epochs):
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        for (obs_b, action_b, log_prob_b, adv_b, return_b) in dataloader:
            _, new_value_b, new_act_dist_b = policy_fn(model, obs_b)

            #Compute new probabilities.
            new_log_prob_b = new_act_dist_b.log_prob(action_b)

            #Normalize advantages.
            if norm_adv:
                adv_b = (adv_b - adv_b.mean()) / (adv_b + 10**-8)

            #Compute ratios.
            log_ratio_b = new_log_prob_b - log_prob_b
            ratio_b = tc.exp(log_ratio_b)

            #Compute surrogate loss
            surr_loss_1 = ratio_b * adv_b
            surr_loss_2 = tc.clamp(ratio_b, 1 - clip_range, 1 + clip_range) * adv_b
            surr_loss = tc.min(surr_loss_1, surr_loss_2).mean()

            #Compute value loss.
            value_loss = mse_loss(new_value_b.reshape(-1), return_b)

            #Compute entropy.
            entropy = new_act_dist_b.entropy().mean()

            #Compute total loss.
            loss = -(surr_loss - value_coeff * value_loss + entr_coeff * entropy)

            #Compute gradient discent.
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            #Compute approximante KL divergence (for more info http://joschu.net/blog/kl-approx.html).
            kl_value = ((ratio_b - 1) - log_ratio_b).mean()

        if kl_target is not None and kl_value > kl_target:
            break

    buffer.reset()