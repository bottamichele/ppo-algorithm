import torch as tc

from torch.nn.functional import mse_loss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from ppo_algorithm.neural_net import BaseActorCritic

def ppo_train_step(model, rollout, optimizer, norm_adv=True, n_epochs=6, batch_size=64, max_grad_norm=0.5, clip_range=0.2, kl_target=None, value_coeff=0.5, entr_coeff=0.0):
    """Perform one train step of PPO.
    
    Parameters
    --------------------
    model: ppo_algorithm.neural_net.BaseActorCritic
        a PPO neural network

    rollout: ppo_algorithm.Rollout
        a rollout which stores timesteps. It will be reseted at the end of train step
    
    optimizer: torch.optim.Optimizer
        an optimizer

    norm_adv: bool, optional
        whether or not to normalize advantage

    n_epochs: int, optional
        number of epochs that model will be trained

    batch_size: int, optional
        minibatch size

    max_grad_norm: float, optional
        maximum gradient norm. If it is None, max gradient norm is not used.

    clip_range: float, optional
        clip range for surrogate loss

    kl_target: float, optional
        KL target threshold. If it is None, KL divergence is not used.

    value_coeff: float, optional
        coefficient for value loss
        
    entr_coeff: float, optional
        entropy coefficient

    Return
    --------------------
    losses: dict
        a dictionary which contains each kind of loss computed at every epoch
    """

    assert isinstance(model, BaseActorCritic), "model must be a subclass of BaseActorCritic"

    surrogate_losses = []
    value_losses = []
    entropy_losses = []
    total_losses = []

    #Build a dataset which contains observations, actions and so on.
    dataset = TensorDataset(rollout.observations.reshape((-1,) + tuple(rollout.observations.shape[2:])), 
                            rollout.actions.reshape((-1,) + tuple(rollout.actions.shape[2:])), 
                            rollout.log_probs.reshape(-1),
                            rollout.advantages.reshape(-1),
                            rollout.returns.reshape(-1))
    
    for _ in range(n_epochs):
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        for (obs_b, action_b, log_prob_b, adv_b, return_b) in dataloader:
            _, new_value_b, new_log_prob_b, entropy_b = model.action_and_value(obs_b, action_b)

            #Normalize advantages.
            if norm_adv and len(adv_b) > 1:
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 10**-8)

            #Compute ratios.
            log_ratio_b = new_log_prob_b - log_prob_b
            ratio_b = tc.exp(log_ratio_b)

            #Compute surrogate loss
            surr_loss_1 = ratio_b * adv_b
            surr_loss_2 = tc.clamp(ratio_b, 1 - clip_range, 1 + clip_range) * adv_b
            surr_loss = tc.min(surr_loss_1, surr_loss_2).mean()
            surrogate_losses.append(surr_loss)

            #Compute value loss.
            value_loss = mse_loss(new_value_b.reshape(-1), return_b)
            value_losses.append(value_loss)

            #Compute entropy.
            entropy = entropy_b.mean()
            entropy_losses.append(entropy)

            #Compute total loss.
            loss = -(surr_loss - value_coeff * value_loss + entr_coeff * entropy)
            total_losses.append(loss)

            #Compute gradient discent.
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None:
                clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            #Compute approximante KL divergence (for more info http://joschu.net/blog/kl-approx.html).
            with tc.no_grad():
                kl_value = ((ratio_b - 1) - log_ratio_b).mean()

            if kl_target is not None and kl_value > kl_target:
                break

    rollout.clear()

    return {"surrogate": surrogate_losses, "value": value_losses, "entropy": entropy_losses, "total": total_losses}