import torch as tc

from torch.nn import Module
from torch.nn.functional import softmax, mse_loss
from torch.optim import Adam
from torch.distributions import Categorical, MultivariateNormal

from ._action_space import ActionSpace
from ._trajectory import Trajectory


# ========================================
# ============ CLASS PPOAgent ============
# ========================================

class PPOAgent:
    """A Proximal Policy Optimization's agent."""

    def __init__(self, actor, action_space):
        """Create a new PPO's agent.

        Parameters
        --------------------
        actor: torch.nn.Module
            a neural network which takes observations as input and gives action values as output

        action_space: ActionSpace
            action space type
        """
        
        #Check the parameters.
        if not isinstance(actor, Module):
            raise TypeError("actor must be a torch.nn.Module's subclass.")
        
        #Variables.
        self._actor = actor
        self._action_space_type = action_space

    def choose_action(self, obs):
        """Choose an action to perform from an observation.
        
        Parameter
        --------------------
        obs: torch.Tensor
            an observation
            
        Return
        --------------------
        action: int | torch.Tensor
            action chosen"""
        
        obs = obs.unsqueeze(0)

        #An action is choosen
        with tc.no_grad():
            action_values = self._actor(obs)

        if self._action_space_type == ActionSpace.DISCRETE:
            action_probs = softmax(action_values, dim=-1)
            action_distr = Categorical(probs=action_probs)
            return action_distr.sample().detach().cpu().item()
        elif self._action_space_type == ActionSpace.CONTINUOUS:
            action_size = action_values.shape[1]
            conv_matrix = tc.diag(tc.full((action_size,), 0.5, device=action_values.device))
            
            action_distr = MultivariateNormal(action_values, conv_matrix)
            return action_distr.sample().squeeze()


# ========================================
# ========== CLASS PPOTrainAgent =========
# ========================================

class PPOTrainAgent:
    """A Proximal Policy Optimization's agent which is trained."""

    def __init__(self, actor, critic, action_space_type, lr=10**-3, gamma=0.95, batch_size=64, n_epochs=8, clip_range=0.2, entropy_coeff=0.0, norm_adv=True, device=tc.device("cpu")):
        """Create new PPO's agent which will get trained.
        
        Parameters
        --------------------
        actor: torch.nn.Module
            a neural network which takes observations as input and gives action values as output
        
        critic: torch.nn.Module
            a neural network which takes observations as input and gives state-value values as output
            
        action_space_type: ppo_algorithm.ActionSpace
            action space type

        lr: float, optional
            learning rate

        gamma: float, optional
            discount factor

        batch_size: int, optional
            batch size

        n_epochs: int, optional
            number of epochs per episode to optimize the surrogate loss

        clip_range: float, optional
            clip value used into the surrogate loss

        entropy_coeff: float, optional
            entropy coefficient

        norm_adv: bool, optional
            whether or not to normalize the advantage
            
        device: torch.device, optional
            aaaa"""
        
        #Check the parameters.
        if not isinstance(actor, Module):
            raise TypeError("actor must be a torch.nn.Module's subclass.")
        
        if not isinstance(critic, Module):
            raise TypeError("critic must be a torch.nn.Module's subclass.")

        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("gamma must be a value between 0.0 and 1.0")
        
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        
        if n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer.")
        
        self.actor = actor.to(device=device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic = critic.to(device=device)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.action_space_type = action_space_type
        self.trajectory = Trajectory()
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.entropy_coeff = entropy_coeff
        self.norm_adv = norm_adv
        self.device = device

    def choose_action(self, obs):
        """Choose action to perform.
        
        Parameter
        --------------------
        obs: torch.Tensor
            an observation
            
        Return
        --------------------
        action: torch.Tensor
            an action chosen
            
        distribution: torch.distributions.distribution.Distribution
            a distribution"""
        
        action, dist = self._get_action(obs.unsqueeze(0), True)

        if self.action_space_type == ActionSpace.CONTINUOUS:
            action = action.squeeze()

        return action, dist
        
    def _get_action(self, obs, no_grad):
        """Get action.
        
        Parameter
        --------------------
        obs: torch.Tensor
            a batch of observations

        no_grad: bool
            True if no gradient required, False otherwise
            
        Return
        --------------------
        action: torch.Tensor
            a batch of actions chosen
            
        distribution: torch.distributions.distribution.Distribution
            a distribution"""
        
        if no_grad:
            with tc.no_grad():
                action_values = self.actor(obs)
        else:
            action_values = self.actor(obs)

        if self.action_space_type == ActionSpace.DISCRETE:
            action_probs = softmax(action_values, dim=-1)
            action_distr = Categorical(probs=action_probs)

            return action_distr.sample(), action_distr
        elif self.action_space_type == ActionSpace.CONTINUOUS:
            action_size = action_values.shape[1]
            conv_matrix = tc.diag(tc.full((action_size,), 0.5, device=self.device))
            
            action_distr = MultivariateNormal(action_values, conv_matrix)
            return action_distr.sample(), action_distr
        
    def train(self):
        """Do train step."""
        
        self.trajectory.compute_returns(self.gamma)

        #Get trajectory's state.
        distributions   = self.trajectory.action_distributions
        log_probs       = tc.cat(self.trajectory.log_probs).to(device=self.device)
        returns         = tc.Tensor(self.trajectory.returns).to(device=self.device)

        if self.trajectory.observations[0].ndim == 1 and self.trajectory.observations[0].shape[0] == 1:
            observations = tc.cat(self.trajectory.observations)
        else:
            observations = tc.stack(self.trajectory.observations)
        observations = observations.to(device=self.device)

        assert self.trajectory.actions[0].ndim == 1
        if self.trajectory.actions[0].shape[0] == 1:
            actions = tc.cat(self.trajectory.actions)
        else:
            actions = tc.stack(self.trajectory.actions)
        actions = actions.to(device=self.device)

        #Compute advantage.
        with tc.no_grad():
            V = self.critic(observations).squeeze(-1)
        adv = returns - V

        if self.norm_adv:
            adv = (adv - adv.mean()) / (adv.std() + 10**-8)

        #Compute train step.
        length = len(distributions)
        batch_idxs = tc.arange(0, length, self.batch_size, dtype=tc.int32, device=self.device)
        batch_idxs = tc.cat((batch_idxs, tc.Tensor([length]).to(dtype=tc.int32, device=self.device)))

        for _ in range(self.n_epochs):
            for i in range(batch_idxs.shape[0]-1):
                #Sample minibatch.
                start = batch_idxs[i].item()
                end = batch_idxs[i + 1].item()

                obs_batch           = observations[start:end]
                action_batch        = actions[start:end]
                distribution_batch  = distributions[start:end]
                log_prob_batch      = log_probs[start:end]
                return_batch        = returns[start:end]
                adv_batch           = adv[start:end]

                #Compute new 
                _, new_distribution_batch = self._get_action(obs_batch, False)
                new_log_prob_batch = new_distribution_batch.log_prob(action_batch)
                V_batch = self.critic(obs_batch).squeeze(-1)

                #Compute ratios.
                ratio_batch = tc.exp(new_log_prob_batch - log_prob_batch)

                #Compute surrogate loss.
                surr_loss_1 = ratio_batch * adv_batch
                surr_loss_2 = tc.clamp(ratio_batch, 1 - self.clip_range, 1 + self.clip_range) * adv_batch
                surr_loss = tc.min(surr_loss_1, surr_loss_2).mean()

                #Compute value-state loss.
                value_loss = mse_loss(V_batch, return_batch)

                #Compute entropy bonus.
                entropy_bonus = new_distribution_batch.entropy().mean()

                #Compute actor loss.
                actor_loss = -(surr_loss + self.entropy_coeff * entropy_bonus)

                #Backward for actor's neural network.
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                #Backward for critic's neural network.
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.trajectory.clear()