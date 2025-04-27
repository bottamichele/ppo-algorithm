import os
import pickle
import torch as tc

from torch.optim import Adam

from ppo_algorithm import Rollout
from ppo_algorithm.neural_net import BaseActorCritic
from ppo_algorithm.training import ppo_train_step

class PPOAgent:
    """A PPO training agent."""

    def __init__(self, model, rollout, lr, batch_size, n_epochs, device, gamma=0.99, gae_coeff=0.95, norm_adv=True, kl_target=None, max_grad_norm=0.5, clip_range=0.2, value_coeff=0.5, entr_coeff=0.0):
        """Create new PPO training agent.
        
        Parameters
        --------------------
        model: ppo_algorithm.neural_net.BaseActorCritic
            a neural network
            
        rollout: ppo_algorithm.Rollout
            a rollout
            
        lr: float
            learning rate
            
        batch_size: int
            minibatch size
            
        n_epochs: int
            number of epochs per each train step

        device: torch.device:
            device traning is run on

        gamma: float, optional
            discount factor

        gae_coeff: float, optional
            GAE lambda
            
        norm_adv: bool, optional
            whether or not to normalize advantages
            
        kl_target: float, optional
            KL target threshold. If it is None, KL divergence is not used.
            
        max_grad_norm: float, optional
            maximum gradient norm. If it is None, max gradient norm is not used.
            
        clip_range: float, optional
            clip range for surrogate loss

        value_coeff: float, optional
            value loss coefficient
            
        entr_coeff: float, optional
            entropy coefficient"""
        
        assert isinstance(model, BaseActorCritic), "model must be a subclass of BaseActorCritic."
        assert isinstance(rollout, Rollout)
        assert rollout.device == device, f"Expexted rollout.device is {device}, but it was {rollout.device}"

        self.rollout = rollout
        self.losses = {}

        #Hyperparameters.
        self.gamma = gamma
        self.gae_coeff = gae_coeff
        self.norm_adv = norm_adv
        self.kl_target = kl_target
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.max_grad_norm = max_grad_norm
        self.clip_range = clip_range
        self.value_coeff = value_coeff
        self.entropy_coeff = entr_coeff

        #PPO's model.
        self.model = model.to(device=device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.device = device

    def choose_action(self, obs):
        """Choose an action from a observation of each enviroment.
        
        Parameters
        --------------------
        obs: torch.Tensor
            observation of each enviroment
            
        Returns
        --------------------
        action: torch.Tensor
            action choosen for each enviroment
            
        value: torch.Tensor
            value.state of each observation
            
        log_prob: torch.Tensor
            log probability of each action chosen"""
        
        assert obs.shape == self.rollout.observations.shape[1:], f"Expected obs.shape is {self.rollout.observations.shape[1:]}, but it was {obs.shape}"
        assert obs.device == self.device, f"Expected obs.device is {self.device}, but it was {obs.device}"

        with tc.no_grad():
            action, value, logprob, _ = self.model.action_and_value(obs)

        return action, value, logprob
    
    def remember(self, obs, action, log_prob, reward, done, value):
        """Store one timestep into rollout.
        
        Parameters
        --------------------
        obs: torch.Tensor
            observation of each enviroment
            
        action: torch.Tensor
            action of each enviroment
            
        log_prob: torch.Tensor
            log probability action of each enviroment
            
        reward: torch.Tensor
            reward of each enviroment
            
        done: torch.Tensor
            obs's each enviroment is a terminal state or not
            
        value: torch.Tensor
            obs's value-state of each enviroment"""
        
        assert obs.device == self.device, f"Expected obs.device is {self.device}, but it was {obs.device}"
        assert action.device == self.device, f"Expected action.device is {self.device}, but it was {action.device}"
        assert log_prob.device == self.device, f"Expected log_prob.device is {self.device}, but it was {log_prob.device}"
        assert reward.device == self.device, f"Expected reward.device is {self.device}, but it was {reward.device}"
        assert done.device == self.device, f"Expected done.device is {self.device}, but it was {done.device}"
        assert value.device == self.device, f"Expected value.device is {self.device}, but it was {value.device}"

        self.rollout.store(obs, action, log_prob, reward, done, value)

    def train(self, last_obs, last_done):
        """Do a train step.
        
        Parameter
        --------------------
        last_obs: torch.Tensor
            last observation of each enviroment before to perform one train step"""
        
        assert last_obs.shape == self.rollout.observations.shape[1:], f"Expected last_obs.shape is {self.rollout.observations.shape[1:]}, but it was {last_obs.shape}"
        assert last_done.shape == self.rollout.dones.shape[1:], f"Expected last_done.shape is {self.rollout.dones.shape[1:]}, but it was {last_done.shape}"
        assert last_obs.device == self.device, f"Expected last_obs.device is {self.device}, but it was {last_obs.device}"
        assert last_done.device == self.device, f"Expected last_done.device is {self.device}, but it was {last_done.device}"

        #Compute advantages and returns.
        with tc.no_grad():
            last_value = self.model.value(last_obs)
            self.rollout.compute_advantages_and_returns(last_value.reshape(-1), last_done, self.gae_coeff, self.gamma)

        #Train step.
        losses = ppo_train_step(self.model, 
                                self.rollout, 
                                self.optimizer, 
                                self.norm_adv, 
                                self.n_epochs, 
                                self.batch_size, 
                                self.max_grad_norm, 
                                self.clip_range, 
                                self.kl_target, 
                                self.value_coeff, 
                                self.entropy_coeff)
        
        #Store all losses computed from last train step.
        for k, v in losses.items():
            l = self.losses.get(k, [])
            l.extend(v)
            self.losses[k] = l

    def save_session(self, path):
        """Save current session of this agent on disk.
        
        Parameter
        --------------------
        path: str
            path where current session of this agent is saved on"""
        
        with open(os.path.join(path, "ppo_agent_session.pkl"), "wb") as session_file:
            pickle.dump(self, session_file)

    def load_session(path):
        """Load a session of PPO agent from disk.
        
        Parameter
        --------------------
        path where a session of PPO agent is on
        
        Return
        --------------------
        agent: PPOAgent
            a PPO training agent"""

        with open(os.path.join(path, "ppo_agent_session.pkl"), "rb") as session_file:
            agent = pickle.load(session_file)

        return agent

    def save_model(self, path):
        """Save model on disk.
        
        Parameter
        --------------------
        path: str
            path where model is saved on"""
        
        tc.save(self.model.state_dict(), os.path.join(path, "model.pth"))