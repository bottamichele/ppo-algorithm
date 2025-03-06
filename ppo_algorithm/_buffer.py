import torch as tc

class Buffer:
    """A buffer which stores all timestep infos needed for training before PPO does a train step."""

    def __init__(self, n_timesteps, n_actors, obs_size, action_shape, obs_dtype=tc.float32, act_dtype=tc.float32, device=tc.device("cpu")):
        """Create new buffer.
        
        Parameters
        --------------------
        n_timesteps: int
            max number of timesteps

        n_actors: int
            number of actor employed

        obs_size: int | tuple
            observation size

        action_shape: int
            action's vector shape

        obs_dtype: torch.dtype
            observation's data type

        act_dtype: torch.dtype
            action's data type
            
        device: torch.device
            device which all infos are stored on"""
        
        assert action_shape >= 1

        if isinstance(obs_size, int):
            self.observations = tc.zeros((n_timesteps, n_actors, obs_size), dtype=obs_dtype, device=device)
        else:
            self.observations = tc.zeros((n_timesteps, n_actors, *obs_size), dtype=obs_dtype, device=device)

        if action_shape == 1:
            self.actions = tc.zeros((n_timesteps, n_actors), dtype=act_dtype, device=device)
        else:
            self.actions = tc.zeros((n_timesteps, n_actors, action_shape), dtype=act_dtype, device=device)

        self.log_probs  = tc.zeros((n_timesteps, n_actors), dtype=tc.float32, device=device)
        self.rewards    = tc.zeros((n_timesteps, n_actors), dtype=tc.float32, device=device)
        self.dones      = tc.zeros((n_timesteps, n_actors), dtype=tc.int32, device=device)
        self.values     = tc.zeros((n_timesteps, n_actors), dtype=tc.float32, device=device)
        self.advantages = tc.zeros((n_timesteps, n_actors), dtype=tc.float32, device=device)
        self.returns    = tc.zeros((n_timesteps, n_actors), dtype=tc.float32, device=device)

        self._n_timesteps = n_timesteps
        self._curr_idx = 0

    def reset(self):
        """Reset the buffer."""

        self.observations.fill_(0)
        self.actions.fill_(0)
        self.log_probs.fill_(0)
        self.rewards.fill_(0)
        self.dones.fill_(0)
        self.values.fill_(0)
        self.advantages.fill_(0)
        self.returns.fill_(0)

        self._curr_idx = 0

    def store(self, obs, action, log_prob, reward, done, value):
        """Store current timestep infos.
        
        Parameters
        --------------------
        obs: torch.Tensor
            observation of each actor
            
        action: torch.Tensor
            action of each actor
            
        log_prob: torch.Tensor
            log probability of each actor
            
        reward: torch.Tensor
            reward of each actor
            
        done: torch.Tensor
            whether or not obsevartion of each actor is a terminal state
            
        value: torch.Tensor
            value-state of each actor"""
        
        assert obs.shape == self.observations.shape[1:], f"Expected shape = {self.observations.shape[1:]}. obs.shape was {obs.shape}"
        assert action.shape == self.actions.shape[1:], f"Expected shape = {self.actions.shape[1:]}. action.shape was {action.shape}"
        assert log_prob.shape == self.log_probs.shape[1:], f"Expected shape = {self.log_probs.shape[1:]}. log_prob.shape was {log_prob.shape}"
        assert reward.shape == self.rewards.shape[1:], f"Expected shape = {self.rewards.shape[1:]}. reward.shape was {reward.shape}"
        assert done.shape == self.dones.shape[1:], f"Expected shape = {self.dones.shape[1:]}. done.shape was {done.shape}"
        assert value.shape == self.values.shape[1:], f"Expected shape = {self.values.shape[1:]}. value.shape was {value.shape}"

        self.observations[self._curr_idx] = obs
        self.actions[self._curr_idx]      = action
        self.log_probs[self._curr_idx]    = log_prob
        self.rewards[self._curr_idx]      = reward
        self.dones[self._curr_idx]        = done
        self.values[self._curr_idx]       = value

        self._curr_idx += 1

    def compute_advantage_and_return(self, last_value, last_done, gae_coeff=0.95, gamma=0.99):
        """Compute advantages and returns.
        
        Parameters
        --------------------
        last_value: torch.Tensor
            value-state of each actor at n_timesteps + 1

        last_done: torch.Tensor
            whether or not observation of each actor is a terminal states at n_timesteps + 1
            
        gae_coeff: float, optional
            GAE coefficient
            
        gamma: float, optional
            discount factor"""

        assert last_value.shape == self.advantages.shape[1:], f"Expected shape = {self.advantages.shape[1:]}. last_value.shape was {last_value.shape}"
        assert last_done.shape == self.advantages.shape[1:], f"Expected shape = {self.advantages.shape[1:]}. last_done.shape was {last_done.shape}"

        if gae_coeff <= 0.0 or gae_coeff > 1.0:
            raise ValueError("gae_coeff must be a value which belongs (0, 1].")
        
        if gamma <= 0.0 or gamma > 1.0:
            raise ValueError("gamma must be a value which belongs (0, 1].")
        
        #Compute advantages.
        last_gae_term = 0

        for i in reversed(range(self._n_timesteps)):
            if i == self._n_timesteps - 1:
                next_value = last_value
                next_done = last_done
            else:
                next_value = self.values[i + 1]
                next_done = self.dones[i + 1]
            
            delta = self.rewards[i] + (1 - next_done) * gamma * next_value - self.values[i]
            
            last_gae_term = delta + gamma * gae_coeff * last_gae_term * (1 - next_done)
            self.advantages[i] = last_gae_term    

        #Compute returns.    
        self.returns = self.advantages + self.values