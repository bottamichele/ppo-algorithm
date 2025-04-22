import torch as tc

class Rollout:
    """A buffer which stores a collection of observations, rewards, actions and so on and is used for a PPO train step."""

    def __init__(self, num_steps, n_envs, obs_shape, action_shape, obs_dtype=tc.float32, act_dtype=tc.float32, device=tc.device("cpu")):
        """Create new rollout.
        
        Parameters
        --------------------
        num_steps: int
            number of steps run an enviroment

        n_envs: int
            number of enviroments used for training

        obs_shape: tuple
            observation shape

        action_shape: tuple
            action shape

        obs_dtype: torch.dtype
            observation's data type

        act_dtype: torch.dtype
            action's data type
            
        device: torch.device
            device which all item are stored on"""
        
        assert isinstance(action_shape, tuple) and (len(action_shape) == 0 or len(action_shape) == 1), "action_shape supports tuple with only zero (discrete actions) and one (continuous actions) dimensions."

        self.observations = tc.zeros((num_steps, n_envs, *obs_shape), dtype=obs_dtype, device=device)
        self.actions    = tc.zeros((num_steps, n_envs, *action_shape), dtype=act_dtype, device=device)
        self.log_probs  = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)
        self.rewards    = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)
        self.dones      = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)
        self.values     = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)
        self.advantages = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)
        self.returns    = tc.zeros((num_steps, n_envs), dtype=tc.float32, device=device)

        self._size = num_steps
        self._curr_idx = 0
        self._device = device

    @property
    def device(self):
        return self._device

    def store(self, obs, action, log_prob, reward, done, value):
        """Store current timestep infos.
        
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

    def clear(self):
        """Clear rollout."""

        self.observations = tc.fill(self.observations, 0)
        self.actions      = tc.fill(self.actions, 0)
        self.log_probs    = tc.fill(self.log_probs, 0)
        self.rewards      = tc.fill(self.rewards, 0)
        self.dones        = tc.fill(self.dones, 0)
        self.values       = tc.fill(self.values, 0)
        self.advantages   = tc.fill(self.advantages, 0)
        self.returns      = tc.fill(self.returns, 0)

        self._curr_idx = 0

    def compute_advantages_and_returns(self, last_value, last_done, gae_coeff=0.95, gamma=0.99):
        """Compute advantages and returns.
        
        Parameters
        --------------------
        last_value: torch.Tensor
            value-state of each actor at num_steps + 1

        last_done: torch.Tensor
            whether or not observation of each actor is a terminal states at num_steps + 1
            
        gae_coeff: float, optional
            GAE coefficient
            
        gamma: float, optional
            discount factor"""

        assert last_value.shape == self.advantages.shape[1:], f"Expected shape = {self.advantages.shape[1:]}. last_value.shape was {last_value.shape}"
        assert last_done.shape == self.advantages.shape[1:], f"Expected shape = {self.advantages.shape[1:]}. last_done.shape was {last_done.shape}"
        assert gamma > 0.0 and gamma <= 1.0, f"The discout factor gamma must be (0, 1]."
        
        with tc.no_grad():
            #Compute advantages.
            last_gae_term = 0.0

            for i in reversed(range(self._size)):
                if i == self._size - 1:
                    next_value = last_value.to(dtype=tc.float32)
                    next_done = last_done.to(dtype=tc.float32)
                else:
                    next_value = self.values[i + 1]
                    next_done = self.dones[i + 1]
                
                delta = self.rewards[i] + (1.0 - next_done) * gamma * next_value - self.values[i]
                self.advantages[i] = last_gae_term = delta + gamma * gae_coeff * last_gae_term * (1 - next_done)  

            #Compute returns.    
            self.returns = self.advantages + self.values