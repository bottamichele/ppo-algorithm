class Trajectory:
    """A trajectory segment which stores stuff, such as observations and actions, of an episode."""

    def __init__(self):
        """Create new trajectory segment."""

        self.observations = []
        self.actions = []
        self.action_distributions = []
        self.log_probs = []
        self.rewards = []
        self.returns = []

    def clear(self):
        """Clear this trajectory segment."""

        self.observations.clear()
        self.actions.clear()
        self.action_distributions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.returns.clear()

    def store(self, obs, action, action_dist, reward):
        """Store current observation, action, action's distrubution and reward.

        Parameters
        --------------------
        obs: torch.Tensor
            an observation

        action: torch.Tensor
            an action

        action_dist: torch.distrbutions.distribution.Distribution
            a distribution

        reward: float
            a reward
        """     
        
        self.observations.append(obs)
        self.actions.append(action)
        self.action_distributions.append(action_dist)
        self.log_probs.append(action_dist.log_prob(action))
        self.rewards.append(reward)

    def compute_returns(self, gamma):
        """Compute discounted rewards.
        
        Parameter
        --------------------
        gamma: float
            discount factor"""
        
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError("gamma must be a value between 0.0 and 1.0")
        
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + discounted_reward * gamma
            self.returns.insert(0, discounted_reward)