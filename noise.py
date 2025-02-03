import numpy as np
import random
import copy


class ActionNoise(object):

    def sample(slef):
        pass

    def reset(self):
        pass


class OUNoise(ActionNoise):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#exploration-vs-exploitation
class NormalActionNoise(ActionNoise):
    """
    Normal or Gaussian noise
    noise∼N(0,σ2)
    """

    def __init__(self, size, seed=0, scale=0.2):
        self.action_dim = size
        self.scale = scale # Standard deviation of the noise
        self.seed = seed
        np.random.seed(self.seed)
    
    def reset(self):
        pass

    def sample(self):
        return np.random.normal(loc=0, scale=self.scale, size=self.action_dim)
        # return np.random.randn(self.action_dim)*self.scale

class DecayingGaussianNoise(ActionNoise):
    """
    Decaying Gausssian Noise
    """
    def __init__(self, action_dim, seed=0, initial_scale=0.25, final_scale=0.01, decay_rate=1e-4):
        self.action_dim = action_dim
        self.scale = initial_scale
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.decay_rate = decay_rate
        self.steps = 0
        self.seed = seed
        np.random.seed(self.seed)
    
    def reset(self):
        """Reset the internal scale and steps."""
        # self.scale = self.initial_scale
        # self.steps = 0
        pass

    def sample(self):
        # Update noise scale
        self.steps += 1
        current_scale = max(self.final_scale, self.scale * np.exp(-self.decay_rate * self.steps))
        # return np.random.normal(self.mu=0, current_scale, size=self.action_dim)
        return np.random.randn(self.action_dim)*current_scale
