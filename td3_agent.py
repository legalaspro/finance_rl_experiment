import numpy as np
import random
from collections import namedtuple, deque
from copy import deepcopy

from td3_model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
EXPLORATION_NOISE = 0.1 # exploration noise
TARGET_POLICY_NOISE = 0.2 # target actor smoothing noise
NOISE_CLIP = 0.5       # noise clip
POLICY_DELAY = 2       # policy delay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, fc_units=96).to(device)
        self.actor_target = deepcopy(self.actor_local)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, fc_units=96).to(device)
        self.critic_target = deepcopy(self.critic_local)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.step_counter = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)
        self.step_counter += 1

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = np.random.normal(0, EXPLORATION_NOISE, size=self.action_size)
            action += noise

        action = np.clip(action, -1.0, 1.0)
        action = (action + 1.0) / 2.0
        return action

    def reset(self):
        pass

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples.
        noised_action = actor_target(next_state) + clipped_noise(0,scale,-c,c)
        Q1_next_target, Q2_next_target = critic1_target(next_state, noised_action), critic2_target(next_state, noised_action)
        Q_targets = r + γ * min(Q1_next_target, Q2_next_target)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models        
        with torch.no_grad():
            clipped_noise = (torch.randn_like(actions) * TARGET_POLICY_NOISE).clamp(
                -NOISE_CLIP, NOISE_CLIP
            )
            actions_next = (self.actor_target(next_states) + clipped_noise).clamp(-1,1)
            Q1_next_targets, Q2_next_targets = self.critic_target(next_states, actions_next)
            Q_next_targets = torch.min(Q1_next_targets, Q2_next_targets)
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * (1-dones) * Q_next_targets)

        # Compute critic loss 
        Q1_expected, Q2_expected = self.critic_local(states, actions)
        Q1_loss = F.mse_loss(Q1_expected, Q_targets) * 0.5
        Q2_loss = F.mse_loss(Q2_expected, Q_targets) * 0.5
        critic_loss = Q1_loss + Q2_loss
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        # ---------------------------- update actor ---------------------------- #
        if self.step_counter % POLICY_DELAY == 0:
            # Compute actor loss each 2nd step
            actions = self.actor_local(states)
            Q1, Q2 = self.critic_local(states, actions)
            actor_loss = -torch.min(Q1, Q2).mean()

            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            
            #-------------------------- update target network ----------------------- # 
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)     
                        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)