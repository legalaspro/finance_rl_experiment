import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights_xavier(m):
    """
    Apply xavier initialization to linear layers or do nothing if other layer.
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, state_size, action_size, fc_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (int): Number of nodes in hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc_out = nn.Linear(fc_units, action_size)
        
        # Optional: call a custom init
        self.apply(init_weights_xavier)
        
        # Optionally keep small final layers
        nn.init.uniform_(self.fc_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_out(x))
        return x

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (int): Number of nodes in the first hidden layer
        """
        super().__init__()
         # Q1 architecture
        self.q1_fc1 = nn.Linear(state_size + action_size, fc_units)
        self.q1_fc2 = nn.Linear(fc_units, fc_units)
        self.q1_out = nn.Linear(fc_units, 1)
        
        # Q2 architecture
        self.q2_fc1 = nn.Linear(state_size + action_size, fc_units)
        self.q2_fc2 = nn.Linear(fc_units, fc_units)
        self.q2_out = nn.Linear(fc_units, 1)
        
        # Initialize
        self.apply(init_weights_xavier)
        
        # Optionally keep small final layers
        nn.init.uniform_(self.q1_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q1_out.bias)
        nn.init.uniform_(self.q2_out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.q2_out.bias)

    
    def forward(self, state, action):
        """
        Returns Q1, Q2. Both are shape (batch, 1).
        """
        # cat state and action
        xu = torch.cat([state, action], dim=-1)

        # Q1
        x1 = F.relu(self.q1_fc1(xu))
        x1 = F.relu(self.q1_fc2(x1))
        q1 = self.q1_out(x1)
        
        # Q2
        x2 = F.relu(self.q2_fc1(xu))
        x2 = F.relu(self.q2_fc2(x2))
        q2 = self.q2_out(x2)
        
        return q1, q2