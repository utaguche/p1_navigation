import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64, dueling_mode=False):
        """Initialize parameters and build model.
        Params
        ----------
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            dueling_mode (bool): whether to use the dueling network architecture
        -------
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.dueling_mode = dueling_mode
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # Linear layer to evaluate the state value function
        self.state_value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # whether to use the dueling network architecture
        if self.dueling_mode:
            # combine with the state value 
            return self.fc3(x) + self.state_value(x)
        else:
            return self.fc3(x)
