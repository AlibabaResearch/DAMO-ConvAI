"""
Subspace class.
"""

import torch
import torch.nn as nn


class Subspace(nn.Module):
    """
    Subspace.
    """
    subspaces = ["D", "I", "S", "V", "DI", "IS", "SV", "DIS", "ISV", "DISV"]

    def __init__(self, hidden_dim, subspace_dim, trigger_subspaces):
        super(Subspace, self).__init__()

        self.trigger_subspaces = trigger_subspaces.split(',') if trigger_subspaces else self.subspaces
        self.trigger_indices = [self.subspaces.index(subspace) for subspace in self.trigger_subspaces]
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.subspace_dim * len(self.subspaces)),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.projection(x)
        out = out.reshape(x.size(0), len(self.subspaces), self.subspace_dim)
        out = torch.cat([out[:, index: index + 1, :] for index in self.trigger_indices], dim=1)
        return out
