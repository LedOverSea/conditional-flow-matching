from typing import List, Optional

import torch
from torch import nn

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "silu": nn.SiLU,
}


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str,
        batch_norm: bool = True,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256, 256]
        dims = [input_size, *hidden_dims, target_size]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DivergenceFreeNet(SimpleDenseNet):
    """Implements a divergence free network as the gradient of a scalar potential function."""

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=1, *args, **kwargs)

    def energy(self, x):
        return self.model(x)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        x = torch.cat([t, x], dim=-1)
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.model(x)), x, create_graph=True)[0]
        return grad[:, :-1]


class TimeInvariantVelocityNet(SimpleDenseNet):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim, target_size=dim, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        del t
        return self.model(x)


class VelocityNet(SimpleDenseNet):
    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=dim, *args, **kwargs)

    def forward(self, t, x, *args, **kwargs):
        """Ignore t run model."""
        # 2025/12/18 新增: 把t转移到x的设备上
        t = t.to(x.device)

        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        x = torch.cat([t, x], dim=-1)
        return self.model(x)


class EnergyVelocityNet(SimpleDenseNet):
    """VelocityNet with energy method for ActionMatchingLitModule compatibility.
    
    This network provides both velocity field computation (via forward) 
    and energy field computation (via energy) required by ActionMatchingLitModule.
    """
    
    def __init__(self, dim: int, *args, **kwargs):
        # Initialize with input_size=dim+1 for (t, x) concatenation
        # target_size should be hidden_dim for intermediate representation
        super().__init__(input_size=dim + 1, target_size=128, *args, **kwargs)  # Use 128 as hidden_dim
        self.dim = dim
        self.hidden_dim = 128
        
        # Create separate heads for velocity and energy
        self.velocity_head = nn.Linear(self.hidden_dim, dim)
        self.energy_head = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, t, x, *args, **kwargs):
        """Forward pass returns velocity field.
        
        Args:
            t: Time parameter
            x: Input data
            
        Returns:
            Velocity field of shape [batch_size, dim]
        """
        # Ensure t and x are on the same device
        t = t.to(x.device)
        
        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]

        # Concatenate t and x
        x_input = torch.cat([t, x], dim=-1)
        
        # Get intermediate representation
        hidden = self.model(x_input)
        
        # Compute velocity from the velocity head
        velocity = self.velocity_head(hidden)
        
        return velocity
    
    def energy(self, x_input):
        """Energy function for ActionMatchingLitModule.
        
        Args:
            x_input: Input data concatenated with time (should be [batch_size, dim+1])
            
        Returns:
            Energy field (scalar) of shape [batch_size]
        """
        # Ensure input has correct shape [batch_size, dim+1]
        if x_input.shape[-1] != self.dim + 1:
            raise ValueError(f"Expected input shape [-1, {self.dim + 1}], got {x_input.shape}")
        
        # Get intermediate representation
        hidden = self.model(x_input)
        
        # Compute energy from the energy head
        energy = self.energy_head(hidden)
        
        # Return scalar energy values
        return energy.squeeze(-1)


if __name__ == "__main__":
    _ = SimpleDenseNet()
    _ = TimeInvariantVelocityNet()


