import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

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

class EnergyBasedNet(SimpleDenseNet):
    """
    基于能量的深度学习网络类
    核心特性：
    - net.energy(t, x): R^(d+1) → R，输入(t, x)输出标量能量
    - net.forward(t, x): 等价于∇ₓ(net.energy)，即能量关于x的梯度
    """
    def __init__(self, dim: int, *args, **kwargs):
        """
        初始化网络
        Args:
            input_dim: x的维度d（t是标量，整体输入维度为d+1）
            hidden_dims: 隐藏层维度列表，默认[64, 64]
        """
        super().__init__(input_size=dim+1, target_size=1, *args, **kwargs)

    def energy(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        能量函数：R^(d+1) → R
        Args:
            t: 时间标量，shape=(batch_size,) 或 ()
            x: 状态向量，shape=(batch_size, d)
        Returns:
            标量能量值，shape=(batch_size, 1)
        """

        t = t.to(x.device)

        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        x = torch.cat([t, x], dim=-1)
        return self.model(x)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：等价于∇ₓ(net.energy)，即能量关于x的梯度
        Args:
            t: 时间标量，shape=(batch_size,) 或 ()
            x: 状态向量，shape=(batch_size, d)（需要求导）
        Returns:
            梯度值，shape=(batch_size, d)
        """
        # 确保x需要计算梯度
        x.requires_grad_(True)
        
        # 计算能量值
        energy_val = self.energy(t, x)
        
        # 计算energy对x的梯度（∇ₓ(energy)）
        # torch.autograd.grad：计算标量对张量的梯度
        grad = torch.autograd.grad(
            outputs=energy_val.sum(),  # sum转为标量，不影响梯度方向
            inputs=x,
            create_graph=True,  # 保留计算图，支持二阶导数（可选）
            retain_graph=True,  # 保留计算图，避免后续计算报错
            only_inputs=True
        )[0]
        
        # 恢复x的requires_grad状态（避免影响后续计算）
        x.requires_grad_(False)
        
        return grad


# ------------------- 测试代码 -------------------
if __name__ == "__main__":
    # 初始化参数：x的维度d=3
    d = 3
    net = EnergyBasedNet(dim=d, activation="selu")
    
    # 生成测试数据：batch_size=2，x维度3
    batch_size = 2
    t = torch.tensor([0.1, 0.2])  # shape=(2,)
    x = torch.randn(batch_size, d)  # shape=(2,3)
    
    # 测试energy函数
    energy_out = net.energy(t, x)
    print("Energy输出形状:", energy_out.shape)  # 预期: (2,1)
    print("Energy输出值:\n", energy_out)
    
    # 测试forward（梯度）函数
    grad_out = net.forward(t, x)
    print("\nForward（梯度）输出形状:", grad_out.shape)  # 预期: (2,3)
    print("Forward（梯度）输出值:\n", grad_out)