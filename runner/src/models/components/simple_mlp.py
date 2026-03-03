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

# 用于兼容action matching，效果不行
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

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        能量函数：R^(d+1) → R
        Args:
            x: 将t和x合并后d+1维的状态向量，shape=(batch_size, d + 1)
        Returns:
            标量能量值，shape=(batch_size, 1)
        """

        return self.model(x)

    def forward(self, t: torch.Tensor, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        前向传播：等价于∇ₓ(net.energy)，即能量关于x的梯度
        适配场景：self.energy 是普通函数，而非 nn.Module
        Args:
            t: 时间标量，shape=(batch_size,) 或 ()
            x: 状态向量，shape=(batch_size, d)
        Returns:
            梯度值，shape=(batch_size, d)
        """
        t = t.to(x.device)

        if t.dim() < 1 or t.shape[0] != x.shape[0]:
            t = t.repeat(x.shape[0])[:, None]
        if t.dim() < 2:
            t = t[:, None]
        # x = torch.cat([t, x], dim=-1)
        # x.requires_grad, t.requires_grad = True, True

        # t = t + torch.rand(size = (x.shape[0], 1)).type_as(x)
        
        
        with torch.inference_mode(mode=False):
            t_leaf = t.clone().requires_grad_(True)
            x_leaf = x.clone().requires_grad_(True)
            st = torch.sum(self.energy(torch.cat([t_leaf, x_leaf], dim=-1)))
            dsdt, dsdx = torch.autograd.grad(st, (t_leaf, x_leaf), create_graph=True)
        return dsdx


        # 以下是旧版本的写法, 于2026/3/1重写
        """ # 核心：强制开启梯度上下文（覆盖PL的no_grad）
        with torch.enable_grad():
            # 正确创建可追踪梯度的叶子张量（去掉detach，避免计算图断裂）
            x_leaf = x.clone().requires_grad_(True)
            batch_size = x_leaf.shape[0]
            device = x_leaf.device
            
            t = t.to(device)
            # 步骤1：压缩所有多余维度，只保留必要维度
            t = t.squeeze()  # 把标量/[1]/[1,1]/[128,1]都转为标量/[128]
            
            # 步骤2：根据t的最终形状，统一转为[batch_size, 1]
            if t.dim() == 0:
                # 情况1：标量 → 扩展为[batch_size, 1]
                t_expanded = t.expand(batch_size, 1)
            elif t.dim() == 1:
                if len(t) == 1:
                    # 情况2：[1] → 扩展为[batch_size, 1]
                    t_expanded = t.expand(batch_size, 1)
                elif len(t) == batch_size:
                    # 情况3：[128] → 增加最后一维，变为[128, 1]
                    t_expanded = t.unsqueeze(-1)
                else:
                    # 情况4：其他长度 → 取第一个值，扩展为[batch_size, 1]
                    t_expanded = t[0].expand(batch_size, 1)
            else:
                # 情况5：高维张量 → 取第一个时间值，扩展为[batch_size, 1]
                t_expanded = t[0, 0].expand(batch_size, 1)

            # 拼接输入 (batch_size, d+1)
            inputs = torch.cat([t_expanded, x_leaf], dim=-1)
            
            # 计算能量值（self.energy是普通函数，无需train/eval切换）
            energy_val = self.energy(inputs)

            # 统一能量输出为一维 (batch_size,)
            energy_val = energy_val.squeeze(-1)
            if energy_val.dim() > 1:
                energy_val = energy_val.reshape(-1)
            
            # 安全计算梯度，捕获所有可能的异常
            try:
                grad = torch.autograd.grad(
                    outputs=energy_val.sum(),  # 标量输出保证梯度计算有效
                    inputs=x_leaf,
                    create_graph=True,         # ODE求解需要二阶梯度，必须开启
                    retain_graph=True,         # 保留计算图供后续ODE步骤使用
                    only_inputs=True,
                    allow_unused=True,         # 容忍x未被使用的边界情况
                )[0]
            except (RuntimeError, AttributeError):
                # 梯度计算失败时返回全0，避免程序崩溃
                grad = torch.zeros_like(x_leaf)

            # 处理梯度为空/形状不匹配的情况
            if grad is None:
                grad = torch.zeros_like(x_leaf)
            elif grad.shape != x_leaf.shape:
                grad = grad.reshape(x_leaf.shape)

            # 确保梯度张量和原x的设备/类型一致
            grad = grad.to(x.dtype).to(x.device)
            
            return grad """

if __name__ == "__main__":
    _ = SimpleDenseNet()
    _ = TimeInvariantVelocityNet()


