"""
model.py — SharedWeightMLP + BSDE前向传播 + 损失函数

网络结构：
  输入: state (B, state_dim) — tau已包含在state中
  隐藏: Linear(state_dim, 64) → BatchNorm → ReLU → Linear(64, 64) → BatchNorm → ReLU
  输出: Y_scalar (B, 1)  |  Z_vector (B, state_dim)
  输出层: Linear(64, 1 + state_dim)

BSDE前向传播（f=0，一维对冲，有意简化）：
  Y = Y0（可训练标量）
  for i in range(n_steps):
      _, Z_i = net(states[:, i, :])
      Y = Y + Z_i[:, 0] * dW1[:, i]   # 仅对冲现货BM（dW2未对冲）
  返回 Y_T 和 Z_norms（用于Z正则化）

注意：Z_vector保持完整维度（state_dim）以实现：
  (a) Z正则化作用于所有分量，提升训练稳定性
  (b) 与未来波动率对冲扩展的前向兼容性
  在线使用时仅使用 Z[:, 0]（现货delta信号）。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedWeightMLP(nn.Module):
    """
    跨所有时间步共享权重的MLP。
    tau已包含在state中，因此不需要单独的时间嵌入层。
    使用BatchNorm提高训练稳定性。
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim

        # 注意：不使用BatchNorm。原因：
        #   BSDE训练中每个batch混合了不同时刻τ∈[0,T]的状态，
        #   BatchNorm的批次统计会混淆时间维度，干扰Z的收敛。
        #   深度BSDE文献（E, Han, Jentzen 2017）使用纯MLP或LayerNorm。
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1 + state_dim),  # Y_scalar + Z_vector
        )

        # Xavier初始化（减少梯度消失）
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: (B, state_dim) — 归一化状态向量

        Returns:
            Y: (B, 1)          — 延续值估计（仅用于诊断）
            Z: (B, state_dim)  — 控制向量（Z[:, 0] = 现货对冲信号）
        """
        out = self.net(state)           # (B, 1 + state_dim)
        Y = out[:, :1]                  # (B, 1)
        Z = out[:, 1:]                  # (B, state_dim)
        return Y, Z


def bsde_forward(
    net: SharedWeightMLP,
    Y0_param: nn.Parameter,
    states: torch.Tensor,      # (B, n_times, state_dim)
    dW1: torch.Tensor,         # (B, n_steps)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    BSDE前向传播（driver f=0，一维对冲）。

    更新规则（Euler-Maruyama，f=0）：
        Y_{i+1} = Y_i + Z_i[0] * dW1_i

    这是有意的一维简化：提升粗糙Heston模型由两个相关BM驱动
    （dW1=现货，dW2=波动率），但demo v1仅对冲可交易资产（dW1）。
    dW2未对冲——这是设计选择，不是模型方程中的错误。

    Args:
        net:      SharedWeightMLP
        Y0_param: 可训练标量参数（t=0时的初始价格估计）
        states:   (B, n_times, state_dim) — 归一化状态
        dW1:      (B, n_steps) — 现货BM增量（未归一化）

    Returns:
        Y_T:     (B,)              — 传播到到期时刻的值
        Z_norms: (B, n_steps)      — 每步Z[:, 0]的平方（用于Z正则化）
    """
    batch_size = states.shape[0]
    n_steps    = dW1.shape[1]

    # Y0扩展到整个batch
    Y = Y0_param.expand(batch_size)    # (B,)

    Z_norms = []

    for i in range(n_steps):
        x_i = states[:, i, :]          # (B, state_dim)
        _, Z_i = net(x_i)              # Z_i: (B, state_dim)

        Z_spot = Z_i[:, 0]             # 现货对冲分量 (B,)
        Z_norms.append(Z_i.pow(2).mean(dim=1))  # 所有分量的L2范数（用于正则化）

        # BSDE更新（f=0的鞅递推）
        Y = Y + Z_spot * dW1[:, i]

    Z_norms_tensor = torch.stack(Z_norms, dim=1)  # (B, n_steps)
    return Y, Z_norms_tensor


def bsde_loss(
    Y_T: torch.Tensor,          # (B,) — 传播的终值
    payoff: torch.Tensor,       # (B,) — 折现到期收益
    Z_norms: torch.Tensor,      # (B, n_steps)
    lambda_z: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    BSDE损失 = 终值误差 + Z正则化

    Args:
        Y_T:      传播到到期的Y值
        payoff:   折现期权收益
        Z_norms:  Z[:, 0]的平方（每步）
        lambda_z: Z正则化强度

    Returns:
        total_loss: 标量损失
        components: 损失分项dict（用于日志记录）
    """
    # 主损失：终值均方误差
    terminal_loss = F.mse_loss(Y_T, payoff)

    # Z正则化：防止Z爆炸或坍缩为零
    z_reg = lambda_z * Z_norms.mean()

    total = terminal_loss + z_reg

    return total, {
        "terminal": terminal_loss.item(),
        "z_reg":    z_reg.item(),
        "total":    total.item(),
    }


if __name__ == "__main__":
    # 快速形状测试
    print("=== model.py 形状测试 ===")
    B, T, D = 32, 20, 7
    net     = SharedWeightMLP(state_dim=D)
    Y0      = nn.Parameter(torch.zeros(1))

    states  = torch.randn(B, T + 1, D)
    dW1     = torch.randn(B, T) * 0.1

    Y_T, Z_norms = bsde_forward(net, Y0, states, dW1)
    payoff  = torch.relu(torch.randn(B))

    loss, comps = bsde_loss(Y_T, payoff, Z_norms)
    loss.backward()

    print(f"  Y_T shape:      {Y_T.shape}      期望: ({B},)")
    print(f"  Z_norms shape:  {Z_norms.shape}  期望: ({B}, {T})")
    print(f"  loss:           {loss.item():.4f}")
    print(f"  loss breakdown: {comps}")
    print(f"  Y0.grad:        {Y0.grad}")
    print("  backward() 成功 ✓")

    # 测试eval模式（ONNX导出时使用）
    net.eval()
    with torch.no_grad():
        dummy = torch.randn(1, D)
        Y_out, Z_out = net(dummy)
    print(f"\n  eval模式 single input:")
    print(f"    Y shape: {Y_out.shape}  期望: (1, 1)")
    print(f"    Z shape: {Z_out.shape}  期望: (1, {D})")
