"""
data_loader.py — 加载C++生成的训练批次，以及BS基准路径生成器（纯Python）

两种用途：
1. load_lrh_batch()：加载C++生成的提升粗糙Heston路径（.npy文件）
2. generate_bs_batch()：纯Python生成GBM路径（用于Gate 1 BS验证）
"""
import json
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 归一化工具
# ============================================================
def load_norm_stats(artifacts_dir: str | Path) -> dict:
    """加载C++生成的归一化统计（normalization.json）"""
    path = Path(artifacts_dir) / "normalization.json"
    with open(path) as f:
        stats = json.load(f)
    stats["mean"] = np.array(stats["mean"], dtype=np.float32)
    stats["std"]  = np.array(stats["std"],  dtype=np.float32)
    return stats


def normalize(states: np.ndarray, stats: dict) -> np.ndarray:
    """z-score归一化：(x - mean) / std，沿最后一维应用"""
    return (states - stats["mean"]) / stats["std"]


def unnormalize(states: np.ndarray, stats: dict) -> np.ndarray:
    return states * stats["std"] + stats["mean"]


# ============================================================
# 数据集：提升粗糙Heston路径
# ============================================================
class LRHDataset(Dataset):
    """
    加载C++生成的训练批次。

    状态张量已由C++进行z-score归一化（使用校准批次的统计）。
    dW1用于BSDE前向传播（未归一化）。
    payoff是折现收益。
    """
    def __init__(self, artifacts_dir: str | Path):
        d = Path(artifacts_dir)
        self.states  = torch.from_numpy(np.load(d / "training_states.npy"))  # (N, T+1, state_dim)
        self.dW1     = torch.from_numpy(np.load(d / "training_dW1.npy"))     # (N, T)
        self.payoff  = torch.from_numpy(np.load(d / "training_payoff.npy"))  # (N,)
        self.tau_grid = torch.from_numpy(np.load(d / "training_tau_grid.npy"))  # (T+1,)

        self.n_paths   = self.states.shape[0]
        self.n_times   = self.states.shape[1]
        self.state_dim = self.states.shape[2]
        self.n_steps   = self.dW1.shape[1]

    def __len__(self):
        return self.n_paths

    def __getitem__(self, idx):
        return self.states[idx], self.dW1[idx], self.payoff[idx]


# ============================================================
# BS基准路径生成器（纯Python，用于Gate 1验证）
# ============================================================
def generate_bs_batch(
    S0: float = 100.0,
    K:  float = 100.0,
    T:  float = 1.0,
    r:  float = 0.05,
    sigma: float = 0.20,
    n_paths: int = 10000,
    n_steps: int = 50,
    seed: int = 0,
    antithetic: bool = True,
) -> dict:
    """
    生成GBM（Black-Scholes）路径用于BSDE验证。

    返回与LRHDataset相同格式的dict：
      states:    (n_paths, n_steps+1, state_dim)  — state_dim = 3（tau, log_moneyness, V_t=sigma^2）
      dW1:       (n_paths, n_steps)
      payoff:    (n_paths,)
      tau_grid:  (n_steps+1,)
    注意：状态已经过z-score归一化（内部计算）。
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps
    disc = np.exp(-r * T)

    n_base = n_paths // 2 if antithetic else n_paths
    state_dim = 3  # [tau, log(S/K), V_t=sigma^2]（V_t在BS中为常数）

    # --- 生成BM增量 ---
    dW1_base = rng.standard_normal((n_base, n_steps)) * np.sqrt(dt)
    if antithetic:
        dW1 = np.concatenate([dW1_base, -dW1_base], axis=0)
    else:
        dW1 = dW1_base

    # --- 仿真路径 ---
    tau_grid = np.array([T - i * dt for i in range(n_steps + 1)], dtype=np.float32)
    log_S0K  = np.log(S0 / K)

    # 使用向量化对数正态更新
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * dW1  # (N, T)

    log_S_t   = np.zeros((n_paths, n_steps + 1))
    log_S_t[:, 0] = log_S0K
    for i in range(n_steps):
        log_S_t[:, i + 1] = log_S_t[:, i] + log_increments[:, i]

    # --- 构建状态张量 ---
    states = np.zeros((n_paths, n_steps + 1, state_dim), dtype=np.float32)
    for t in range(n_steps + 1):
        states[:, t, 0] = tau_grid[t]         # tau
        states[:, t, 1] = log_S_t[:, t]       # log(S_t/K)
        states[:, t, 2] = sigma ** 2           # V_t（常数）

    # --- z-score归一化（从校准样本计算）---
    cal_rng  = np.random.default_rng(0)
    dW1_cal  = cal_rng.standard_normal((500, n_steps)) * np.sqrt(dt)
    log_S_cal = np.zeros((500, n_steps + 1))
    log_S_cal[:, 0] = log_S0K
    for i in range(n_steps):
        log_S_cal[:, i + 1] = log_S_cal[:, i] + \
            (r - 0.5 * sigma**2) * dt + sigma * dW1_cal[:, i]

    cal_states = np.zeros((500, n_steps + 1, state_dim), dtype=np.float32)
    for t in range(n_steps + 1):
        cal_states[:, t, 0] = tau_grid[t]
        cal_states[:, t, 1] = log_S_cal[:, t]
        cal_states[:, t, 2] = sigma ** 2

    cal_flat  = cal_states.reshape(-1, state_dim)
    norm_mean = cal_flat.mean(axis=0)
    norm_std  = cal_flat.std(axis=0)
    norm_std  = np.where(norm_std < 1e-6, 1.0, norm_std)  # 防止除零

    states = (states - norm_mean) / norm_std

    # --- 折现收益 ---
    S_T     = K * np.exp(log_S_t[:, -1])
    payoff  = np.maximum(S_T - K, 0.0) * disc

    return {
        "states":    torch.from_numpy(states.astype(np.float32)),
        "dW1":       torch.from_numpy(dW1.astype(np.float32)),
        "payoff":    torch.from_numpy(payoff.astype(np.float32)),
        "tau_grid":  torch.from_numpy(tau_grid),
        "norm_mean": norm_mean,
        "norm_std":  norm_std,
        "state_dim": state_dim,
        "n_paths":   n_paths,
        "n_steps":   n_steps,
        "K": K, "T": T, "r": r, "sigma": sigma,
        # 解析BS价格（用于验证）
        "bs_price":  _bs_call(S0, K, T, r, sigma),
        "bs_delta":  _bs_delta(S0, K, T, r, sigma),
    }


def _bs_call(S, K, T, r, sigma):
    from math import log, sqrt, exp, erf
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N  = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    return S * N(d1) - K * exp(-r * T) * N(d2)


def _bs_delta(S, K, T, r, sigma):
    from math import log, sqrt, erf
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return 0.5 * (1 + erf(d1 / sqrt(2)))


# ============================================================
# LRH数据集的DataLoader工厂
# ============================================================
def make_lrh_dataloader(artifacts_dir: str | Path, batch_size: int = 2048,
                         shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    ds = LRHDataset(artifacts_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False)


if __name__ == "__main__":
    # 快速验证
    print("--- BS批次生成器测试 ---")
    bs = generate_bs_batch(n_paths=1000, n_steps=20, seed=0)
    print(f"  states shape: {bs['states'].shape}")
    print(f"  dW1 shape:    {bs['dW1'].shape}")
    print(f"  payoff shape: {bs['payoff'].shape}")
    print(f"  BS解析价格:   {bs['bs_price']:.4f}")
    print(f"  BS delta(t=0): {bs['bs_delta']:.4f}")
    print(f"  MC均值收益:   {bs['payoff'].mean().item():.4f}")

    import os
    artifacts = Path(__file__).parent.parent / "artifacts"
    if (artifacts / "training_states.npy").exists():
        print("\n--- LRH数据集测试 ---")
        ds = LRHDataset(artifacts)
        print(f"  路径数:     {ds.n_paths}")
        print(f"  时间步数:   {ds.n_steps}")
        print(f"  状态维度:   {ds.state_dim}")
        s, dw, p = ds[0]
        print(f"  第一条路径 states[0,0]: {s[0]}")
    else:
        print("\n  LRH批次未找到。请先运行: cd demo && ./build/demo_runner")
