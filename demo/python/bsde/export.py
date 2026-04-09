"""
export.py — 将训练好的BSDE模型导出为ONNX格式

导出内容：
  - neural_bsde.onnx：单步MLP（不含时间循环，循环在C++中执行）
  - Y0_init.json：初始价格参数

Gate 2验证（C++集成前必须通过）：
  对1000个随机输入，PyTorch与ONNX Runtime的输出差异 < 1e-5

用法:
  python python/export.py --checkpoint artifacts/checkpoints/best.pt
  python python/export.py --checkpoint artifacts/checkpoints/best.pt --validate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from model import SharedWeightMLP


def export_onnx(checkpoint_path: str | Path, artifacts_dir: str | Path,
                validate: bool = True):
    """
    从checkpoint加载模型并导出为ONNX。

    重要：仅导出单步MLP（net.forward）。
    BSDE时间循环（bsde_forward）保留在C++推理端。
    """
    artifacts = Path(artifacts_dir)
    ckpt_path = Path(checkpoint_path)

    print(f"[导出] 从 {ckpt_path} 加载checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dim  = ckpt["state_dim"]
    hidden_dim = ckpt.get("hidden_dim", 64)
    Y0_val     = ckpt["Y0"]

    print(f"  state_dim={state_dim}, hidden_dim={hidden_dim}, Y0={Y0_val:.4f}")
    print(f"  训练epoch={ckpt['epoch']}, loss={ckpt['loss']:.5f}")

    net = SharedWeightMLP(state_dim=state_dim, hidden_dim=hidden_dim)
    net.load_state_dict(ckpt["net"])
    net.eval()

    # --------------------------------------------------------
    # ONNX导出（仅单步MLP）
    # 输入:  state (batch, state_dim)
    # 输出:  Y (batch, 1)  |  Z (batch, state_dim)
    # --------------------------------------------------------
    onnx_path = artifacts / "neural_bsde.onnx"
    dummy_input = torch.randn(1, state_dim)

    print(f"[导出] 导出ONNX到 {onnx_path}...")
    torch.onnx.export(
        net,
        dummy_input,
        str(onnx_path),
        input_names=["state"],
        output_names=["Y", "Z"],
        dynamic_axes={
            "state": {0: "batch"},
            "Y":     {0: "batch"},
            "Z":     {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX导出完成: {onnx_path}")

    # --------------------------------------------------------
    # 保存Y0初始值（C++推理使用）
    # --------------------------------------------------------
    y0_path = artifacts / "Y0_init.json"
    with open(y0_path, "w") as f:
        json.dump({"Y0": Y0_val, "state_dim": state_dim}, f, indent=2)
    print(f"  Y0已保存: {y0_path}")

    # --------------------------------------------------------
    # Gate 2验证：PyTorch vs ONNX批次比较
    # --------------------------------------------------------
    if validate:
        print(f"\n[Gate 2] 运行ONNX批次验证 (1000个随机输入)...")
        try:
            import onnxruntime as ort
        except ImportError:
            print("  警告: onnxruntime未安装，跳过验证。")
            print("  安装: pip install onnxruntime")
            return

        sess = ort.InferenceSession(str(onnx_path))

        # 从训练分布中采样随机输入（均值0，标准差1 — 归一化后的状态）
        rng     = np.random.default_rng(seed=2024)
        n_test  = 1000
        test_inputs = rng.standard_normal((n_test, state_dim)).astype(np.float32)

        # PyTorch推理
        net.eval()
        with torch.no_grad():
            pt_Y, pt_Z = net(torch.from_numpy(test_inputs))
        pt_Y = pt_Y.numpy()
        pt_Z = pt_Z.numpy()

        # ONNX Runtime推理
        ort_Y, ort_Z = sess.run(None, {"state": test_inputs})

        # 比较
        max_err_Y = np.abs(pt_Y - ort_Y).max()
        max_err_Z = np.abs(pt_Z - ort_Z).max()

        print(f"  Y最大绝对误差: {max_err_Y:.2e}  (阈值: 1e-5)")
        print(f"  Z最大绝对误差: {max_err_Z:.2e}  (阈值: 1e-5)")

        if max_err_Y < 1e-5 and max_err_Z < 1e-5:
            print(f"[Gate 2] ✓ PyTorch与ONNX输出匹配 — 通过")
        else:
            print(f"[Gate 2] ✗ 误差超过阈值 — C++集成前请勿继续")
            print(f"  注意：BatchNorm在推理时使用运行统计，")
            print(f"  请确保模型以net.eval()模式导出，并已收敛（BatchNorm统计稳定）。")

    print(f"\n[导出完成]")
    print(f"  artifacts/neural_bsde.onnx")
    print(f"  artifacts/Y0_init.json")
    print(f"\n  下一步（Gate 2通过后）:")
    print(f"  cd build && cmake .. -DBUILD_ONNX_DEMO=ON -DONNXRUNTIME_ROOT=$HOME/onnxruntime && make")
    print(f"  ./demo_runner")


def main():
    parser = argparse.ArgumentParser(description="BSDE模型ONNX导出")
    parser.add_argument("--checkpoint", type=str,
                        default="artifacts/checkpoints/best.pt",
                        help="checkpoint文件路径")
    parser.add_argument("--artifacts",  type=str,
                        default=None,
                        help="artifacts目录（默认：demo/artifacts）")
    parser.add_argument("--validate",   action="store_true", default=True,
                        help="导出后运行Gate 2批次验证（默认开启）")
    parser.add_argument("--no-validate", dest="validate", action="store_false",
                        help="跳过Gate 2验证")
    args = parser.parse_args()

    if args.artifacts is None:
        args.artifacts = str(Path(__file__).parent.parent / "artifacts")

    export_onnx(args.checkpoint, args.artifacts, validate=args.validate)


if __name__ == "__main__":
    main()
