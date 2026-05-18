"""
ONNX 一致性测试
验证 PyTorch 模型和 ONNX Runtime 推理结果一致

用法:
  python test_onnx_consistency.py --model_path ../data/models/phase4_resnet_m_best_model.pt --onnx_path ../data/models/resnet_m.onnx --arch resnet_m
"""

import argparse
import numpy as np
import torch
import onnxruntime as ort
from model import create_model, CHANNELS, BOARD_SIZE


def test_consistency(model_path, onnx_path, arch, num_tests=10):
    # 加载 PyTorch 模型
    model = create_model(arch=arch, device="cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # 加载 ONNX 模型
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    print(f"=== ONNX Consistency Test ===")
    print(f"  Arch:       {arch}")
    print(f"  PyTorch:    {model_path}")
    print(f"  ONNX:       {onnx_path}")
    print(f"  Tests:      {num_tests}")
    print()

    max_policy_diff = 0.0
    max_value_diff = 0.0

    for i in range(num_tests):
        # 随机输入
        x = np.random.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32)

        # PyTorch 推理
        with torch.no_grad():
            pt_policy, pt_value = model(torch.from_numpy(x))
        pt_policy = pt_policy.numpy()[0]
        pt_value = pt_value.numpy()[0, 0]

        # ONNX Runtime 推理
        ort_outputs = session.run(None, {input_name: x})
        ort_policy = ort_outputs[0][0]
        ort_value = ort_outputs[1][0, 0]

        # 比较
        policy_diff = np.max(np.abs(pt_policy - ort_policy))
        value_diff = abs(pt_value - ort_value)

        max_policy_diff = max(max_policy_diff, policy_diff)
        max_value_diff = max(max_value_diff, value_diff)

        if i < 3:
            print(f"  Test {i+1}: policy_max_diff={policy_diff:.2e}, value_diff={value_diff:.2e}")

    print()
    print(f"  Max policy diff: {max_policy_diff:.2e}")
    print(f"  Max value diff:  {max_value_diff:.2e}")

    # 判定
    threshold = 1e-5
    if max_policy_diff < threshold and max_value_diff < threshold:
        print(f"\n  ✅ PASS: All diffs < {threshold}")
        return True
    else:
        print(f"\n  ❌ FAIL: Diffs exceed {threshold}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--arch", type=str, default="resnet_m",
                        choices=["mlp", "cnn", "resnet_s", "resnet_m", "resnet_l"])
    parser.add_argument("--num_tests", type=int, default=20)
    args = parser.parse_args()

    success = test_consistency(args.model_path, args.onnx_path, args.arch, args.num_tests)
    exit(0 if success else 1)
