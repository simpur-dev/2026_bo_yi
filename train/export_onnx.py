"""
将 PyTorch 模型导出为 ONNX 格式，供 C++ ONNX Runtime 推理使用

用法:
  python export_onnx.py --model_path ../data/models/best_model.pt --output ../data/models/model.onnx
"""

import argparse
import torch
from model import DotsAndBoxesNet, CHANNELS, BOARD_SIZE


def export_onnx(model_path, output_path):
    model = DotsAndBoxesNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch 模型路径")
    parser.add_argument("--output", type=str, default="../data/models/model.onnx", help="ONNX 输出路径")
    args = parser.parse_args()
    export_onnx(args.model_path, args.output)
