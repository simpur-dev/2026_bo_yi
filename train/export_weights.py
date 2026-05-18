"""
将 PyTorch 模型权重导出为二进制文件，供 C++ 手写前向传播使用

导出格式:
  - 按层顺序写入所有参数
  - 每个参数: 先写 4 字节整数表示元素总数，再写 float32 数据
  - 同时输出一个 .txt 描述文件记录每个参数的名称、形状和偏移

用法:
  python export_weights.py --model_path ../data/models/best_model.pt --output ../data/models/weights.bin
"""

import argparse
import struct
import torch
import numpy as np
from model import create_model


def is_official_weight_path(path):
    normalized = path.replace("\\", "/")
    return normalized.endswith("data/models/weights.bin") or normalized.endswith("../data/models/weights.bin")


def export_weights(model_path, output_bin, output_desc, arch="mlp"):
    model = create_model(arch=arch, device="cpu")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with open(output_bin, "wb") as f_bin, open(output_desc, "w") as f_desc:
        offset = 0
        for name, param in model.named_parameters():
            data = param.detach().cpu().numpy().astype(np.float32).flatten()
            num_elements = len(data)
            shape = list(param.shape)

            f_desc.write(f"{name} {shape} offset={offset} count={num_elements}\n")

            # 写入元素数量
            f_bin.write(struct.pack("i", num_elements))
            offset += 4

            # 写入 float32 数据
            f_bin.write(data.tobytes())
            offset += num_elements * 4

        # 对 BatchNorm 的 running_mean 和 running_var 也要导出
        for name, buf in model.named_buffers():
            data = buf.detach().cpu().numpy().astype(np.float32).flatten()
            num_elements = len(data)
            shape = list(buf.shape)

            f_desc.write(f"{name} {shape} offset={offset} count={num_elements}\n")

            f_bin.write(struct.pack("i", num_elements))
            offset += 4

            f_bin.write(data.tobytes())
            offset += num_elements * 4

    print(f"Exported weights to {output_bin}")
    print(f"Exported description to {output_desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="PyTorch 模型路径")
    parser.add_argument("--output", type=str, default="../data/models/candidate_weights.bin", help="二进制权重输出路径")
    parser.add_argument("--arch", type=str, default="mlp",
                        choices=["mlp", "cnn", "resnet_s", "resnet_m", "resnet_l"],
                        help="网络架构 (必须与训练时一致)")
    parser.add_argument("--promote", action="store_true", help="允许导出覆盖正式 weights.bin")
    args = parser.parse_args()

    output_bin = args.output
    if is_official_weight_path(output_bin) and not args.promote:
        raise SystemExit("Refusing to overwrite official weights.bin without --promote")
    output_desc = args.output.replace(".bin", "_desc.txt")
    export_weights(args.model_path, output_bin, output_desc, arch=args.arch)
