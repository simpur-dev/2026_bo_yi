"""
Dots and Boxes AlphaZero 策略价值网络

输入: batch × 7 × 11 × 11
输出:
  policy: batch × 60 (每条边的推荐概率)
  value:  batch × 1  (局面价值 [-1, 1])

架构:
  mlp        - 轻量 MLP (257K params)，可 C++ 手写推理
  resnet_s   - 4 blocks × 64 channels (~120K params)，快速调试
  resnet_m   - 6 blocks × 128 channels (~550K params)，主力候选
  resnet_l   - 10 blocks × 256 channels (~3.5M params)，长期强力模型
  cnn        - resnet_s 的别名 (向下兼容)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 常量 ==========
BOARD_SIZE = 11
CHANNELS = 7
ACTION_SIZE = 60

# ========== ResNet 预设 ==========
RESNET_PRESETS = {
    "resnet_s": {"num_res_blocks": 4,  "num_filters": 64,  "value_hidden": 128},
    "resnet_m": {"num_res_blocks": 6,  "num_filters": 128, "value_hidden": 256},
    "resnet_l": {"num_res_blocks": 10, "num_filters": 256, "value_hidden": 256},
}

ALL_ARCHS = ["mlp", "cnn", "resnet_s", "resnet_m", "resnet_l"]


class ResidualBlock(nn.Module):
    """残差块: Conv -> BN -> ReLU -> Conv -> BN -> skip connection -> ReLU"""

    def __init__(self, num_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class DotsAndBoxesNet(nn.Module):
    """
    AlphaZero 风格 ResNet 策略价值双头网络

    支持通过 num_filters / num_res_blocks / value_hidden 参数化：
      resnet_s:  4 blocks × 64 ch,  value_hidden=128
      resnet_m:  6 blocks × 128 ch, value_hidden=256
      resnet_l: 10 blocks × 256 ch, value_hidden=256
    """

    def __init__(self, in_channels=CHANNELS, num_filters=64,
                 num_res_blocks=4, action_size=ACTION_SIZE,
                 value_hidden=128, policy_channels=2):
        super().__init__()
        self.board_size = BOARD_SIZE

        # 骨干网络: 初始卷积 + 残差块
        self.initial_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(num_filters, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * BOARD_SIZE * BOARD_SIZE, action_size)

        # 价值头
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 7, 11, 11) 棋盘特征张量

        Returns:
            policy_logits: (batch, 60) 策略 logits（未 softmax）
            value: (batch, 1) 局面价值 [-1, 1]
        """
        # 骨干
        out = F.relu(self.initial_bn(self.initial_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        # 策略头
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


class DotsAndBoxesMLP(nn.Module):
    """
    轻量 MLP 策略价值网络

    优点:
      - 结构简单，易于在 C++ 中手写前向传播
      - 无 BatchNorm/Conv，推理速度快
      - 适合作为初版验证网络

    结构:
      Flatten(7×11×11=847) -> FC(256) -> ReLU -> FC(128) -> ReLU
      -> Policy Head: FC(60)
      -> Value Head: FC(1) + Tanh
    """

    INPUT_SIZE = CHANNELS * BOARD_SIZE * BOARD_SIZE  # 847
    HIDDEN1 = 256
    HIDDEN2 = 128

    def __init__(self, action_size=ACTION_SIZE):
        super().__init__()

        self.fc1 = nn.Linear(self.INPUT_SIZE, self.HIDDEN1)
        self.fc2 = nn.Linear(self.HIDDEN1, self.HIDDEN2)
        self.policy_head = nn.Linear(self.HIDDEN2, action_size)
        self.value_head = nn.Linear(self.HIDDEN2, 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 7, 11, 11) 棋盘特征张量

        Returns:
            policy_logits: (batch, 60) 策略 logits（未 softmax）
            value: (batch, 1) 局面价值 [-1, 1]
        """
        x = x.view(x.size(0), -1)  # Flatten -> (batch, 847)
        x = F.relu(self.fc1(x))    # -> (batch, 256)
        x = F.relu(self.fc2(x))    # -> (batch, 128)

        policy_logits = self.policy_head(x)   # -> (batch, 60)
        value = torch.tanh(self.value_head(x)) # -> (batch, 1)

        return policy_logits, value


def create_model(arch="cnn", device="cpu"):
    """
    创建并返回模型实例

    Args:
        arch: 架构名称
            "mlp"      - 轻量 MLP
            "cnn"      - ResNet small (向下兼容别名)
            "resnet_s" - 4 blocks × 64 channels
            "resnet_m" - 6 blocks × 128 channels
            "resnet_l" - 10 blocks × 256 channels
        device: 目标设备
    """
    if arch == "mlp":
        model = DotsAndBoxesMLP().to(device)
    elif arch in RESNET_PRESETS:
        preset = RESNET_PRESETS[arch]
        model = DotsAndBoxesNet(
            num_filters=preset["num_filters"],
            num_res_blocks=preset["num_res_blocks"],
            value_hidden=preset["value_hidden"],
        ).to(device)
    elif arch == "cnn":
        # 向下兼容: cnn = resnet_s
        preset = RESNET_PRESETS["resnet_s"]
        model = DotsAndBoxesNet(
            num_filters=preset["num_filters"],
            num_res_blocks=preset["num_res_blocks"],
            value_hidden=preset["value_hidden"],
        ).to(device)
    else:
        raise ValueError(f"Unknown arch: {arch}. Choose from: {ALL_ARCHS}")
    return model


if __name__ == "__main__":
    import sys

    arch = sys.argv[1] if len(sys.argv) > 1 else "resnet_s"
    print(f"=== Architecture: {arch.upper()} ===")

    model = create_model(arch=arch)
    dummy_input = torch.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy, value = model(dummy_input)

    print(f"Policy shape: {policy.shape}")   # (1, 60)
    print(f"Value shape:  {value.shape}")    # (1, 1)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 列出所有参数名和形状
    print("\nParameters:")
    for name, param in model.named_parameters():
        print(f"  {name:30s} {list(param.shape)}")

    # 列出所有架构的参数量
    print("\n=== All architectures ===")
    for a in ALL_ARCHS:
        m = create_model(arch=a)
        n = sum(p.numel() for p in m.parameters())
        print(f"  {a:12s} {n:>10,} params")

    # MLP 权重总量（用于估算 C++ 推理内存需求）
    if arch == "mlp":
        total_bytes = sum(p.numel() * 4 for p in model.parameters())
        print(f"\nWeight file size (float32): {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")
