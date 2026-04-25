"""
Dots and Boxes AlphaZero 策略价值网络

输入: batch × 7 × 11 × 11
输出:
  policy: batch × 60 (每条边的推荐概率)
  value:  batch × 1  (局面价值 [-1, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========== 网络参数 ==========
BOARD_SIZE = 11
CHANNELS = 7
ACTION_SIZE = 60
NUM_RES_BLOCKS = 4
NUM_FILTERS = 64


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
    """AlphaZero 风格的策略价值双头网络"""

    def __init__(self, in_channels=CHANNELS, num_filters=NUM_FILTERS,
                 num_res_blocks=NUM_RES_BLOCKS, action_size=ACTION_SIZE):
        super().__init__()

        # 骨干网络: 初始卷积 + 残差块
        self.initial_conv = nn.Conv2d(in_channels, num_filters, 3, padding=1, bias=False)
        self.initial_bn = nn.BatchNorm2d(num_filters)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, action_size)

        # 价值头
        self.value_conv = nn.Conv2d(num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 128)
        self.value_fc2 = nn.Linear(128, 1)

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


def create_model(device="cpu"):
    """创建并返回模型实例"""
    model = DotsAndBoxesNet().to(device)
    return model


if __name__ == "__main__":
    # 测试网络
    model = create_model()
    dummy_input = torch.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    policy, value = model(dummy_input)
    print(f"Policy shape: {policy.shape}")   # (1, 60)
    print(f"Value shape: {value.shape}")     # (1, 1)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
