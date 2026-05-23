"""
训练数据集

Schema v1 (原始):
{
    "board": [7×11×11 一维浮点数组],
    "player": 1 或 -1,
    "legal_mask": [60 维 0/1 数组],
    "policy": [60 维概率分布],
    "value": 浮点数 [-1, 1]
}

Schema v2 (扩展, 向下兼容):
  新增 game_id, move_index, decision_index, value_margin,
  black_score_final, white_score_final, winner, phase,
  teacher, simulations, temperature,
  root_policy_entropy, root_policy_confidence
"""

import json
import os
import random
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from augmentation import augment_sample, NUM_TRANSFORMS

BOARD_SIZE = 11
CHANNELS = 7
ACTION_SIZE = 60


class DotsAndBoxesDataset(Dataset):
    """AlphaZero 训练数据集"""

    def __init__(self, data_dir, file_pattern="*.jsonl", max_samples=0, min_policy_confidence=0.0, filenames=None, augment=False, value_mode="margin", q_weight=0.25, sample_weight_mode="none", sample_weight_clip=5.0, player_channel_mode="original"):
        """
        Args:
            data_dir: 数据目录路径
            file_pattern: 文件匹配模式
            max_samples: 最多加载多少条样本，0 表示全部
            min_policy_confidence: 最低 policy 置信度过滤
            filenames: 指定加载的文件名列表
            augment: 是否启用 D4 对称增强 (8 倍)
            value_mode: 价值目标模式
                "margin"  - 使用 value_margin (默认)
                "wdl"     - 使用 value (+1/-1/0)
                "q"       - 使用 root_q (MCTS Q值)
                "q+z"     - 混合: q_weight*Q + (1-q_weight)*z
                "q+margin"- 混合: q_weight*Q + (1-q_weight)*value_margin
            q_weight: Q 值权重 (value_mode="q+z" 时使用)
        """
        self.samples = []
        self.max_samples = max_samples
        self.min_policy_confidence = min_policy_confidence
        self.filenames = filenames
        self.augment = augment
        self.value_mode = value_mode
        self.q_weight = q_weight
        self.sample_weight_mode = sample_weight_mode
        self.sample_weight_clip = sample_weight_clip
        self.player_channel_mode = player_channel_mode
        self.sample_weights = None
        self._load_data(data_dir)
        self._compute_sample_weights()

    def _load_data(self, data_dir):
        """从目录中加载所有 JSONL 文件"""
        if not os.path.exists(data_dir):
            print(f"Warning: data directory {data_dir} does not exist")
            return

        filtered = 0
        filenames = self.filenames if self.filenames is not None else sorted(os.listdir(data_dir))
        for filename in filenames:
            if filename.endswith(".jsonl"):
                filepath = os.path.join(data_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            sample = json.loads(line)
                            if self.min_policy_confidence > 0.0 and max(sample["policy"]) < self.min_policy_confidence:
                                filtered += 1
                                continue
                            self.samples.append(sample)
                            if self.max_samples > 0 and len(self.samples) >= self.max_samples:
                                print(f"Loaded {len(self.samples)} samples from {data_dir}, filtered {filtered}")
                                return

        print(f"Loaded {len(self.samples)} samples from {data_dir}, filtered {filtered}")

    def _weight_key(self, sample):
        if self.sample_weight_mode == "player":
            return sample.get("player", 0)
        if self.sample_weight_mode == "winner":
            return sample.get("winner", 0)
        if self.sample_weight_mode == "player_winner":
            return (sample.get("player", 0), sample.get("winner", 0))
        return None

    def _compute_sample_weights(self):
        if self.sample_weight_mode == "none" or not self.samples:
            self.sample_weights = None
            return

        keys = [self._weight_key(sample) for sample in self.samples]
        counts = Counter(keys)
        total = len(keys)
        groups = max(len(counts), 1)
        weights = [total / (groups * counts[key]) for key in keys]

        if self.sample_weight_clip > 0.0:
            weights = [min(weight, self.sample_weight_clip) for weight in weights]

        mean_weight = sum(weights) / max(len(weights), 1)
        if mean_weight > 0.0:
            weights = [weight / mean_weight for weight in weights]

        self.sample_weights = weights
        print(f"Sample weight mode: {self.sample_weight_mode} groups={dict(counts)} min={min(weights):.3f} max={max(weights):.3f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 棋盘特征: 7 × 11 × 11
        board = np.array(sample["board"], dtype=np.float32).reshape(CHANNELS, BOARD_SIZE, BOARD_SIZE)

        # 合法动作 mask: 60
        legal_mask = np.array(sample["legal_mask"], dtype=np.float32)

        # 策略目标: 60（已经是概率分布）
        policy = np.array(sample["policy"], dtype=np.float32)

        # 价值目标: 根据 value_mode 选择
        if self.value_mode == "q" and "root_q" in sample:
            value = np.float32(sample["root_q"])
        elif self.value_mode == "q+z" and "root_q" in sample:
            q = sample["root_q"]
            z = sample["value"]
            value = np.float32(self.q_weight * q + (1 - self.q_weight) * z)
        elif self.value_mode == "q+margin" and "root_q" in sample and "value_margin" in sample:
            q = sample["root_q"]
            margin = sample["value_margin"]
            value = np.float32(self.q_weight * q + (1 - self.q_weight) * margin)
        elif self.value_mode == "wdl":
            value = np.float32(sample["value"])
        elif self.value_mode == "margin" and "value_margin" in sample:
            value = np.float32(sample["value_margin"])
        else:
            value = np.float32(sample["value"])

        # D4 对称增强: 随机选择一种变换
        if self.augment:
            t = np.random.randint(0, NUM_TRANSFORMS)
            board, legal_mask, policy = augment_sample(board, legal_mask, policy, t)

        if self.player_channel_mode == "zero":
            board[5, :, :] = 0.0
        elif self.player_channel_mode == "random_flip" and np.random.rand() < 0.5:
            board[5, :, :] *= -1.0

        item = (
            torch.from_numpy(board),
            torch.from_numpy(legal_mask),
            torch.from_numpy(policy),
            torch.tensor(value),
        )
        if self.sample_weights is not None:
            return item + (torch.tensor(np.float32(self.sample_weights[idx])),)
        return item


def create_dataloader(data_dir, batch_size=256, shuffle=True, num_workers=0, max_samples=0, min_policy_confidence=0.0, augment=False, value_mode="margin", q_weight=0.25, sample_weight_mode="none", sample_weight_clip=5.0, player_channel_mode="original"):
    """创建 DataLoader"""
    dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=augment, value_mode=value_mode, q_weight=q_weight, sample_weight_mode=sample_weight_mode, sample_weight_clip=sample_weight_clip, player_channel_mode=player_channel_mode)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def create_dataloaders(data_dir, batch_size=256, shuffle=True, num_workers=0, max_samples=0, val_split=0.0, split_seed=2026, min_policy_confidence=0.0, split_mode="sample", augment=False, value_mode="margin", q_weight=0.25, sample_weight_mode="none", sample_weight_clip=5.0, player_channel_mode="original", val_file_strategy="recent"):
    if val_split <= 0.0:
        dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=augment, value_mode=value_mode, q_weight=q_weight, sample_weight_mode=sample_weight_mode, sample_weight_clip=sample_weight_clip, player_channel_mode=player_channel_mode)
        if len(dataset) == 0:
            return None, None
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None
    if split_mode == "file":
        filenames = sorted(filename for filename in os.listdir(data_dir) if filename.endswith(".jsonl"))
        if len(filenames) < 2:
            dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=augment, value_mode=value_mode, q_weight=q_weight, sample_weight_mode=sample_weight_mode, sample_weight_clip=sample_weight_clip, player_channel_mode=player_channel_mode)
            if len(dataset) == 0:
                return None, None
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None
        # val 集分割策略 (val_file_strategy):
        #   "recent" (默认): 取文件名排序后的尾部 (最近 iter 数据)。
        #     理论依据 Sugiyama 2012 Density Ratio Estimation §1.2:
        #     ERM 必须让 val 与 deploy 分布同分布；AlphaZero 的 deploy =
        #     最新自对弈数据。
        #   "random": 旧行为，用 split_seed 随机选 (已知会造成分布偏移，仅
        #     用于诊断对照实验，不建议生产)。
        val_file_count = int(len(filenames) * val_split)
        val_file_count = max(1, min(val_file_count, len(filenames) - 1))
        if val_file_strategy == "random":
            rng = random.Random(split_seed)
            shuffled = list(filenames)
            rng.shuffle(shuffled)
            val_files = sorted(shuffled[:val_file_count])
            train_files = sorted(shuffled[val_file_count:])
        else:  # "recent"
            val_files = sorted(filenames[-val_file_count:])
            train_files = sorted(filenames[:-val_file_count])
        print(f"val_file_strategy={val_file_strategy}  val_files={val_files}")
        train_dataset = DotsAndBoxesDataset(
            data_dir,
            max_samples=max_samples,
            min_policy_confidence=min_policy_confidence,
            filenames=train_files,
            augment=augment,
            value_mode=value_mode,
            q_weight=q_weight,
            sample_weight_mode=sample_weight_mode,
            sample_weight_clip=sample_weight_clip,
            player_channel_mode=player_channel_mode,
        )
        val_dataset = DotsAndBoxesDataset(
            data_dir,
            min_policy_confidence=min_policy_confidence,
            filenames=val_files,
            augment=False,  # 验证集不增强
            value_mode=value_mode,
            q_weight=q_weight,
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return None, None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Train files: {len(train_files)}  Val files: {len(val_files)}")
        print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    base_dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=False, value_mode=value_mode, q_weight=q_weight)
    if len(base_dataset) == 0:
        return None, None
    val_size = int(len(base_dataset) * val_split)
    val_size = max(1, min(val_size, len(base_dataset) - 1))
    train_size = len(base_dataset) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(len(base_dataset), generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_base = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=augment, value_mode=value_mode, q_weight=q_weight, sample_weight_mode=sample_weight_mode, sample_weight_clip=sample_weight_clip, player_channel_mode=player_channel_mode)
    val_base = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence, augment=False, value_mode=value_mode, q_weight=q_weight)
    train_dataset = Subset(train_base, train_indices)
    val_dataset = Subset(val_base, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train samples: {train_size}  Val samples: {val_size}")
    return train_loader, val_loader
