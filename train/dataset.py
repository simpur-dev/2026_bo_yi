"""
训练数据集

数据格式（JSONL，每行一条样本）:
{
    "board": [7 × 11 × 11 的一维浮点数组],
    "player": 1 或 -1,
    "legal_mask": [60 维 0/1 数组],
    "policy": [60 维概率分布],
    "value": 浮点数 [-1, 1]
}
"""

import json
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

BOARD_SIZE = 11
CHANNELS = 7
ACTION_SIZE = 60


class DotsAndBoxesDataset(Dataset):
    """AlphaZero 训练数据集"""

    def __init__(self, data_dir, file_pattern="*.jsonl", max_samples=0, min_policy_confidence=0.0, filenames=None):
        """
        Args:
            data_dir: 数据目录路径
            file_pattern: 文件匹配模式
        """
        self.samples = []
        self.max_samples = max_samples
        self.min_policy_confidence = min_policy_confidence
        self.filenames = filenames
        self._load_data(data_dir)

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

        # 价值目标: 标量
        value = np.float32(sample["value"])

        return (
            torch.from_numpy(board),
            torch.from_numpy(legal_mask),
            torch.from_numpy(policy),
            torch.tensor(value),
        )


def create_dataloader(data_dir, batch_size=256, shuffle=True, num_workers=0, max_samples=0, min_policy_confidence=0.0):
    """创建 DataLoader"""
    dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader


def create_dataloaders(data_dir, batch_size=256, shuffle=True, num_workers=0, max_samples=0, val_split=0.0, split_seed=2026, min_policy_confidence=0.0, split_mode="sample"):
    dataset = DotsAndBoxesDataset(data_dir, max_samples=max_samples, min_policy_confidence=min_policy_confidence)
    if len(dataset) == 0:
        return None, None
    if val_split <= 0.0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None
    if split_mode == "file":
        filenames = sorted(filename for filename in os.listdir(data_dir) if filename.endswith(".jsonl"))
        if len(filenames) < 2:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), None
        rng = random.Random(split_seed)
        rng.shuffle(filenames)
        val_file_count = int(len(filenames) * val_split)
        val_file_count = max(1, min(val_file_count, len(filenames) - 1))
        val_files = sorted(filenames[:val_file_count])
        train_files = sorted(filenames[val_file_count:])
        train_dataset = DotsAndBoxesDataset(
            data_dir,
            max_samples=max_samples,
            min_policy_confidence=min_policy_confidence,
            filenames=train_files,
        )
        val_dataset = DotsAndBoxesDataset(
            data_dir,
            min_policy_confidence=min_policy_confidence,
            filenames=val_files,
        )
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            return None, None
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print(f"Train files: {len(train_files)}  Val files: {len(val_files)}")
        print(f"Train samples: {len(train_dataset)}  Val samples: {len(val_dataset)}")
        return train_loader, val_loader

    val_size = int(len(dataset) * val_split)
    val_size = max(1, min(val_size, len(dataset) - 1))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(split_seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Train samples: {train_size}  Val samples: {val_size}")
    return train_loader, val_loader
