from typing import List, Optional, Tuple

import pytorch_lightning as pl

from src.datamodule.dataset import CelebADataset
from torch.utils.data.dataloader import DataLoader


class CelebADatamodule(pl.LightningDataModule):
    def __init__(
            self,
            data_path: str,
            img_type: str = "png",
            image_size: tuple[int, int] = (256, 256),
            train_batch_size: int = 8,
            eval_batch_size: int = 4,
            num_workers: int = 5,
            aug: bool = False,
            use_indices: bool = False,
            indices_data_root: str = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.img_type = img_type
        self.image_size = image_size
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.aug = aug
        self.use_indices = use_indices
        self.indices_data_root = indices_data_root

        print(f"Load data from: {self.data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = CelebADataset(
                self.data_path, self.img_type, "train", self.aug, self.image_size,
                self.use_indices, self.indices_data_root
            )
            self.val_dataset = CelebADataset(
                self.data_path, self.img_type, "val", self.aug, self.image_size,
                self.use_indices, self.indices_data_root
            )
        if stage == "test" or stage is None:
            self.test_dataset = CelebADataset(
                self.data_path, self.img_type, "test", self.aug, self.image_size,
                self.use_indices, self.indices_data_root
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            batch_size=self.train_batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            batch_size=self.eval_batch_size,
        )