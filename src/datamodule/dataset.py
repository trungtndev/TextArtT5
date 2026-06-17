import os

import torch
import torchvision
from PIL import Image
import torchvision.transforms.v2 as v2
from torch.utils.data.dataset import Dataset
import pandas as pd
import json
from typing import Tuple


class CelebADataset(Dataset):
    def __init__(self, contain_path, img_type='png',
                 split: str = "train", aug: bool = False, image_size: Tuple[int, int] = (256, 256),
                 use_indices: bool = False, contain_indices_path: str = None) -> None:
        super().__init__()
        self.img_type = img_type
        self.use_indices = use_indices
        self.image_size = image_size
        self.contain_indices_path = contain_indices_path
        self.split = split
        self.aug = aug

        if self.img_type == 'png':
            self.img_dir = os.path.join(contain_path, 'img_align_celeba_png')
        elif self.img_type == 'jpg':
            self.img_dir = os.path.join(contain_path, 'img_align_celeba')

        split_file = os.path.join(contain_path, f"{self.split}_attr_list.txt")
        df_split = pd.read_csv(split_file, sep=r"\s+", header=None, usecols=[0], names=['image_name'])
        self.image_names = df_split['image_name'].str.split('.').str[0].tolist()

        caption_path = os.path.join(contain_path, 'text', 'captions.json')
        with open(caption_path, 'r', encoding='utf-8') as f:
            raw_captions = json.load(f)
        self.captions_map = {key.split('.')[0]: value for key, value in raw_captions.items()}

        self.indices_map = None
        if self.use_indices:
            indices_path = os.path.join(contain_indices_path, self.split, "indices.json")
            print(f"indices_path is {indices_path}")
            if os.path.exists(indices_path):
                print("indices_path is existed")
                with open(indices_path, "r", encoding="utf-8") as f:
                    try:
                        self.indices_map = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"[VQGANDataset] Error {indices_path}: {e}")
                        self.indices_map = None
            else:
                print("indices_path is not existed")

        trans_list = [
            v2.Resize(max(image_size), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomCrop(image_size) if self.split == "train" and self.aug else v2.CenterCrop(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
        self.transform = v2.Compose(trans_list)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        base_name = self.image_names[idx]

        caption_entry = self.captions_map.get(base_name, {})

        if isinstance(caption_entry, dict):
            txt = caption_entry.get('overall_caption', '')
            # attr_captions = caption_entry.get('attribute_wise_captions', {})
        else:
            print(f"Warn: No caption for {base_name}")
            txt = ""
            # attr_captions = {}

        actual_fname = f"{base_name}.{self.img_type}"
        img_path = os.path.join(self.img_dir, actual_fname)

        img = self.transform(
            torchvision.io.read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        )

        sample = {
            "fname": base_name,
            "img": img,
            "txt": txt,
            # "attr_captions": attr_captions
        }

        if self.use_indices and self.indices_map is not None:
            inds = self.indices_map.get(base_name)
            sample["inds_tensor"] = torch.tensor(inds, dtype=torch.long)

        return sample
