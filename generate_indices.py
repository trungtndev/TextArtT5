import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

from src.datamodule.dataset import CelebADataset
from src.vqgan import VQGAN, EMAVQGAN
import setproctitle

setproctitle.setproctitle("ExtractIndicesGumbelVQ")

@torch.no_grad()
def extract_indices_from_batch(model, images: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        images = images.to(model.device)
        h = model.encoder(images)
        h = model.quant_conv(h)
        _, _, info = model.quantize(h)
        # info: (perplexity, min_encodings, indices)
        indices = info[-1]
        # indices shape: (B, H, W) or (B, HW)
        if indices.dim() not in (2, 3):
            raise ValueError(f"Unexpected indices shape: {indices.shape}")

        return indices.cpu().numpy().astype(np.int64)



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",type=str,required=False,default="./vqgan_tokenizer",help="Path to VQGAN checkpoint"
    )

    parser.add_argument(
        "--data_root",type=str,default="data",
    )

    parser.add_argument(
        "--type", type=str,default="emavq",
    )

    parser.add_argument(
        "--split",type=str,required=True,
    )

    parser.add_argument(
        "--incides_data_root",type=str,required=True,help="Output json path"
    )

    parser.add_argument(
        "--batch_size",type=int,default=32,
    )

    parser.add_argument(
        "--num_workers",type=int,default=12,
    )
    parser.add_argument(
        "--image_height", type=int, default=192,
    )

    parser.add_argument(
        "--image_width",type=int,default=512,
    )

    args = parser.parse_args()
    JSON_OUTPUT_PATH = f"{args.incides_data_root}/{args.split}/indices.json"


    print(f"Loading VQGAN checkpoint from {args.ckpt}")
    if args.type == "emavq":
        model = EMAVQGAN.from_pretrained(args.ckpt)
    elif args.type == "vq":
        model = VQGAN.from_pretrained(args.ckpt)

    model = model.eval()
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()

    data_dir = os.path.join(args.data_root, args.split)
    print(f"Loading dataset from {data_dir}")

    dataset = CelebADataset(
        contain_path=args.data_root, img_type="png", image_size=(args.image_height, args.image_width),
        split=args.split, aug=False, use_indices=False
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_indices = []
    all_fnames = []

    print(f"Encoding split '{args.split}' with {len(dataset)} images...")
    for batch in tqdm(loader):
        fnames = batch["fname"]
        images = batch["img"]

        indices_np = extract_indices_from_batch(model, images)  # (B, H, W) or (B, HW)
        assert indices_np.shape[0] == len(fnames)

        all_indices.append(indices_np)
        all_fnames.extend(fnames)

    if len(all_indices) == 0:
        print("No samples found; nothing to save.")
        return

    all_indices = np.concatenate(all_indices, axis=0)  # (N, H, W) or (N, HW)

    out_dir = os.path.dirname(JSON_OUTPUT_PATH) or "."
    os.makedirs(out_dir, exist_ok=True)

    data = {}
    for fname, inds in zip(all_fnames, all_indices):
        data[fname] = inds.tolist()

    with open(JSON_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Saved indices to {JSON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
