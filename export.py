import argparse
from lit_vqgan import LitVQGAN


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to lightning checkpoint"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./vqgan_tokenizer",
        help="Output directory"
    )

    args = parser.parse_args()

    model = LitVQGAN.load_from_checkpoint(
        args.ckpt,
        map_location="cpu"
    ).eval()

    model.model.save_pretrained(
        args.output,
        safe_serialization=True
    )


if __name__ == "__main__":
    main()