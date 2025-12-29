import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume",  action="store_true", help="Whether to use checkpoints/existing models or delete old models and start from scratch")
    parser.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    parser.add_argument("--dataset_name", type=str, default="DIV2K", help="Name of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adam: learning rate, defaulted to 0.0001 to match SRGAN paper")
    parser.add_argument("--b1", type=float, default=0.90, help="Adam: decay of first order momentum of gradient, defaulted to 0.90 to match SRGAN paper") # matches SRGAN paper
    parser.add_argument("--b2", type=float, default=0.999, help="Adam: decay of second order momentum of gradient")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--checkpoint_interval", type=int, default=5000, help="Interval between model checkpoints")
    parser.add_argument("--pretrain_steps", type=int, default=200, help="Target generator pretraining steps to achieve")
    parser.add_argument("--train_steps", type=int, default=2_000_000, help="Target training steps to achieve, learning rate will be switched at train_steps/2")
    parser.add_argument("--num_res_blocks", type=int, default=16, help="Number of residual blocks in SRGAN generator")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of input image channels")

    # infer arguments
    parser.add_argument("--in_dir", type=str, default=None, help="Path to input directory that contains images that need to be super-resolved")

    opt = parser.parse_args()

    if opt.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(opt.device)

    opt.device = device

    return opt

