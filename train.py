from src.srgan.configs import parse_args
from src.srgan.train import train_srgan

if __name__ == "__main__":
    opt = parse_args()
    train_srgan(opt)