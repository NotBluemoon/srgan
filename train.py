from src.srgan.configs import parse_train_args
from src.srgan.train import train_srgan

if __name__ == "__main__":
    opt = parse_train_args()
    train_srgan(opt)