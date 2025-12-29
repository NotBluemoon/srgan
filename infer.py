from src.srgan.configs import parse_args
from src.srgan.infer import infer

if __name__ == "__main__":
    opt = parse_args()
    infer(opt)