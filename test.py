from src.srgan.configs import parse_args
from src.srgan.test import test_srgan

if __name__ == "__main__":
    opt = parse_args()
    test_srgan()