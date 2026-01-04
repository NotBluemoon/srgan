from collections import deque
import math, csv, os
from pathlib import Path
# import matplotlib
# import matplotlib.pyplot as plt


class TrainLogger:
    def __init__(self, opt):
        self.loss_D_hist = deque(maxlen=opt.checkpoint_interval)
        self.loss_G_hist = deque(maxlen=opt.checkpoint_interval)
        self.content_loss_hist = deque(maxlen=opt.checkpoint_interval)
        self.adversarial_loss_hist = deque(maxlen=opt.checkpoint_interval)

        project_root =  Path(__file__).resolve().parents[3]
        pretrain_log_dir = project_root / 'outputs' / 'logs'
        pretrain_log_path = project_root / 'outputs' / 'logs' / 'train_log.csv'

        if not os.path.exists(pretrain_log_path):
            os.makedirs(pretrain_log_dir, exist_ok=True)
            with open(pretrain_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "avg_loss_D", 'avg_loss_G', "avg_content_loss", 'avg_adversarial_loss', 'psnr'
                ])

    def update(self, opt, step, loss_D, content_loss, adversarial_loss, loss_G):
        loss_D = float(loss_D)
        content_loss = float(content_loss)
        adversarial_loss = float(adversarial_loss)
        loss_G = float(loss_G)

        self.loss_D_hist.append(loss_D)
        self.loss_G_hist.append(loss_G)
        self.content_loss_hist.append(content_loss)
        self.adversarial_loss_hist.append(adversarial_loss)

        if step % opt.checkpoint_interval == 0:
            project_root =  Path(__file__).resolve().parents[3]
            pretrain_log_path = project_root / 'outputs' / 'logs' / 'train_log.csv'

            avg_loss_D = sum(self.loss_D_hist) / len(self.loss_D_hist)
            avg_loss_G = sum(self.loss_G_hist) / len(self.loss_G_hist)
            avg_content_loss = sum(self.content_loss_hist) / len(self.content_loss_hist)
            avg_adversarial_loss = sum(self.adversarial_loss_hist) / len(self.adversarial_loss_hist)
            psnr = 10 * math.log10(1.0 / avg_loss_G)

            # append to log csv file
            with open(pretrain_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, avg_loss_D, avg_loss_G, avg_content_loss, avg_adversarial_loss, psnr
                ])




