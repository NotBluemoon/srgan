from collections import deque
import math, csv, os
from pathlib import Path
# import matplotlib
# import matplotlib.pyplot as plt


class PretrainLogger:
    def __init__(self, opt):
        self.loss_hist = deque(maxlen=opt.checkpoint_interval)
        self.step = 0

        project_root =  Path(__file__).resolve().parents[3]
        pretrain_log_path = project_root / 'outputs' / 'logs' / 'pretrain_log.csv'

        if not os.path.exists(pretrain_log_path):
            with open(pretrain_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "avg_loss", "rmse", 'psnr'
                ])

    def update(self, opt, step, loss):
        loss = float(loss)
        self.loss_hist.append(loss)

        if step % opt.checkpoint_interval == 0:
            project_root =  Path(__file__).resolve().parents[3]
            pretrain_log_path = project_root / 'outputs' / 'logs' / 'pretrain_log.csv'

            avg = sum(self.loss_hist) / len(self.loss_hist)
            rmse = math.sqrt(avg)
            psnr = 10 * math.log10(1.0 / avg)

            # append to log csv file
            with open(pretrain_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step, avg, rmse, psnr,
                ])

    # def plot_graph (self, opt):
    #
    #     matplotlib.use('Agg')
    #     steps,


