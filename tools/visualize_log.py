import numpy as np
import matplotlib.pyplot as plt
import re
import os


if __name__ == "__main__":
    logs = ['/home/zhiyum/FCOS/training_dir/fcos_imprv_R_50_FPN_1x_0.5/',
            '/home/mengqinj/capstone/output/stage2',
            '/home/mengqinj/capstone/output/stage3']
    colors = ['r', 'b', 'g']

    fig, ax = plt.subplots()

    for i, log in enumerate(logs):
        iters = []
        loss = []
        with open(os.path.join(log, 'log.txt'), 'r') as f:
            for line in f.readlines():
                find_iter = re.findall(r' iter: \d+ ', line)
                if len(find_iter) > 0:
                    iters.append(int(find_iter[0].split()[1]))
                    loss.append(float(re.findall(r' loss: \d+\.\d+ ', line)[0].split()[1]))
        ax.plot(iters, loss, colors[i]+'-')

    plt.show()

    # print(iters)
    # print(loss)
