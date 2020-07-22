import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from data import all_files
from tqdm import tqdm


def find_path(filename):
    for path in all_files:
        filename_ = path.split("/")[-1]
        if filename_ == filename:
            return path


def draw(p, path):
    filename = path.split("/")[-1]
    try:
        table = pd.read_table(path, header=None, sep=" ")
        mean = np.mean(table[1])
        var = np.var(table[1])

        if var >= 0.02 or mean <= 4.0:  # 丢弃星等小于4的数据
            x = list(range(len(table)))
            y = table[1] - np.mean(table[1])
            time = list(table[0])
            time_ = time[1:] + [time[-1]]
            time_sub = np.array(time_) - np.array(time)
            lines = (np.array(np.where(time_sub >= 0.5)) + 1).tolist()[0]
            lines = [0] + lines  # 第一条线为开始处
            lines.append(len(table))  # 最后一条线为结尾处
            plt.clf()
            plt.xlabel('index')
            plt.ylabel('magnorm')
            plt.title(filename)
            plt.xlim(0, len(x))
            plt.scatter(x, y, s=2, marker='o')
            for line in lines:
                plt.axvline(x=line, ls="-", c="red")
            plt.savefig("../dataset/mean_lower_than_4/%s.jpg"%filename, pad_inches=0.0)
    except:
        pass

p = plt.figure(figsize=(15, 3), dpi=80)
for path in tqdm(all_files):
    draw(p, path)