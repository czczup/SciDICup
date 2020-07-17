from data import all_files
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(all_files):
    ys = []
    for path in tqdm(all_files):
        try:
            table = pd.read_table(path, header=None, sep=" ")
            y = table[1] - np.mean(table[1])
            y = y.astype(np.float32)
            ys.append(y)
        except:
            pass
    return np.array(ys)

if __name__ == '__main__':
    data = load_data(all_files)
    np.save("../dataset/data.npy", data)
