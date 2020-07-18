from data import all_files
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(all_files):
    data = []
    pathes = []
    for path in tqdm(all_files):
        try:
            table = pd.read_table(path, header=None, sep=" ")
            y = table[1] - np.mean(table[1])
            y = y.astype(np.float32)
            data.append(y)
            pathes.append(path)
        except:
            pass
    return np.array(data), np.array(pathes)

if __name__ == '__main__':
    data, filenames = load_data(all_files)
    np.save("../dataset/data.npy", data)
    np.save("../dataset/filenames.npy", filenames)
