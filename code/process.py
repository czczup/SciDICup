from data import all_files
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(all_files):
    data = []
    pathes = []
    count = 0
    for path in tqdm(all_files):
        try:
            table = pd.read_table(path, header=None, sep=" ")
            mean = np.mean(table[1])
            var = np.var(table[1])
            if var <= 0.02 and mean >= 4.0: # 丢弃星等小于4的数据
                y = table[1] - mean # 平移数据
                time = list(table[0])
                time_ = time[1:] + [time[-1]]
                time_sub = np.array(time_) - np.array(time)
                lines = (np.array(np.where(time_sub >= 0.5)) + 1).tolist()[0]
                lines = [0] + lines  # 第一条线为开始处
                lines.append(len(table))  # 最后一条线为结尾处
                y_output = []
                for i in range(len(lines)-1): # 数据段的数量比分段线少1
                    y_part = y[lines[i]:lines[i+1]]
                    if len(y_part) < 300: # 长度小于300的数据段直接丢弃
                        pass
                    else:
                        y_part = y_part[80: -80] # 长度大于300的数据段掐头去尾
                        y_part = y_part - np.mean(y_part) # 平移
                        y_output += y_part.tolist()
                if len(y_output) >= 400:
                    y_output = np.array(y_output).astype(np.float32)
                    y_output[y_output >= 1.5] = 0  # 去除极端噪声
                    y_output[y_output <= -1.5] = 0  # 去除极端噪声
                    data.append(y_output)
                    pathes.append(path)
                else: count += 1
        except:
            print(path)
    print(count)
    return np.array(data), np.array(pathes)



def data_mean_lower_than_4(all_files):
    count = 0
    f = open("../dataset/mean_lower_than_4.txt", "w+")
    for path in tqdm(all_files):
        try:
            table = pd.read_table(path, header=None, sep=" ")
            mean = np.mean(table[1])
            if mean < 4.0: # 丢弃星等小于4的数据
                f.write("%s\n"%path)
                count += 1
        except:
            pass
    print(count)


if __name__ == '__main__':
    # data_mean_lower_than_4(all_files)
    data, pathes = load_data(all_files)
    print(data.shape)
    np.save("../dataset/data.npy", data)
    np.save("../dataset/pathes.npy", pathes)
