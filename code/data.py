import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import random
from tqdm import tqdm


positions = {
    "ref_021_18450425-G0014_703997_4523": [[1980, 2100]],  # flare_star
    "ref_031_12010765-G0013_520170_227006": [[725, 825]],
    "ref_021_24310595-G0013_746468_14893": [[1100, 1300]],  # flare_star
    "ref_022_06300085-G0013_1548728_32012": [[1325, 1700]],  # flare_star
    "ref_022_15730595-G0013_391462_6330": [[2075, 2350]],  # flare_star
    "ref_022_15730595-G0013_481532_2590": [[300, 550], [800, 1000]],
    "ref_033_04500085-G0013_1485992_22874": [[425, 800]],  # flare_star
    "ref_033_04700255-G0013_25519_18341": [[1285, 1420]],  # flare_star
    "ref_024_12230255-G0013_303557_17479": [[625, 750]],  # flare_star
    "ref_033_15300085-G0013_1742006_16050": [[900, 1050]],  # flare_star
    "ref_024_12230255-G0013_393825_55632": [[1280, 1380]],  # flare_star
    "ref_033_16810765-G0013_482792_15702": [[725, 900]],  # flare_star
    "ref_043_04500085-G0013_2358479_2504": [[4050, 4300]],  # flare_star
    "ref_023_15730595-G0013_378567_3040": [[550, 750]],  # flare_star
    "ref_044_16280425-G0013_364820_9174": [[3950, 4100]],  # flare_star
    "ref_041_14110425-G0013_422384_26026": [[4200, 4550]],  # flare_star
    "ref_024_13500085-G0013_1397642_27534": [[300, 360]],
    "ref_043_18590595-G0013_709318_8662": [[3350, 3600]],
}


def get_train_filename():
    train_abnormal = pd.read_table("../dataset/AstroSet-v0.1/AstroSet/train_abnormal", header=None, sep=",")
    flare_stars, microlensings = [], []
    for index, value in train_abnormal.iterrows():
        if value[1] == 'flare star':
            flare_stars.append(value)
        elif value[1] == 'microlensing':
            microlensings.append(value)
    return flare_stars, microlensings


def crop_data(data):
    background, cropped_data = [], []
    for item in data:
        filename = item[0]
        for position in positions[filename]:
            left, right = position[0], position[1]
            path = "../dataset/AstroSet-v0.1/" + item[2] + "/" + filename
            table = pd.read_table(path, header=None, sep=" ")
            x = list(range(len(table)))
            y = table[1] - np.mean(table[1])
            x_, y_ = x[left:right], y[left:right]
            cropped_data.append(y_)
        background += (y[:positions[filename][0][0]].tolist() + y[positions[filename][-1][1]:].tolist())
    return background, cropped_data


def pool2d(A, kernel_size, stride, padding):
    A = np.reshape(A.tolist(), [A.shape[0], 1])
    A = np.pad(A, padding, mode='constant')
    # Window view of A
    output_shape = ((A.shape[0] - kernel_size[0]) // stride + 1, (A.shape[1] - kernel_size[1]) // stride + 1)
    A_w = as_strided(A, shape=output_shape + (kernel_size[0], kernel_size[1]),
                     strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)
    return A_w.mean(axis=(1, 2)).reshape([output_shape[0]])


def data_downsampling(data):
    results = []
    for item in data:
        if len(item) >= 100:
            temp = pool2d(np.array(item), [2, 1], 2, 0)
            results.append(temp)
    return results


def get_all_filename():
    dirs = os.listdir("../dataset/AstroSet-v0.1/AstroSet")
    dirs = [item for item in dirs if item[0] == '0']
    files = []
    for dir_ in dirs:
        filenames = os.listdir("../dataset/AstroSet-v0.1/AstroSet/" + dir_)
        filenames = ["../dataset/AstroSet-v0.1/AstroSet/" + dir_ + "/" + filename for filename in filenames if filename != 'abstract']
        files += filenames
    return files

def average(x):
    width = x.shape[0]
    x_move1 = x.reshape([width, 1])
    ones_5 = np.ones((5,)) / 5
    ones_10 = np.ones((10,)) / 10
    ones_20 = np.ones((20,)) / 20
    x_move5 = np.convolve(x, ones_5, mode="same").astype(np.float32).reshape([width, 1])
    x_move10 = np.convolve(x, ones_10, mode="same").astype(np.float32).reshape([width, 1])
    x_move20 = np.convolve(x, ones_20, mode="same").astype(np.float32).reshape([width, 1])
    x_output = np.stack([x_move1, x_move5, x_move10, x_move20], axis=-1)
    del x_move1, x_move5, x_move10, x_move20
    return x_output


def load_all_data():
    data = np.load("../dataset/data.npy")
    data = [item.reshape([item.shape[0], 1, 1]) for item in data]
    pathes = np.load("../dataset/pathes.npy")
    pathes = [np.array(item) for item in pathes]
    assert len(pathes) == len(data), "error"
    return data, pathes


flare_stars, microlensings = get_train_filename()
all_files = get_all_filename()
background1, cropped_flare_stars = crop_data(flare_stars)
background2, cropped_microlensings = crop_data(microlensings)
background_all = background1 + background2
background = []
for i in range(0, len(background_all), 10):
    array = np.array(background_all[i:i+10])
    array = array - np.mean(array)
    min_ = np.min(array)
    max_ = np.max(array)
    if min_ < -0.05 or max_ > 0.05:
        pass
    else:
        array = array.tolist()
        background += array
flare_stars_ = cropped_flare_stars + data_downsampling(cropped_flare_stars)
microlensings_ = cropped_microlensings + data_downsampling(cropped_microlensings)


def average(x):
    width = x.shape[0]
    x_move1 = x.reshape([width, 1])
    ones_5 = np.ones((5,)) / 5
    ones_10 = np.ones((10,)) / 10
    ones_20 = np.ones((20,)) / 20
    x_move5 = np.convolve(x, ones_5, mode="same").astype(np.float32).reshape([width, 1])
    x_move10 = np.convolve(x, ones_10, mode="same").astype(np.float32).reshape([width, 1])
    x_move20 = np.convolve(x, ones_20, mode="same").astype(np.float32).reshape([width, 1])
    x_output = np.stack([x_move1, x_move5, x_move10, x_move20], axis=-1)
    del x_move1, x_move5, x_move10, x_move20
    return x_output


def data_augmentation(data):
    start = random.randint(0, len(background) - 400) # 在背景中随机产生一个起点
    data_output = []
    for item in data:
        length = 400 - len(item) # 截取的背景的长度
        rand = random.randint(0, length) # 左侧和右侧至少为20
        rand2 = random.uniform(0.8, 1.2)
        left_part = np.array(background[start:start + rand]) * rand2
        left_part = left_part.tolist()
        right_part = np.array(background[start + rand:start + length]) * rand2
        right_part = right_part.tolist()
        item = np.array(item) * random.uniform(0.75, 1.25)
        item = [_+random.uniform(-0.05, +0.05) for _ in item]
        array = []
        array.extend(left_part)
        array.extend(item)
        array.extend(right_part)
        array = np.array(array).reshape([400, 1, 1])
        data_output.append(array)
    return data_output


def generate_flare_stars(num=104000): #TODO
    flare_stars = []
    for i in tqdm(range(num//24)):
        flare_stars += data_augmentation(flare_stars_)
    return flare_stars


def generate_microlensings(num=104000): #TODO
    microlensings = []
    microlensing_template = (pd.read_table("../dataset/microlensing.txt", header=None)[0] + 1) / 5
    microlensing_template = [item + random.uniform(-0.05, +0.05) for item in microlensing_template]
    for i in tqdm(range(num//24)):
        microlensings += data_augmentation(microlensings_) + \
                         data_augmentation(microlensings_) + \
                         data_augmentation([microlensing_template]*(24-14))
    return microlensings


def generate_class0(num=114000):
    class0 = []
    microlensing_template = (pd.read_table("../dataset/microlensing.txt", header=None)[0] + 1) / 5
    microlensing_template = [item + random.uniform(-0.05, +0.05) for item in microlensing_template]
    for i in tqdm(range(num//38)):
        flare_stars_aug = data_augmentation(flare_stars_) # 24
        microlensings_aug = data_augmentation(microlensings_) + data_augmentation([microlensing_template]*(7)) # 7*2
        class0.extend(flare_stars_aug + microlensings_aug)
    class0 = [np.array(item) for item in class0]
    return class0


def generate_class1(all_data, num=114000):
    def crop_data(y):
        length = len(y)
        data = []
        for i in range(10):
            rand = random.randint(0, length-400)
            data.append(y[rand:rand+400,:,:])
        return data
    files = random.sample(all_data, num//10)
    class1 = []
    for file in tqdm(files):
        class1 += crop_data(file)
    class1 = [np.array(item) for item in class1]
    return class1


if __name__ == '__main__':
    data, pathes = load_all_data()
    print(len(data), len(pathes))
    count = 0
    for index, item in enumerate(data):

        if item.shape[0] == 0:
            print(item.shape, pathes[index])
            count += 1
    print(count)

