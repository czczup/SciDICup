from data import load_all_data
from tqdm import tqdm
from model import Model
from data import generate_class0
from data import generate_class1
from train import train
from test import test
import numpy as np
import tensorflow as tf
import os


def crop_all_data(data, pathes):
    cropped_data = []
    new_pathes = []
    index = 0
    for array in tqdm(data):
        array_ = [array[i:i+400,:,:] for i in range(0, len(array), 100)][:-4]
        path = [pathes[index]] * len(array_)
        cropped_data += array_
        new_pathes += path
        index += 1
    return cropped_data, new_pathes


def load_model(dirId):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("models/" + dirId + "/logs/", sess.graph)
    last_file = tf.train.latest_checkpoint("models/" + dirId)
    var_list = [var for var in tf.global_variables() if "moving" in var.name]
    var_list += [var for var in tf.global_variables() if "global_step" in var.name]
    var_list += tf.trainable_variables()
    saver = tf.train.Saver(var_list=var_list, max_to_keep=100)
    if last_file:
        tf.logging.info('Restoring model from {}'.format(last_file))
        saver.restore(sess, last_file)
    return sess, writer, saver


def average(x):
    x_move1 = x.reshape([400, 1])
    ones_5 = np.ones((5,)) / 5
    ones_10 = np.ones((10,)) / 10
    ones_20 = np.ones((20,)) / 20
    x_move5 = np.convolve(x, ones_5, mode="same").astype(np.float32).reshape([400, 1])
    x_move10 = np.convolve(x, ones_10, mode="same").astype(np.float32).reshape([400, 1])
    x_move20 = np.convolve(x, ones_20, mode="same").astype(np.float32).reshape([400, 1])
    x_output = np.stack([x_move1, x_move5, x_move10, x_move20], axis=-1)
    del x_move1, x_move5, x_move10, x_move20
    return x_output


def generate_data_v1(all_data, num):
    class0 = generate_class0(num=num)
    class1 = generate_class1(all_data, num=num)
    class0 = [[item, 0] for item in class0]
    class1 = [[item, 1] for item in class1]
    train_data = class0 + class1
    np.random.shuffle(train_data)
    x = [item[0] for item in train_data]
    y = [item[1] for item in train_data]
    return x, y


def generate_data_v2(all_data, collections, num):
    class0 = generate_class0(num=num) # [None, 400, 1 , 4]
    class1 = generate_class1(all_data, num=num)
    collections = collections[-num//4:]
    class1 = class1[0:num-len(collections)] + collections
    class0 = [[item, 0] for item in class0]
    class1 = [[item, 1] for item in class1]
    train_data = class0 + class1
    np.random.shuffle(train_data)
    x = [item[0] for item in train_data]
    y = [item[1] for item in train_data]
    return x, y


if __name__ == '__main__':
    deviceId = input("device id: ")
    os.environ["CUDA_VISIBLE_DEVICES"] = deviceId
    dirId = input("dir id: ")
    mode = input("mode: ")
    model = Model()  # 载入模型
    data, pathes = load_all_data()  # 载入全部数据

    sess, writer, saver = load_model(dirId)  # 加载模型权重
    x, y = generate_data_v1(data, num=23250)
    cropped_data, new_pathes = crop_all_data(data, pathes)  # 将数据按长度400进行切分

    if mode == 'train':
        i = 0
        # old_data_collection, old_path_collecton = [], []
        while True:
            f = open("./models/" + dirId + "/logs/record.txt", "a+")
            if i == 0:
                train(x, y, sess, model, saver, writer, dirId, train_step=200)
            else:
                train(x, y, sess, model, saver, writer, dirId, train_step=50)
            data_collection, path_collection = test(cropped_data, new_pathes, sess, model, num=22800 * 2)
            # old_data_collection += data_collection
            # old_data_collection = old_data_collection[-23250:]
            # x, y = generate_data_v2(data, old_data_collection, num=23250)
            x, y = generate_data_v1(data, num=22800)
            f.write("%d - number of hard samples: %d\n" % (i, len(data_collection)))
            f.write("%d - percent of hard samples: %.2f%%\n" % (i, len(data_collection) / len(cropped_data) * 100))
            f.close()
            np.save("./models/"+dirId+"/%d-data.npy"%i, np.array(data_collection))
            np.save("./models/"+dirId+"/%d-pathes.npy"%i, np.array(path_collection))
            print("%d - number of hard samples: %d" % (i, len(data_collection)))
            print("%d - percent of hard samples: %.2f%%" % (i, len(data_collection) / len(cropped_data) * 100))
            i += 1
    elif mode == 'test':
        data_collection, path_collection = test(cropped_data, new_pathes, sess, model, num=22800 * 2)
        print("number of hard samples: %d" % (len(data_collection)))
        print("percent of hard samples: %.2f%%" % (len(data_collection) / len(cropped_data) * 100))
        np.save("./models/"+dirId+"/output-data.npy", np.array(data_collection))
        np.save("./models/"+dirId+"/output-pathes.npy", np.array(path_collection))
