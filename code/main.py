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


def crop_all_data(all_data):
    cropped_data = []
    for array in tqdm(all_data):
        array_ = [array[i:i + 400] for i in range(0, len(array), 100)][:-4]
        cropped_data += array_
    return cropped_data


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
    class0 = generate_class0(num=num)
    class1 = generate_class1(all_data, num=num)
    print("number of hard samples: %d" % len(collections))
    collections = collections[:num//2]
    class1 = class1[num-len(collections)] + collections
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
    model = Model()  # 载入模型
    all_data = load_all_data()  # 载入全部数据
    cropped_data = crop_all_data(all_data[:10000])  # 将数据按长度400进行切分
    sess, writer, saver = load_model(dirId)  # 加载模型权重
    x, y = generate_data_v1(all_data, num=108000)
    i = 0
    f = open("./models/"+dirId+"/logs/record.txt", "w+")
    while True:
        train(x, y, sess, model, saver, writer, dirId, train_step=200)
        collections = test(cropped_data, sess, model)
        f.write("%d - number of hard samples: %d\n" % (i,len(collections)))
        i += 1
        # x, y = generate_data_v2(all_data, collections, num=108000)
