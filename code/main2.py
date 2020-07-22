from data import load_all_data
from tqdm import tqdm
from model import Model
from data import generate_flare_stars
from data import generate_microlensings
from train import train
from test import test
import numpy as np
import tensorflow as tf
import os


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


def generate_data(num=104000):
    class0 = generate_flare_stars(num=num)
    class1 = generate_microlensings(num=num)
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

    sess, writer, saver = load_model(dirId)  # 加载模型权重
    x, y = generate_data(num=20800)
    train(x, y, sess, model, saver, writer, dirId, train_step=200)
    while True:
        x, y = generate_data(num=20800)
        train(x, y, sess, model, saver, writer, dirId, train_step=50)

