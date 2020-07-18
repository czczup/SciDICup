import tensorflow as tf
from tqdm import tqdm
import numpy as np

def test(data, pathes, sess, model, num=200000):
    data = [data[i:i+num] for i in range(0, len(data), num)]
    pathes = [pathes[i:i+num] for i in range(0, len(pathes), num)]

    data_collection, path_collection = [], []
    for i in tqdm(range(len(data))):
        x = np.array(data[i])
        path = np.array((pathes[i]))
        argmaxs = sess.run(tf.argmax(model.output, axis=1), feed_dict={model.input: x, model.training: False})
        data_collection += x[argmaxs==0].tolist()
        path_collection += path[argmaxs==0].tolist()
    data_collection = [np.array(item) for item in data_collection]
    path_collection = [np.array(item) for item in path_collection]

    print(data_collection[0].shape)
    return data_collection, path_collection
