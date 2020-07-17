import tensorflow as tf
from tqdm import tqdm
import numpy as np

def test(data, sess, model, num=200000):
    xs = [data[i:i+num] for i in range(0, len(data), num)]
    collections = []
    for x in tqdm(xs):
        x = np.array(x)
        argmaxs = sess.run(tf.argmax(model.output, axis=1), feed_dict={model.input: x, model.training: False})
        collections += x[argmaxs==0].tolist()
    return collections
