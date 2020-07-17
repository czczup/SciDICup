import tensorflow as tf
from functools import reduce
from operator import mul


class Model(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 400], name='input')

        with tf.name_scope("label"):
            self.label = tf.placeholder(tf.int32, [None], name='label')
            self.one_hot = tf.one_hot(indices=self.label, depth=2, name='one_hot')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.training = tf.placeholder(tf.bool)

        self.output = self.network(self.input)

        self.loss = self.get_loss(self.output, self.one_hot)

        self.batch_size = 10800
        self.amount = 10800
        self.step_per_epoch = self.amount // self.batch_size

        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.one_hot, 1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(1E-3).minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()


    def network(self, x):
        x = tf.layers.dense(x, units=200)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, units=100)
        x = tf.nn.relu(x)
        x = tf.layers.dense(x, units=2)
        return x


    def get_loss(self, output, onehot):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=onehot)
            loss = tf.reduce_mean(losses)
            tf.summary.scalar('loss', loss)
        return loss

    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        return num_params


if __name__ == '__main__':
    model = Model()
    print(model.get_num_params())