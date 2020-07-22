import tensorflow as tf
from functools import reduce
from operator import mul


class Model(object):
    def __init__(self):
        self.input = tf.placeholder(tf.float32, [None, 400, 1, 1], name='input')

        with tf.name_scope("label"):
            self.label = tf.placeholder(tf.int32, [None], name='label')
            self.one_hot = tf.one_hot(indices=self.label, depth=2, name='one_hot')

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.training = tf.placeholder(tf.bool)

        self.output = self.network(self.input)

        self.loss = self.get_loss(self.output, self.one_hot)

        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.one_hot, 1))
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('train_accuracy', self.accuracy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            learning_rate = tf.train.exponential_decay(1E-3, global_step=self.global_step, decay_steps=200, decay_rate=0.95)

            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)

        self.merged = tf.summary.merge_all()

    def conv2d(self, x, input_filters, output_filters, kernel, strides=1, padding="SAME"):
        with tf.name_scope('conv'):
            shape = [kernel, 1, input_filters, output_filters]
            weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
            return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding=padding, name='conv')

    def residual(self, x, num_filters, strides, with_shortcut=False):
        with tf.name_scope('residual'):
            conv1 = self.conv2d(x, num_filters[0], num_filters[1], kernel=1, strides=strides)
            bn1 = tf.layers.batch_normalization(conv1, axis=3, training=self.training, momentum=0.95)
            relu1 = tf.nn.relu(bn1)
            conv2 = self.conv2d(relu1, num_filters[1], num_filters[2], kernel=3)
            bn2 = tf.layers.batch_normalization(conv2, axis=3, training=self.training, momentum=0.95)
            relu2 = tf.nn.relu(bn2)
            conv3 = self.conv2d(relu2, num_filters[2], num_filters[3], kernel=1)
            bn3 = tf.layers.batch_normalization(conv3, axis=3, training=self.training, momentum=0.95)
            if with_shortcut:
                shortcut = self.conv2d(x, num_filters[0], num_filters[3], kernel=1, strides=strides)
                bn_shortcut = tf.layers.batch_normalization(shortcut, axis=3, training=self.training, momentum=0.95)
                residual = tf.nn.relu(bn_shortcut + bn3)
            else:
                residual = tf.nn.relu(x + bn3)
        return residual

    def global_average_pooling(self, x):
        return tf.reduce_mean(x, [1, 2])

    def network(self, x):

        x = self.conv2d(x, 1, 8, 7, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.conv2d(x, 8, 16, 3, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.conv2d(x, 16, 32, 3, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.conv2d(x, 32, 64, 3, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.conv2d(x, 64, 128, 3, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.conv2d(x, 128, 256, 3, 2)
        x = tf.layers.batch_normalization(x, axis=3, training=self.training, momentum=0.95)
        x = tf.nn.relu(x)
        print(x)
        x = self.global_average_pooling(x)
        print(x)
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