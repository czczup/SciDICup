import time

def train(x, y, sess, model, saver, writer, dirId, train_step=100):
    for i in range(train_step):
        time1 = time.time()
        step = sess.run(model.global_step)
        _, loss_ = sess.run([model.optimizer, model.loss], feed_dict={model.input: x,
                                                                      model.label: y,
                                                                      model.training: True})
        acc_train, summary = sess.run([model.accuracy, model.merged], feed_dict={model.input: x,
                                                                                 model.label: y,
                                                                                 model.training: False})
        print('[step %d]: loss: %.6f, time: %.3fs, acc: %.3f'%(step, loss_, (time.time() - time1), acc_train))
        writer.add_summary(summary, step)

    print("Save the model Successfully")
    saver.save(sess, "models/" + dirId + "/model.ckpt", global_step=step)


